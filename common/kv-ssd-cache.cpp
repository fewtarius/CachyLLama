// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// SSD-Backed KV Cache with Hot/Warm/Cold Tiering
// Per-checkpoint file storage with ring buffer eviction.

#include "kv-ssd-cache.h"

#include "log.h"

#include <cstring>
#include <cinttypes>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#ifdef __linux__
#include <sys/sysinfo.h>
#endif

// Magic numbers
static const uint32_t KV_SSD_MAGIC_INDEX = 0x4B564944; // "KVID"
static const uint32_t KV_SSD_MAGIC_REC   = 0x4B565243; // "KVRC"
static const uint32_t KV_SSD_VERSION     = 2;           // v2 = per-file format
static const size_t   KV_SSD_INDEX_SIZE  = 4096;

// =============================================================================
// Internal helpers
// =============================================================================

static uint64_t now_ms() {
    auto tp = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        tp.time_since_epoch()).count();
}

// FNV-1a hash for token sequences
uint64_t kv_ssd_hash_tokens(const uint32_t* tokens, size_t count) {
    if (!tokens || count == 0) return 0;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < count; i++) {
        uint32_t v = tokens[i];
        h ^= (uint64_t)(v & 0xFF);
        h *= 1099511628211ULL;
        h ^= (uint64_t)((v >> 8) & 0xFF);
        h *= 1099511628211ULL;
        h ^= (uint64_t)((v >> 16) & 0xFF);
        h *= 1099511628211ULL;
        h ^= (uint64_t)((v >> 24) & 0xFF);
        h *= 1099511628211ULL;
    }
    return h;
}

static size_t get_available_ram() {
#ifdef __linux__
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
#endif
    return 8ULL * 1024 * 1024 * 1024;
}

// Write exactly `count` bytes to fd at offset.
static bool pwrite_all(int fd, const void* buf, size_t count, off_t offset) {
    const char* ptr = (const char*)buf;
    size_t remaining = count;
    off_t off = offset;
    while (remaining > 0) {
        ssize_t n = pwrite(fd, ptr, remaining, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        ptr += n;
        off += n;
        remaining -= (size_t)n;
    }
    return true;
}

// Read exactly `count` bytes from fd at offset.
static bool pread_all(int fd, void* buf, size_t count, off_t offset) {
    char* ptr = (char*)buf;
    size_t remaining = count;
    off_t off = offset;
    while (remaining > 0) {
        ssize_t n = pread(fd, ptr, remaining, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false; // unexpected EOF
        ptr += n;
        off += n;
        remaining -= (size_t)n;
    }
    return true;
}

// Get checkpoint file path: {model_dir}/ckpt-{id}.bin
static std::string ckpt_path(const kv_ssd_cache* c, uint64_t id) {
    return c->model_dir + "/ckpt-" + std::to_string(id) + ".bin";
}

// Get index file path: {model_dir}/index.bin
static std::string index_path(const kv_ssd_cache* c) {
    return c->model_dir + "/index.bin";
}

// =============================================================================
// Index persistence
// =============================================================================

// Write the index file (next_id + compat_hash).
static bool write_index_file(kv_ssd_cache* c) {
    kv_ssd_index_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = KV_SSD_MAGIC_INDEX;
    hdr.version = KV_SSD_VERSION;
    hdr.next_id = c->next_id;
    hdr.compat_hash = c->compat_hash;

    std::string path = index_path(c);
    int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOG_WRN("SSD cache: failed to write index: %s\n", strerror(errno));
        return false;
    }
    bool ok = pwrite_all(fd, &hdr, sizeof(hdr), 0);
    fsync(fd);
    close(fd);
    return ok;
}

// Read the index file. Returns false if not found or invalid.
static bool read_index_file(kv_ssd_cache* c) {
    std::string path = index_path(c);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;

    kv_ssd_index_header hdr;
    bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
    close(fd);
    if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX || hdr.version != KV_SSD_VERSION) return false;

    c->next_id = hdr.next_id;
    if (c->compat_hash == 0) c->compat_hash = hdr.compat_hash;
    return true;
}

// Load a single checkpoint file into the in-memory index.
static bool load_checkpoint_file(kv_ssd_cache* c, const std::string& filepath) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) return false;

    kv_ssd_record rec;
    bool ok = pread_all(fd, &rec, sizeof(rec), 0);
    close(fd);
    if (!ok || rec.magic != KV_SSD_MAGIC_REC) return false;

    // Build index entry
    kv_ssd_checkpoint ckpt;
    ckpt.id = rec.id;
    ckpt.slot_id = rec.slot_id;
    ckpt.pos_min = rec.pos_min;
    ckpt.pos_max = rec.pos_max;
    ckpt.n_tokens = rec.n_tokens;
    ckpt.turn_created = rec.turn_created;
    ckpt.turn_id = rec.turn_created;
    ckpt.token_hash = rec.token_hash;
    ckpt.compat_hash = rec.compat_hash;
    ckpt.token_count = (size_t)rec.token_count;
    ckpt.tier = KV_TIER_COLD;
    ckpt.data_size = (size_t)rec.data_size;
    ckpt.last_access = 0;
    ckpt.access_count = 0;

    if (rec.token_count > 0) {
        ckpt.token_prefix.assign(rec.token_prefix, rec.token_prefix + rec.token_count);
    }

    c->index[rec.id] = ckpt;
    c->slot_latest[rec.slot_id] = rec.id;
    return true;
}

// Scan the model directory for ckpt-*.bin files and load their headers.
static size_t scan_checkpoint_files(kv_ssd_cache* c) {
    DIR* dir = opendir(c->model_dir.c_str());
    if (!dir) return 0;

    size_t loaded = 0;
    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        // Match ckpt-*.bin
        if (strncmp(ent->d_name, "ckpt-", 5) != 0) continue;
        size_t dlen = strlen(ent->d_name);
        if (dlen < 5 || strcmp(ent->d_name + dlen - 4, ".bin") != 0) continue;

        std::string filepath = c->model_dir + "/" + ent->d_name;
        if (load_checkpoint_file(c, filepath)) {
            loaded++;
        } else {
            LOG_WRN("SSD cache: warning: failed to load %s\n", filepath.c_str());
        }
    }
    closedir(dir);
    return loaded;
}

// Delete a checkpoint file from disk.
static bool delete_checkpoint_file(kv_ssd_cache* c, uint64_t id) {
    std::string path = ckpt_path(c, id);
    if (unlink(path.c_str()) != 0 && errno != ENOENT) {
        LOG_WRN("SSD cache: failed to delete %s: %s\n", path.c_str(), strerror(errno));
        return false;
    }
    return true;
}

// =============================================================================
// Ring buffer eviction (internal, caller must hold mutex)
// =============================================================================

// Evict oldest cold checkpoints until count <= max_cold_checkpoints.
// Deletes files from disk and removes from in-memory index.
static void ring_buffer_evict(kv_ssd_cache* c) {
    int max_cold = c->config.max_cold_checkpoints;
    if (max_cold <= 0) return;

    // Count cold entries
    size_t cold_count = 0;
    for (const auto& [id, ckpt] : c->index) {
        if (ckpt.tier == KV_TIER_COLD) cold_count++;
    }

    if ((int)cold_count <= max_cold) return;

    // Sort cold entries by turn_created (oldest first)
    std::vector<std::pair<uint64_t, uint32_t>> cold_by_age; // (id, turn_created)
    for (const auto& [id, ckpt] : c->index) {
        if (ckpt.tier == KV_TIER_COLD) {
            cold_by_age.emplace_back(id, ckpt.turn_created);
        }
    }
    std::sort(cold_by_age.begin(), cold_by_age.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    int to_evict = (int)cold_count - max_cold;
    int evicted = 0;
    for (int i = 0; i < to_evict && i < (int)cold_by_age.size(); i++) {
        uint64_t id = cold_by_age[i].first;
        auto ckpt_it = c->index.find(id);
        if (ckpt_it == c->index.end()) continue;

        uint32_t slot = ckpt_it->second.slot_id;

        // Remove from slot_latest if this was the latest for that slot
        auto slot_it = c->slot_latest.find(slot);
        if (slot_it != c->slot_latest.end() && slot_it->second == id) {
            c->slot_latest.erase(slot_it);
        }

        // Delete file from disk
        delete_checkpoint_file(c, id);

        // Remove from in-memory index
        c->index.erase(ckpt_it);
        evicted++;
    }

    if (evicted > 0) {
        LOG_INF("SSD cache: ring buffer evicted %d checkpoints (limit=%d, remaining=%zu)\n",
                evicted, max_cold, c->index.size());
    }
}

// =============================================================================
// Tier management (internal, caller must hold mutex)
// =============================================================================

// Demote LRU hot entries to warm until hot_bytes <= budget.
static void demote_hot_to_warm(kv_ssd_cache* c) {
   while (c->hot_bytes > c->config.hot_ram_bytes && !c->hot_cache.empty()) {
       uint64_t lru_id = 0;
       uint64_t lru_time = UINT64_MAX;

       for (const auto& [id, ckpt] : c->index) {
            if (ckpt.tier == KV_TIER_HOT && ckpt.last_access < lru_time) {
                lru_time = ckpt.last_access;
                lru_id = id;
            }
       }
        if (lru_id == 0) break;

        auto it = c->hot_cache.find(lru_id);
        if (it == c->hot_cache.end()) {
            c->index[lru_id].tier = KV_TIER_COLD;
            continue;
        }

        size_t sz = it->second.size();
        c->warm_cache[lru_id] = std::move(it->second);
        c->hot_cache.erase(it);
        c->hot_bytes -= sz;
        c->warm_bytes += sz;
        c->index[lru_id].tier = KV_TIER_WARM;

        LOG_INF("SSD cache: demoted checkpoint %lu hot->warm (hot=%zu MiB, warm=%zu MiB)\n",
                (unsigned long)lru_id, c->hot_bytes / 1024 / 1024, c->warm_bytes / 1024 / 1024);
    }
}

// Demote LRU warm entries to cold (data stays on disk).
static void demote_warm_to_cold(kv_ssd_cache* c) {
    while (c->warm_bytes > c->config.warm_ram_bytes && !c->warm_cache.empty()) {
        uint64_t lru_id = 0;
        uint64_t lru_time = UINT64_MAX;

        for (const auto& [id, ckpt] : c->index) {
            if (ckpt.tier == KV_TIER_WARM && ckpt.last_access < lru_time) {
                lru_time = ckpt.last_access;
                lru_id = id;
            }
        }
        if (lru_id == 0) break;

        auto it = c->warm_cache.find(lru_id);
        if (it == c->warm_cache.end()) {
            c->index[lru_id].tier = KV_TIER_COLD;
            continue;
        }

        size_t sz = it->second.size();
        c->warm_cache.erase(it);
        c->warm_bytes -= sz;
        c->index[lru_id].tier = KV_TIER_COLD;

        LOG_INF("SSD cache: demoted checkpoint %lu warm->cold (warm=%zu MiB)\n",
                (unsigned long)lru_id, c->warm_bytes / 1024 / 1024);
    }
}

// Make room in hot tier for `needed` bytes.
static void make_room_hot(kv_ssd_cache* c, size_t needed) {
    while (c->hot_bytes + needed > c->config.hot_ram_bytes && !c->hot_cache.empty()) {
        demote_hot_to_warm(c);
        if (c->hot_bytes + needed <= c->config.hot_ram_bytes) break;
        demote_warm_to_cold(c);
        demote_hot_to_warm(c);
        if (c->hot_bytes + needed > c->config.hot_ram_bytes) break;
    }
}

// Promote a checkpoint to hot tier (load from SSD file if needed).
static bool promote_to_hot(kv_ssd_cache* c, uint64_t id) {
    auto it = c->index.find(id);
    if (it == c->index.end()) return false;

    auto& ckpt = it->second;

    // Already hot
    if (ckpt.tier == KV_TIER_HOT) {
        ckpt.last_access = now_ms();
        ckpt.access_count++;
        return true;
    }

    // If warm, move to hot
    if (ckpt.tier == KV_TIER_WARM) {
        auto wit = c->warm_cache.find(id);
        if (wit != c->warm_cache.end()) {
            // Move data out BEFORE make_room_hot. make_room_hot calls
            // demote_warm_to_cold with prefer_conv_hash, which can evict
            // THIS entry from warm_cache — invalidating wit. Moving
            // data first prevents use-after-erase segfault.
            auto data = std::move(wit->second);
            size_t sz = data.size();
            c->warm_cache.erase(wit);
            c->warm_bytes -= sz;

            make_room_hot(c, sz);

            c->hot_cache[id] = std::move(data);
            c->hot_bytes += sz;
            ckpt.tier = KV_TIER_HOT;
            ckpt.last_access = now_ms();
            ckpt.access_count++;
            return true;
        }
    }

    // Cold - read from SSD file
    std::string filepath = ckpt_path(c, id);
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_WRN("SSD cache: checkpoint file %s not found\n", filepath.c_str());
        // File was deleted (ring buffer evicted) but index still had entry - clean up
        c->index.erase(id);
        return false;
    }

    // Read record header
    kv_ssd_record rec;
    if (!pread_all(fd, &rec, sizeof(rec), 0) || rec.magic != KV_SSD_MAGIC_REC || rec.id != id) {
        close(fd);
        return false;
    }

    size_t data_size = (size_t)rec.data_size;
    make_room_hot(c, data_size);

    // Read checkpoint data (follows the record header)
    std::vector<uint8_t> data(data_size);
    if (!pread_all(fd, data.data(), data_size, (off_t)sizeof(kv_ssd_record))) {
        close(fd);
        return false;
    }
    close(fd);

    c->hot_cache[id] = std::move(data);
    c->hot_bytes += data_size;
    ckpt.tier = KV_TIER_HOT;
    ckpt.last_access = now_ms();
    ckpt.access_count++;
    c->stats_loads++;

    LOG_INF("SSD cache: promoted checkpoint %lu cold->hot (%zu MiB)\n",
            (unsigned long)id, data_size / 1024 / 1024);
    return true;
}

// =============================================================================
// Public API
// =============================================================================

kv_ssd_cache* kv_ssd_init(const char* path, const kv_ssd_config* cfg, uint64_t conv_hash) {
    if (!path) return nullptr;

    kv_ssd_cache* c = new kv_ssd_cache();
    if (cfg) c->config = *cfg;
    c->conv_hash = conv_hash;

    // Auto-size RAM budgets
    if (c->config.auto_size) {
        size_t avail = get_available_ram();
        size_t usable = (size_t)((double)avail * (1.0 - c->config.memory_reserve));
        c->config.hot_ram_bytes = (usable * 3) / 4;
        c->config.warm_ram_bytes = usable / 4;
        const size_t MIN_HOT  = 512ULL * 1024 * 1024;
        const size_t MIN_WARM = 256ULL * 1024 * 1024;
        bool boosted = false;
        if (c->config.hot_ram_bytes < MIN_HOT) {
            c->config.hot_ram_bytes = MIN_HOT;
            boosted = true;
        }
        if (c->config.warm_ram_bytes < MIN_WARM) {
            c->config.warm_ram_bytes = MIN_WARM;
            boosted = true;
        }
        LOG_INF("SSD cache: auto-sized hot=%zu MiB warm=%zu MiB (avail=%zu MiB)\n",
                c->config.hot_ram_bytes / 1024 / 1024,
                c->config.warm_ram_bytes / 1024 / 1024,
                avail / 1024 / 1024);
        if (boosted) {
            LOG_INF("SSD cache: floors applied (min hot=%zu MiB, min warm=%zu MiB)\n",
                    MIN_HOT / 1024 / 1024, MIN_WARM / 1024 / 1024);
        }
    }

    // Create base directory
    mkdir(path, 0755);

    // Create conversation-specific directory
    char conv_hex[17];
    snprintf(conv_hex, sizeof(conv_hex), "%016lx", (unsigned long)conv_hash);
    c->base_path = std::string(path);
    c->model_dir = std::string(path) + "/" + conv_hex;
    mkdir(c->model_dir.c_str(), 0755);

    // Load index file for next_id
    if (!read_index_file(c)) {
        LOG_INF("SSD cache: no index file, starting fresh\n");
        c->index.clear();
        c->slot_latest.clear();
        c->next_id = 1;
    }

    // Scan checkpoint files to rebuild in-memory index
    size_t loaded = scan_checkpoint_files(c);
    LOG_INF("SSD cache: loaded %zu checkpoints from %s (next_id=%lu)\n",
            loaded, c->model_dir.c_str(), (unsigned long)c->next_id);

    c->initialized = true;
    return c;
}

void kv_ssd_free(kv_ssd_cache* cache) {
    if (!cache) return;

    // Write final index file
    write_index_file(cache);

    LOG_INF("SSD cache: shutdown (stored=%lu hits=%lu misses=%lu evicts=%lu)\n",
            (unsigned long)cache->stats_stores,
            (unsigned long)cache->stats_hits,
            (unsigned long)cache->stats_misses,
            (unsigned long)cache->stats_evicts);

    delete cache;
}

void kv_ssd_set_compat_hash(kv_ssd_cache* cache, uint64_t compat_hash) {
    if (!cache) return;
    std::lock_guard<std::mutex> lock(cache->mutex);
    cache->compat_hash = compat_hash;
    LOG_INF("SSD cache: model compat_hash set to %016lx\n", (unsigned long)compat_hash);
}

uint64_t kv_ssd_store(kv_ssd_cache* cache,
                  uint32_t slot_id,
                  const uint8_t* data, size_t data_size,
                  int32_t pos_min, int32_t pos_max,
                  uint64_t n_tokens, uint32_t turn_id,
                  const uint32_t* tokens, size_t tokens_size,
                  uint64_t compat_hash)
{
    if (!cache || !cache->initialized || !data || data_size == 0) return 0;

    std::lock_guard<std::mutex> lock(cache->mutex);

    uint64_t id = cache->next_id++;

    // Compute token hash
    uint64_t token_hash = kv_ssd_hash_tokens(tokens, tokens_size);
    size_t token_count = tokens ? std::min(tokens_size, (size_t)KV_SSD_TOKEN_PREFIX_MAX) : 0;

    // Build record header
    kv_ssd_record rec;
    memset(&rec, 0, sizeof(rec));
    rec.magic = KV_SSD_MAGIC_REC;
    rec.version = KV_SSD_VERSION;
    rec.id = id;
    rec.slot_id = slot_id;
    rec.pos_min = pos_min;
    rec.pos_max = pos_max;
    rec.n_tokens = n_tokens;
    rec.turn_created = turn_id;
    rec.data_size = data_size;
    rec.token_hash = token_hash;
    rec.compat_hash = compat_hash;
    rec.token_count = (uint32_t)token_count;
    if (tokens && token_count > 0) {
        memcpy(rec.token_prefix, tokens, token_count * sizeof(uint32_t));
    }

    // Write checkpoint file: header + data in one file
    std::string filepath = ckpt_path(cache, id);
    int fd = open(filepath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        LOG_WRN("SSD cache: failed to create %s: %s\n", filepath.c_str(), strerror(errno));
        cache->next_id--;
        return 0;
    }

    bool ok = true;
    if (!pwrite_all(fd, &rec, sizeof(rec), 0)) {
        LOG_WRN("SSD cache: failed to write record header: %s\n", strerror(errno));
        ok = false;
    }
    if (ok && !pwrite_all(fd, data, data_size, (off_t)sizeof(kv_ssd_record))) {
        LOG_WRN("SSD cache: failed to write checkpoint data: %s\n", strerror(errno));
        ok = false;
    }
    fsync(fd);
    close(fd);

    if (!ok) {
        unlink(filepath.c_str());
        cache->next_id--;
        return 0;
    }

    // Update index file with new next_id
    write_index_file(cache);

    // Make room in hot tier
    make_room_hot(cache, data_size);

    // Store in hot cache
    cache->hot_cache[id].assign(data, data + data_size);
    cache->hot_bytes += data_size;

    // Build index entry
    kv_ssd_checkpoint ckpt;
    ckpt.id = id;
    ckpt.slot_id = slot_id;
    ckpt.pos_min = pos_min;
    ckpt.pos_max = pos_max;
    ckpt.n_tokens = n_tokens;
    ckpt.turn_id = turn_id;
    ckpt.turn_created = turn_id;
    ckpt.token_hash = token_hash;
    ckpt.compat_hash = compat_hash;
    ckpt.token_count = token_count;
    ckpt.tier = KV_TIER_HOT;
    if (token_count > 0) {
        ckpt.token_prefix.assign(rec.token_prefix, rec.token_prefix + token_count);
    }
    ckpt.data_size = data_size;
    ckpt.last_access = now_ms();
    ckpt.access_count = 1;

    cache->index[id] = ckpt;
    cache->slot_latest[slot_id] = id;
    cache->stats_stores++;

    LOG_INF("SSD cache: stored checkpoint %lu slot=%u tokens=%lu size=%zu MiB "
            "(hot=%zu MiB warm=%zu MiB total=%zu)\n",
            (unsigned long)id, slot_id, (unsigned long)n_tokens,
            data_size / 1024 / 1024,
            cache->hot_bytes / 1024 / 1024,
            cache->warm_bytes / 1024 / 1024,
            cache->index.size());

    return id;
}

bool kv_ssd_load(kv_ssd_cache* cache, uint64_t checkpoint_id,
                 std::vector<uint8_t>& out_data)
{
    if (!cache || !cache->initialized || checkpoint_id == 0) return false;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Reject checkpoints from incompatible model configs
    if (cache->compat_hash != 0) {
        auto ckpt_it = cache->index.find(checkpoint_id);
        if (ckpt_it != cache->index.end() && ckpt_it->second.compat_hash != cache->compat_hash) {
            LOG_WRN("SSD cache: rejecting checkpoint %lu - compat_hash mismatch "
                    "(stored=%016lx current=%016lx)\n",
                    (unsigned long)checkpoint_id,
                    (unsigned long)ckpt_it->second.compat_hash,
                    (unsigned long)cache->compat_hash);
            cache->stats_misses++;
            return false;
        }
    }

    // Check hot cache
    auto hot_it = cache->hot_cache.find(checkpoint_id);
    if (hot_it != cache->hot_cache.end()) {
        out_data = hot_it->second;
        auto& ckpt = cache->index[checkpoint_id];
        ckpt.last_access = now_ms();
        ckpt.access_count++;
        cache->stats_hits++;
        return true;
    }

    // Check warm cache
    auto warm_it = cache->warm_cache.find(checkpoint_id);
    if (warm_it != cache->warm_cache.end()) {
        out_data = warm_it->second;
        promote_to_hot(cache, checkpoint_id);
        cache->stats_hits++;
        return true;
    }

    // Load from SSD file and promote to hot
    if (promote_to_hot(cache, checkpoint_id)) {
        auto it = cache->hot_cache.find(checkpoint_id);
        if (it != cache->hot_cache.end()) {
            out_data = it->second;
            cache->stats_hits++;
            return true;
        }
    }

    cache->stats_misses++;
    return false;
}

uint64_t kv_ssd_find_match(kv_ssd_cache* cache,
                           const uint32_t* tokens, size_t tokens_size,
                           uint32_t current_turn,
                           uint64_t max_n_tokens,
                           int32_t n_past)
{
    (void)current_turn;  // unused - was for cross-conversation matching
    (void)n_past;        // unused - was for tiered search
    if (!cache || !cache->initialized || !tokens || tokens_size == 0) return 0;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Within-conversation search: find the checkpoint with the longest
    // common prefix that fits within the task.
    uint64_t best_id = 0;
    int32_t best_lcp = 0;
    uint32_t best_turn = 0;
    uint64_t best_n_tokens = 0;

    for (const auto& [id, ckpt] : cache->index) {
        if (cache->compat_hash != 0 && ckpt.compat_hash != cache->compat_hash) continue;
        if (max_n_tokens > 0 && ckpt.n_tokens >= max_n_tokens) continue;

        // Compute longest common prefix with stored token prefix
        const size_t cmp_count = std::min(tokens_size, ckpt.token_prefix.size());
        int32_t lcp = 0;
        for (size_t i = 0; i < cmp_count; i++) {
            if (tokens[i] == ckpt.token_prefix[i]) lcp++;
            else break;
        }

        if (lcp < (int32_t)cmp_count) continue;

        // Prefer most recent turn. Within the same turn, prefer more tokens
        // (larger checkpoint = more cache reuse = less reprocessing).
        if (ckpt.turn_created > best_turn ||
            (ckpt.turn_created == best_turn && ckpt.n_tokens > best_n_tokens)) {
            best_turn = ckpt.turn_created;
            best_n_tokens = ckpt.n_tokens;
            best_lcp = lcp;
            best_id = id;
        }
    }

    if (best_id != 0) {
        auto it = cache->index.find(best_id);
        uint64_t ntok = it != cache->index.end() ? (uint64_t)it->second.n_tokens : 0;
        LOG_INF("SSD cache: within-conv match checkpoint %lu conv=%016lx"
                " turn=%u n_tokens=%lu lcp=%d\n",
                (unsigned long)best_id, (unsigned long)cache->conv_hash,
                best_turn, (unsigned long)ntok, best_lcp);
    }

    return best_id;
}

uint64_t kv_ssd_find_by_slot(kv_ssd_cache* cache,
                             uint32_t slot_id,
                             uint64_t min_tokens,
                             uint32_t current_turn)
{
    if (!cache || !cache->initialized) return 0;

    std::lock_guard<std::mutex> lock(cache->mutex);

    uint64_t best_id = 0;
    uint64_t best_tokens = 0;

    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.slot_id != slot_id) continue;
        if (cache->compat_hash != 0 && ckpt.compat_hash != cache->compat_hash) continue;
        if (min_tokens > 0 && ckpt.n_tokens < min_tokens) continue;
        if (current_turn > 0 && ckpt.turn_id + (uint32_t)cache->config.warm_turns < current_turn) continue;
        if (ckpt.n_tokens > best_tokens) {
            best_tokens = ckpt.n_tokens;
            best_id = id;
        }
    }

    return best_id;
}

void kv_ssd_on_turn_complete(kv_ssd_cache* cache, uint32_t turn_id) {
    if (!cache || !cache->initialized) return;

    std::lock_guard<std::mutex> lock(cache->mutex);

    // Demote hot entries inactive for too many turns
    std::vector<uint64_t> to_demote_hw;
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.tier == KV_TIER_HOT && ckpt.turn_id + (uint32_t)cache->config.hot_turns <= turn_id) {
            to_demote_hw.push_back(id);
        }
    }
    for (uint64_t id : to_demote_hw) {
        auto it = cache->hot_cache.find(id);
        if (it != cache->hot_cache.end()) {
            size_t sz = it->second.size();
            cache->warm_cache[id] = std::move(it->second);
            cache->hot_cache.erase(it);
            cache->hot_bytes -= sz;
            cache->warm_bytes += sz;
            cache->index[id].tier = KV_TIER_WARM;
            cache->stats_evicts++;
        }
    }

    // Demote warm entries inactive for too many turns
    std::vector<uint64_t> to_demote_wc;
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.tier == KV_TIER_WARM && ckpt.turn_id + (uint32_t)cache->config.warm_turns <= turn_id) {
            to_demote_wc.push_back(id);
        }
    }
    for (uint64_t id : to_demote_wc) {
        auto it = cache->warm_cache.find(id);
        if (it != cache->warm_cache.end()) {
            size_t sz = it->second.size();
            cache->warm_cache.erase(it);
            cache->warm_bytes -= sz;
            cache->index[id].tier = KV_TIER_COLD;
            cache->stats_evicts++;
        }
    }

    // Ring buffer eviction: delete oldest cold checkpoints from disk
    ring_buffer_evict(cache);

    LOG_INF("SSD cache: turn %u complete (hot=%zu MiB warm=%zu MiB cold=%zu checkpoints=%zu)\n",
            turn_id,
            cache->hot_bytes / 1024 / 1024,
            cache->warm_bytes / 1024 / 1024,
            cache->index.size() - cache->hot_cache.size() - cache->warm_cache.size(),
            cache->index.size());
}

const kv_ssd_checkpoint* kv_ssd_get_meta(kv_ssd_cache* cache, uint64_t id) {
    if (!cache || !cache->initialized) return nullptr;
    auto it = cache->index.find(id);
    if (it == cache->index.end()) return nullptr;
    return &it->second;
}

void kv_ssd_get_stats(kv_ssd_cache* cache,
                      size_t* out_hot_bytes, size_t* out_warm_bytes,
                      size_t* out_cold_count, size_t* out_total_count,
                      uint64_t* out_hits, uint64_t* out_misses)
{
    if (!cache) return;
    std::lock_guard<std::mutex> lock(cache->mutex);
    if (out_hot_bytes)  *out_hot_bytes  = cache->hot_bytes;
    if (out_warm_bytes) *out_warm_bytes = cache->warm_bytes;
    if (out_total_count) *out_total_count = cache->index.size();
    if (out_cold_count) {
        size_t cold = 0;
        for (const auto& [id, ckpt] : cache->index) {
            if (ckpt.tier == KV_TIER_COLD) cold++;
        }
        *out_cold_count = cold;
    }
    if (out_hits)   *out_hits   = cache->stats_hits;
    if (out_misses) *out_misses = cache->stats_misses;
}

uint32_t kv_ssd_get_max_turn_id(kv_ssd_cache* cache) {
    if (!cache || !cache->initialized) return 0;
    std::lock_guard<std::mutex> lock(cache->mutex);

    uint32_t max_turn = 0;
    for (const auto& [id, ckpt] : cache->index) {
        if (ckpt.turn_created > max_turn) {
            max_turn = ckpt.turn_created;
        }
    }
    return max_turn;
}

// =============================================================================
// Conversation continuation and global operations
// =============================================================================

// Scan all conversation directories for a fuzzy prefix match.
// Used for conversation continuation detection after restart.
// Returns conv_hash of best match, or 0 if none found above min_overlap.
uint64_t kv_ssd_find_continuation(
    const char* base_path,
    const uint32_t* tokens, size_t tokens_size,
    float min_overlap,
    uint64_t compat_hash)
{
    if (!base_path || !tokens || tokens_size == 0) return 0;

    DIR* base_dir = opendir(base_path);
    if (!base_dir) return 0;

    uint64_t best_conv = 0;
    float best_score = 0.0f;

    struct dirent* ent;
    while ((ent = readdir(base_dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;

        // Each subdirectory is a conv_hash (16 hex chars)
        std::string conv_dir = std::string(base_path) + "/" + ent->d_name;
        struct stat st;
        if (stat(conv_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

        // Parse conv_hash from directory name
        uint64_t conv_hash = 0;
        if (sscanf(ent->d_name, "%016lx", &conv_hash) != 1) continue;

        // Load the index file for this conversation
        std::string index_file = conv_dir + "/index.bin";
        int fd = open(index_file.c_str(), O_RDONLY);
        if (fd < 0) continue;

        kv_ssd_index_header hdr;
        bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
        close(fd);
        if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX) continue;

        // Skip directories with mismatched model config
        if (compat_hash != 0 && hdr.compat_hash != 0 && hdr.compat_hash != compat_hash) continue;

        // Scan for checkpoint with best prefix overlap
        DIR* conv_fd = opendir(conv_dir.c_str());
        if (!conv_fd) continue;

        float best_conv_score = 0.0f;

        struct dirent* ckpt_ent;
        while ((ckpt_ent = readdir(conv_fd)) != nullptr) {
            if (strncmp(ckpt_ent->d_name, "ckpt-", 5) != 0) continue;
            size_t dlen = strlen(ckpt_ent->d_name);
            if (dlen < 5 || strcmp(ckpt_ent->d_name + dlen - 4, ".bin") != 0) continue;

            std::string ckpt_file = conv_dir + "/" + ckpt_ent->d_name;
            int cfd = open(ckpt_file.c_str(), O_RDONLY);
            if (cfd < 0) continue;

            kv_ssd_record rec;
            bool rok = pread_all(cfd, &rec, sizeof(rec), 0);
            close(cfd);
            if (!rok || rec.magic != KV_SSD_MAGIC_REC) continue;

            // Compute overlap with stored prefix
            size_t cmp_count = std::min(tokens_size, (size_t)rec.token_count);
            if (cmp_count < 16) continue; // too short to be meaningful

            size_t matches = 0;
            for (size_t i = 0; i < cmp_count; i++) {
                if (tokens[i] == rec.token_prefix[i]) matches++;
                else break;
            }

            float score = (float)matches / (float)cmp_count;
            if (score > best_conv_score) best_conv_score = score;
        }
        closedir(conv_fd);

        if (best_conv_score >= min_overlap && best_conv_score > best_score) {
            best_score = best_conv_score;
            best_conv = conv_hash;
        }
    }
    closedir(base_dir);

    if (best_conv != 0) {
        LOG_INF("SSD cache: continuation found conv=%016lx overlap=%.1f%%\n",
                (unsigned long)best_conv, best_score * 100.0f);
    }

    return best_conv;
}

// Get maximum turn_id across all conversation directories.
uint32_t kv_ssd_get_max_turn_id_global(const char* base_path) {
    if (!base_path) return 0;

    DIR* base_dir = opendir(base_path);
    if (!base_dir) return 0;

    uint32_t max_turn = 0;

    struct dirent* ent;
    while ((ent = readdir(base_dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;

        std::string conv_dir = std::string(base_path) + "/" + ent->d_name;
        struct stat st;
        if (stat(conv_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;

        // Parse conv_hash (validate 16-char hex name)
        uint64_t conv_hash_test = 0;
        if (sscanf(ent->d_name, "%016lx", &conv_hash_test) != 1) continue;

        std::string index_file = conv_dir + "/index.bin";
        int fd = open(index_file.c_str(), O_RDONLY);
        if (fd < 0) continue;

        kv_ssd_index_header hdr;
        bool ok = pread_all(fd, &hdr, sizeof(hdr), 0);
        close(fd);
        if (!ok || hdr.magic != KV_SSD_MAGIC_INDEX) continue;

        // Quick scan of checkpoint files for max turn_created
        DIR* conv_fd = opendir(conv_dir.c_str());
        if (!conv_fd) continue;

        struct dirent* ckpt_ent;
        while ((ckpt_ent = readdir(conv_fd)) != nullptr) {
            if (strncmp(ckpt_ent->d_name, "ckpt-", 5) != 0) continue;
            size_t dlen = strlen(ckpt_ent->d_name);
            if (dlen < 5 || strcmp(ckpt_ent->d_name + dlen - 4, ".bin") != 0) continue;

            std::string ckpt_file = conv_dir + "/" + ckpt_ent->d_name;
            int cfd = open(ckpt_file.c_str(), O_RDONLY);
            if (cfd < 0) continue;

            kv_ssd_record rec;
            bool rok = pread_all(cfd, &rec, sizeof(rec), 0);
            close(cfd);
            if (rok && rec.magic == KV_SSD_MAGIC_REC && rec.turn_created > max_turn) {
                max_turn = rec.turn_created;
            }
        }
        closedir(conv_fd);
    }
    closedir(base_dir);

    return max_turn;
}
