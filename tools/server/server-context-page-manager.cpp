// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration using kv_ssd_cache

#include "server-context-page-manager.h"
#include "server-context-ssd-cache.h"
#include "server-context.h"
#include "server-task.h"
#include "llama.h"
#include "build-info.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

namespace llama {

server_context_page_manager::server_context_page_manager(
    const char* ssd_path,
    const kv_eviction_config* cfg,
    size_t /* n_tokens_total */,
    size_t max_cross_slot_checkpoints
) : max_cross_slot_checkpoints_(max_cross_slot_checkpoints)
{
    ssd_base_path_ = ssd_path;
    mkdir(ssd_path, 0755);

    kv_ssd_config ssd_cfg;
    if (cfg) {
        ssd_cfg.hot_ram_bytes = cfg->max_hot_bytes > 0 ? cfg->max_hot_bytes : 2ULL * 1024 * 1024 * 1024;
        ssd_cfg.warm_ram_bytes = cfg->max_warm_bytes > 0 ? cfg->max_warm_bytes : 1ULL * 1024 * 1024 * 1024;
        ssd_cfg.hot_window_tokens = cfg->hot_window_tokens;
        ssd_cfg.hot_turns = cfg->turn_inactivity_threshold > 0 ? cfg->turn_inactivity_threshold : 2;
        ssd_cfg.warm_turns = cfg->turn_inactivity_threshold > 0 ? cfg->turn_inactivity_threshold * 2 : 4;
        ssd_cfg.auto_size = cfg->auto_size;
        ssd_cfg.max_cold_checkpoints = cfg->max_cold_checkpoints;
        ssd_cfg.memory_reserve = cfg->memory_reserve;
    }
    if (ssd_cfg.hot_ram_bytes == 0) ssd_cfg.hot_ram_bytes = 2ULL * 1024 * 1024 * 1024;
    if (ssd_cfg.warm_ram_bytes == 0) ssd_cfg.warm_ram_bytes = 1ULL * 1024 * 1024 * 1024;
    if (ssd_cfg.hot_turns == 0) ssd_cfg.hot_turns = 2;
    if (ssd_cfg.warm_turns == 0) ssd_cfg.warm_turns = 4;

    // Store config for creating per-conversation caches later
    // (save a copy of the config)
    config_ = ssd_cfg;
}

server_context_page_manager::~server_context_page_manager() {
    // Each unique_ptr in conv_caches_ handles its own kv_ssd_free
}

void server_context_page_manager::set_model_info(const struct llama_model* model,
                                                   int cache_type_k, int cache_type_v) {
    if (!model) return;

    char desc_buf[2048];
    int desc_len = llama_model_desc(model, desc_buf, sizeof(desc_buf));
    if (desc_len < 0) {
        LOG_WRN("SSD cache: llama_model_desc() failed, skipping compat_hash\n");
        return;
    }

    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < desc_len; i++) {
        h ^= (uint64_t)(unsigned char)desc_buf[i];
        h *= 1099511628211ULL;
    }
    // Include build commit in compat_hash so checkpoints from different
    // builds are automatically rejected (state serialization format may change)
    const char * commit = llama_commit();
    if (commit) {
        for (int i = 0; commit[i]; i++) {
            h ^= (uint64_t)(unsigned char)commit[i];
            h *= 1099511628211ULL;
        }
    }
    uint32_t tk = (uint32_t)cache_type_k;
    h ^= (uint64_t)(tk & 0xFF);         h *= 1099511628211ULL;
    h ^= (uint64_t)((tk >> 8) & 0xFF);  h *= 1099511628211ULL;
    h ^= (uint64_t)((tk >> 16) & 0xFF); h *= 1099511628211ULL;
    h ^= (uint64_t)((tk >> 24) & 0xFF); h *= 1099511628211ULL;
    uint32_t tv = (uint32_t)cache_type_v;
    h ^= (uint64_t)(tv & 0xFF);         h *= 1099511628211ULL;
    h ^= (uint64_t)((tv >> 8) & 0xFF);  h *= 1099511628211ULL;
    h ^= (uint64_t)((tv >> 16) & 0xFF); h *= 1099511628211ULL;
    h ^= (uint64_t)((tv >> 24) & 0xFF); h *= 1099511628211ULL;

    model_compat_hash_ = h;

    // Set compat_hash on any already-created cache instances
    for (auto& [conv, wrapper] : conv_wrappers_) {
        wrapper->set_compat_hash(h);
    }

    LOG_INF("SSD cache: model compat_hash %016lx (arch dims + type_k=%d type_v=%d)\n",
            (unsigned long)h, cache_type_k, cache_type_v);
}

server_ssd_cache* server_context_page_manager::get_or_create_cache(uint64_t conv_hash) {
    if (conv_hash == 0) return nullptr;

    auto it = conv_wrappers_.find(conv_hash);
    if (it != conv_wrappers_.end()) {
        return it->second.get();
    }

    // Evict oldest conversation if at max
    if ((int)conv_caches_.size() >= max_conversations) {
        uint64_t oldest_conv = 0;
        time_t oldest_mtime = 0;

        for (const auto& [cv, cache] : conv_caches_) {
            std::string dir = ssd_base_path_ + "/";
            char hex[17];
            snprintf(hex, sizeof(hex), "%016lx", (unsigned long)cv);
            dir += hex;

            struct stat st;
            if (stat(dir.c_str(), &st) == 0) {
                if (oldest_conv == 0 || st.st_mtime < oldest_mtime) {
                    oldest_mtime = st.st_mtime;
                    oldest_conv = cv;
                }
            }
        }

        if (oldest_conv != 0) {
            LOG_WRN("SSD cache: evicting conversation %016lx (max=%d reached)\n",
                     (unsigned long)oldest_conv, max_conversations);

            // Delete conversation directory and all its files
            std::string dir = ssd_base_path_ + "/";
            char hex[17];
            snprintf(hex, sizeof(hex), "%016lx", (unsigned long)oldest_conv);
            dir += hex;

            DIR* d = opendir(dir.c_str());
            if (d) {
                struct dirent* ent;
                while ((ent = readdir(d)) != nullptr) {
                    if (ent->d_name[0] == '.') continue;
                    std::string file = dir + "/" + ent->d_name;
                    unlink(file.c_str());
                }
                closedir(d);
            }
            rmdir(dir.c_str());

            conv_wrappers_.erase(oldest_conv);
            conv_caches_.erase(oldest_conv);
        }
    }

    // Create new cache for this conversation
    auto raw = kv_ssd_init(ssd_base_path_.c_str(), &config_, conv_hash);
    if (!raw) return nullptr;

    auto cache_ptr = std::unique_ptr<kv_ssd_cache>(raw);
    auto wrapper = std::make_unique<server_ssd_cache>(raw);

    // Apply model compat_hash if already set
    if (model_compat_hash_ != 0) {
        wrapper->set_compat_hash(model_compat_hash_);
    }

    server_ssd_cache* result = wrapper.get();
    conv_caches_[conv_hash] = std::move(cache_ptr);
    conv_wrappers_[conv_hash] = std::move(wrapper);

    LOG_INF("SSD cache: created new conversation cache conv=%016lx (total=%zu)\n",
             (unsigned long)conv_hash, conv_caches_.size());

    return result;
}

uint64_t server_context_page_manager::get_timestamp_ms() const {
    auto now = std::chrono::system_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

void server_context_page_manager::evict_slot_internal(uint32_t slot_id) {
    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return;
    checkpoints_.erase(it);
}

bool server_context_page_manager::store_checkpoint(
    uint32_t slot_id,
    struct llama_context* ctx,
    const common_prompt_checkpoint& ckpt,
    uint32_t turn_id
) {
    return store_checkpoint_with_tokens(slot_id, ctx, ckpt, nullptr, 0, turn_id);
}

bool server_context_page_manager::store_checkpoint_with_tokens(
    uint32_t slot_id,
    struct llama_context* ctx,
    const common_prompt_checkpoint& ckpt,
    const llama_token* tokens,
    size_t tokens_size,
    uint32_t turn_id,
    uint64_t conv_hash
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (!ckpt.data_tgt.data()) return false;

    // Get or create per-conversation cache
    server_ssd_cache* sc = get_or_create_cache(conv_hash);
    if (!sc) return false;

    // Evict if needed
    if (checkpoints_.size() >= max_cross_slot_checkpoints_) {
        auto it = std::min_element(checkpoints_.begin(), checkpoints_.end(),
            [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });
        if (it != checkpoints_.end()) evict_slot_internal(it->first);
    }

    uint64_t ckpt_id = sc->store(slot_id, ctx, ckpt, tokens, tokens_size, turn_id);
    if (ckpt_id == 0) return false;

    stored_checkpoint sc2;
    sc2.checkpoint_id = ckpt_id;
    sc2.slot_id = slot_id;
    sc2.turn_id = turn_id;
    sc2.size_bytes = ckpt.data_tgt.size() + ckpt.data_dft.size();
    sc2.n_tokens = ckpt.n_tokens;
    sc2.pos_min = ckpt.pos_min;
    sc2.pos_max = ckpt.pos_max;
    sc2.last_access = get_timestamp_ms();
    sc2.access_count = 0;
    if (tokens && tokens_size > 0) {
        sc2.tokens.assign(tokens, tokens + std::min(tokens_size, (size_t)256));
    }

    checkpoints_.emplace(slot_id, std::move(sc2));
    return true;
}

bool server_context_page_manager::load_checkpoint(
    uint32_t slot_id,
    uint32_t /* turn_id */,
    struct llama_context* ctx,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return false;

    // Find which cache has this checkpoint
    server_ssd_cache* sc = nullptr;
    // We need to know which conversation cache stores this checkpoint.
    // Since we removed conv_hash from checkpoint metadata, we iterate all caches.
    for (auto& [conv, wrapper] : conv_wrappers_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(
            conv_caches_[conv].get(), it->second.checkpoint_id);
        if (meta) {
            sc = wrapper.get();
            break;
        }
    }
    if (!sc) return false;

    // Load from SSD cache, which will promote to hot tier
    bool ok = sc->load(it->second.checkpoint_id, ctx, out_pos_min, out_pos_max, out_n_tokens);

    if (ok) {
        it->second.last_access = get_timestamp_ms();
        it->second.access_count++;
        cache_hits_++;
    } else {
        cache_misses_++;
    }

    return ok;
}

bool server_context_page_manager::load_checkpoint_by_id(
    uint64_t checkpoint_id,
    struct llama_context* ctx,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens
) {
    if (checkpoint_id == 0) return false;

    // Find which cache has this checkpoint
    server_ssd_cache* sc = nullptr;
    for (auto& [conv, wrapper] : conv_wrappers_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(
            conv_caches_[conv].get(), checkpoint_id);
        if (meta) {
            sc = wrapper.get();
            break;
        }
    }
    if (!sc) return false;

    bool ok = sc->load(checkpoint_id, ctx, out_pos_min, out_pos_max, out_n_tokens);

    if (ok) {
        cache_hits_++;
    } else {
        cache_misses_++;
    }

    return ok;
}

void server_context_page_manager::prefetch_for_slot(uint32_t slot_id, uint32_t /* turn_id */) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return;

    // Find and load from whichever cache has this checkpoint
    for (auto& [conv, cache] : conv_caches_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(cache.get(), it->second.checkpoint_id);
        if (meta && meta->tier == KV_TIER_COLD) {
            std::vector<uint8_t> dummy;
            kv_ssd_load(cache.get(), it->second.checkpoint_id, dummy);
            return;
        }
    }
}

void server_context_page_manager::on_turn_complete(uint32_t turn_id) {
    // Notify all cache instances
    for (auto& [conv, wrapper] : conv_wrappers_) {
        wrapper->on_turn_complete(turn_id);
    }

    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto& [slot_id, sc] : checkpoints_) {
        sc.turn_id = turn_id;
    }
}

bool server_context_page_manager::find_matching_checkpoint(
    const llama_token* tokens,
    size_t tokens_size,
    uint32_t current_turn,
    uint32_t& out_slot_id,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens,
    uint64_t conv_hash,
    int32_t n_past,
    uint64_t max_n_tokens
) {
    // Try exact conversation match first
    uint64_t effective_conv = conv_hash;

    // If this conv_hash doesn't have a cache yet, try continuation matching
    if (effective_conv != 0 && conv_wrappers_.find(effective_conv) == conv_wrappers_.end()) {
        uint64_t continuation = kv_ssd_find_continuation(
            ssd_base_path_.c_str(),
            (const uint32_t*)tokens, tokens_size,
            0.90f, model_compat_hash_);
        if (continuation != 0) {
            effective_conv = continuation;
            LOG_INF("SSD cache: reusing conversation %016lx (90%%+ prefix match)\n",
                     (unsigned long)continuation);
        }
    }

    server_ssd_cache* sc = get_or_create_cache(effective_conv);
    if (!sc) return false;

    uint64_t ckpt_id = sc->find_match(tokens, tokens_size, current_turn, max_n_tokens, n_past);
    if (ckpt_id == 0) {
        cache_misses_++;
        return false;
    }

    // Look up in checkpoints_ map first
    for (const auto& [slot_id, cp] : checkpoints_) {
        if (cp.checkpoint_id == ckpt_id) {
            out_slot_id = slot_id;
            out_pos_min = cp.pos_min;
            out_pos_max = cp.pos_max;
            out_n_tokens = cp.n_tokens;
            cache_hits_++;
            return true;
        }
    }

    // Look up in the cache's own metadata
    kv_ssd_cache* raw = conv_caches_[effective_conv].get();
    const kv_ssd_checkpoint* meta = kv_ssd_get_meta(raw, ckpt_id);
    if (meta) {
        out_slot_id = meta->slot_id;
        out_pos_min = meta->pos_min;
        out_pos_max = meta->pos_max;
        out_n_tokens = meta->n_tokens;
        cache_hits_++;
        return true;
    }

    cache_misses_++;
    return false;
}

bool server_context_page_manager::find_and_load_checkpoint(
    const llama_token* tokens,
    size_t tokens_size,
    uint32_t current_turn,
    struct llama_context* ctx,
    int32_t& out_pos_min,
    int32_t& out_pos_max,
    uint64_t& out_n_tokens,
    uint64_t conv_hash,
    int32_t n_past,
    uint64_t max_n_tokens
) {
    uint64_t effective_conv = conv_hash;

    // Try continuation matching if no cache exists for this conv_hash
    if (effective_conv != 0 && conv_wrappers_.find(effective_conv) == conv_wrappers_.end()) {
        uint64_t continuation = kv_ssd_find_continuation(
            ssd_base_path_.c_str(),
            (const uint32_t*)tokens, tokens_size,
            0.90f, model_compat_hash_);
        if (continuation != 0) {
            effective_conv = continuation;
            LOG_INF("SSD cache: reusing conversation %016lx for cold restart\n",
                     (unsigned long)continuation);
        }
    }

    server_ssd_cache* sc = get_or_create_cache(effective_conv);
    if (!sc) return false;

    uint64_t ckpt_id = sc->find_match(tokens, tokens_size, current_turn, max_n_tokens, n_past);
    if (ckpt_id == 0) {
        cache_misses_++;
        return false;
    }

    bool ok = sc->load(ckpt_id, ctx, out_pos_min, out_pos_max, out_n_tokens);
    if (ok) {
        cache_hits_++;
    } else {
        cache_misses_++;
    }
    return ok;
}

void server_context_page_manager::evict_slot(uint32_t slot_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    evict_slot_internal(slot_id);
}

bool server_context_page_manager::get_checkpoint_data(uint32_t slot_id, std::vector<uint8_t>& out_data) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = checkpoints_.find(slot_id);
    if (it == checkpoints_.end()) return false;

    // Find which cache has this checkpoint
    for (auto& [conv, cache] : conv_caches_) {
        const kv_ssd_checkpoint* meta = kv_ssd_get_meta(cache.get(), it->second.checkpoint_id);
        if (meta) {
            return kv_ssd_load(cache.get(), it->second.checkpoint_id, out_data);
        }
    }
    return false;
}

void server_context_page_manager::get_stats(
    size_t* hot_bytes, size_t* warm_bytes, size_t* cold_bytes,
    size_t* total_checkpoints, size_t* max_checkpoints,
    uint64_t* hits, uint64_t* misses, float* hit_rate
) const {
    size_t hot_sum = 0, warm_sum = 0, cold_sum = 0, total_sum = 0;
    for (const auto& [conv, cache] : conv_caches_) {
        size_t h, w, c, t;
        kv_ssd_get_stats(cache.get(), &h, &w, &c, &t, nullptr, nullptr);
        hot_sum += h;
        warm_sum += w;
        cold_sum += c;
        total_sum += t;
    }
    if (hot_bytes) *hot_bytes = hot_sum;
    if (warm_bytes) *warm_bytes = warm_sum;
    if (cold_bytes) *cold_bytes = cold_sum;
    if (total_checkpoints) *total_checkpoints = total_sum;
    if (max_checkpoints) *max_checkpoints = max_cross_slot_checkpoints_;
    if (hits) *hits = cache_hits_;
    if (misses) *misses = cache_misses_;
    if (hit_rate) {
        uint64_t h = cache_hits_, m = cache_misses_;
        *hit_rate = (h + m) > 0 ? (float)h / (float)(h + m) : 0.0f;
    }
}

uint32_t server_context_page_manager::get_max_turn_id() const {
    return kv_ssd_get_max_turn_id_global(ssd_base_path_.c_str());
}

} // namespace llama