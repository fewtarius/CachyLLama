// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius

#include "llama-moe-residency.h"
#include "llama-moe-coact.h"

#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#    include <io.h>
#    include <windows.h>

// Windows equivalent of getpagesize()
static int getpagesize() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<int>(si.dwPageSize);
}

// Advice constants for madvise() on Windows
#        ifndef MADV_WILLNEED
#            define MADV_WILLNEED 1
#        endif
#        ifndef MADV_DONTNEED
#            define MADV_DONTNEED 2
#        endif
#        ifndef MADV_FREE
#            define MADV_FREE 3
#        endif

// No-op madvise() on Windows
static int madvise(void *, size_t, int) {
    // Windows has no direct equivalent; ignore advice
    return 0;
}
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif


#ifdef _WIN32
// Simple Windows helpers to mimic mmap/munmap for file-backed mappings
static void * win_mmap_file(HANDLE hFile, size_t size, HANDLE * outMapHandle) {
    if (!hFile || hFile == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READWRITE, 0, static_cast<DWORD>(size), nullptr);
    if (!hMap) {
        return nullptr;
    }
    void * p = MapViewOfFile(hMap, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!p) {
        CloseHandle(hMap);
        return nullptr;
    }
    if (outMapHandle) {
        *outMapHandle = hMap;
    }
    return p;
}

static void win_munmap(void * addr, HANDLE hMap) {
    if (addr) {
        UnmapViewOfFile(addr);
    }
    if (hMap && hMap != INVALID_HANDLE_VALUE) {
        CloseHandle(hMap);
    }
}
#endif


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline size_t page_align_down(size_t x) {
    return x & ~(size_t(getpagesize()) - 1);
}

static inline size_t page_align_up(size_t x) {
    return (x + size_t(getpagesize()) - 1) & ~(size_t(getpagesize()) - 1);
}

// Cache scoring helpers

static inline double recency_score(uint64_t current_token, uint64_t last_access) {
    if (current_token <= last_access) return 1.0;
    const double dt = double(current_token - last_access);
    return 1.0 / (1.0 + dt);
}

static inline double frequency_score(uint64_t current_token, uint64_t loaded_at,
                                     uint64_t access_count) {
    if (current_token <= loaded_at) return double(access_count);
    const double age = double(current_token - loaded_at);
    return double(access_count) / (1.0 + age);
}

// Combined recency+frequency score. Higher = keep longer.
static inline double rf_score(uint64_t current_token,
                              const llama_moe_layer_residency_internal::cache_entry & e) {
    return 0.5 * recency_score(current_token, e.last_access) +
           0.5 * frequency_score(current_token, e.loaded_at, e.access_count);
}

// Find the slot in `cache` with the lowest rf_score (most evictable).
// Returns the slot index.
static int find_evict_slot(const std::vector<llama_moe_layer_residency_internal::cache_entry> & cache,
                           uint64_t current_token) {
    int best = -1;
    double best_score = 1e30;
    for (size_t i = 0; i < cache.size(); ++i) {
        if (!cache[i].occupied) return (int) i;
        const double s = rf_score(current_token, cache[i]);
        if (s < best_score) {
            best_score = s;
            best = (int) i;
        }
    }
    return best;
}

// madvise a region. Aligns to page boundaries so the kernel can act on it.
// Silently ignores invalid pointers (non-mmap'd regions).
static void safe_madvise(void * base, size_t len, int advice) {
    if (!base || len == 0) return;
    uintptr_t p = reinterpret_cast<uintptr_t>(base);
    uintptr_t page_start = p & ~(uintptr_t(getpagesize()) - 1);
    uintptr_t end = p + len;
    uintptr_t page_end = (end + uintptr_t(getpagesize()) - 1) & ~(uintptr_t(getpagesize()) - 1);
    size_t aligned_len = page_end - page_start;
    if (aligned_len == 0) return;
    (void) madvise(reinterpret_cast<void *>(page_start), aligned_len, advice);
}

template <typename Fn>
static void for_each_tensor(llama_moe_layer_residency_internal & lr, Fn fn) {
    struct entry { ggml_tensor * t; size_t stride; };
    entry entries[4] = {
        { lr.t_gate,    lr.gate_stride    },
        { lr.t_up,      lr.up_stride      },
        { lr.t_down,    lr.down_stride    },
        { lr.t_gate_up, lr.gate_up_stride },
    };
    for (auto & e : entries) {
        if (e.t && e.t->data && e.stride > 0) {
            fn(e.t->data, e.stride);
        }
    }
}

// ---------------------------------------------------------------------------
// build()
// ---------------------------------------------------------------------------

bool llama_moe_residency_build(
        const struct llama_model * model,
        struct llama_moe_residency_internal_cfg cfg,
        struct llama_moe_residency_state * out) {
    if (!out) return false;
    out->cfg = cfg;
    if (!cfg.enabled) return false;
    if (!model) return false;

    const auto & hparams = model->hparams;
    const int n_expert = hparams.n_expert;
    const int n_expert_used = hparams.n_expert_used;
    const int n_layer = hparams.n_layer();
    if (n_expert <= 0 || n_expert_used <= 0) {
        return false;
    }

    out->layers.clear();
    out->layers.reserve(n_layer);
    out->n_layers = n_layer;
    out->n_expert = n_expert;
    out->n_expert_used = n_expert_used;

    int layers_with_experts = 0;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model->layers[il];
        ggml_tensor * t_gate = layer.ffn_gate_exps;
        ggml_tensor * t_up   = layer.ffn_up_exps;
        ggml_tensor * t_down = layer.ffn_down_exps;
        ggml_tensor * t_gu   = layer.ffn_gate_up_exps;

        const bool has_any = (t_gate && t_up && t_down) || t_gu;
        if (!has_any) continue;

        llama_moe_layer_residency_internal lr;
        lr.model_layer = il;
        lr.n_expert = n_expert;
        lr.t_gate    = t_gate;
        lr.t_up      = t_up;
        lr.t_down    = t_down;
        lr.t_gate_up = t_gu;

        if (t_gate) lr.gate_stride    = t_gate->nb[2];
        if (t_up)   lr.up_stride      = t_up->nb[2];
        if (t_down) lr.down_stride    = t_down->nb[2];
        if (t_gu)   lr.gate_up_stride = t_gu->nb[2];

        // Allocate per-layer R+F cache. Sized to max_resident_per_layer.
        lr.cache.assign(cfg.max_resident_per_layer,
                        llama_moe_layer_residency_internal::cache_entry{});
        lr.slot_of.assign(n_expert, -1);

        out->layers.push_back(std::move(lr));
        layers_with_experts++;
    }

    if (layers_with_experts == 0) {
        out->layers.clear();
        out->n_layers = 0;
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// touch()
// ---------------------------------------------------------------------------

void llama_moe_residency_touch(
        struct llama_moe_residency_state * st,
        int layer_idx,
        int expert_id,
        bool * was_already_loaded) {
    if (!st || !st->cfg.enabled) return;
    if (layer_idx < 0 || layer_idx >= (int) st->layers.size()) return;
    if (expert_id < 0 || expert_id >= st->layers[layer_idx].n_expert) return;

    auto & lr = st->layers[layer_idx];
    lr.token_counter++;

    int slot = lr.slot_of[expert_id];
    bool hit = (slot >= 0 && lr.cache[slot].occupied);
    if (was_already_loaded) *was_already_loaded = hit;

    if (hit) {
        // Cache hit: update recency/frequency.
        auto & e = lr.cache[slot];
        e.last_access = lr.token_counter;
        e.access_count++;
        lr.hits++;
        st->total_hits++;
        return;
    }

    // Cache miss: find an evict slot (lowest R+F score, or first empty).
    const int evict_slot = find_evict_slot(lr.cache, lr.token_counter);
    if (evict_slot < 0) return;  // shouldn't happen

    // If the chosen slot is occupied, evict it first.
    if (lr.cache[evict_slot].occupied) {
        const int evicted_id = lr.cache[evict_slot].expert_id;
        if (evicted_id >= 0 && evicted_id < (int) lr.slot_of.size()) {
            lr.slot_of[evicted_id] = -1;
            const size_t eoff = (size_t) evicted_id;
            // MADV_FREE (not MADV_DONTNEED): the kernel can drop these
            // pages if it needs the memory but is free to keep them
            // resident otherwise. MADV_DONTNEED forced the kernel to
            // evict immediately, which on memory-constrained systems
            // (Flip: 8 GB OS-only RAM, 32 GB physical) turned every
            // cold miss into a disk page-fault and dropped prompt eval
            // from ~215 t/s to ~56 t/s on Qwen3.6-35B-A3B Q4_K_XL
            // (3.8x regression, root-caused 2026-07-27). With
            // MADV_FREE the kernel LRU + memory pressure decide
            // eviction; pages stay hot while memory is available.
            for_each_tensor(lr, [&](void * base, size_t stride) {
                safe_madvise((uint8_t *) base + eoff * stride, stride, MADV_FREE);
            });
            st->total_evicted++;
        }
    }

    // Install the new entry.
    auto & e = lr.cache[evict_slot];
    e.expert_id    = expert_id;
    e.last_access  = lr.token_counter;
    e.access_count = 1;
    e.loaded_at    = lr.token_counter;
    e.occupied     = true;
    lr.slot_of[expert_id] = evict_slot;
    lr.misses++;
    st->total_misses++;
    st->total_touched++;

    // Mark pages as WILLNEED for all present tensors.
    const size_t off = (size_t) expert_id;
    for_each_tensor(lr, [&](void * base, size_t stride) {
        safe_madvise((uint8_t *) base + off * stride, stride, MADV_WILLNEED);
    });
}

void llama_moe_residency_touch_layer_selection(
        struct llama_moe_residency_state * st,
        int model_layer,
        const int32_t * expert_ids,
        int n_expert_ids) {
    if (!st || !st->cfg.enabled) return;
    if (n_expert_ids <= 0 || !expert_ids) return;

    int idx = -1;
    for (size_t i = 0; i < st->layers.size(); ++i) {
        if (st->layers[i].model_layer == model_layer) { idx = (int) i; break; }
    }
    if (idx < 0) return;

    for (int i = 0; i < n_expert_ids; ++i) {
        llama_moe_residency_touch(st, idx, expert_ids[i], nullptr);
    }
}

// ---------------------------------------------------------------------------
// prewarm()
// ---------------------------------------------------------------------------

void llama_moe_residency_prewarm(
        struct llama_moe_residency_state * st,
        const int * const * top_experts) {
    if (!st || !st->cfg.enabled) return;
    if (!st->cfg.prewarm_on_init) return;

    const int K = st->cfg.prewarm_top_k;
    if (K <= 0) return;

    for (size_t il = 0; il < st->layers.size(); ++il) {
        const int * layer_top = top_experts ? top_experts[il] : nullptr;
        auto & lr = st->layers[il];
        for (int k = 0; k < K; ++k) {
            int expert_id;
            if (layer_top) {
                expert_id = layer_top[k];
            } else {
                expert_id = k;
            }
            if (expert_id < 0) continue;
            if (expert_id >= lr.n_expert) continue;
            if (lr.slot_of[expert_id] >= 0) continue;  // already loaded

            // Install in next free slot (no eviction during prewarm).
            int slot = -1;
            for (size_t s = 0; s < lr.cache.size(); ++s) {
                if (!lr.cache[s].occupied) { slot = (int) s; break; }
            }
            if (slot < 0) continue;  // cache full
            lr.token_counter++;
            auto & e = lr.cache[slot];
            e.expert_id    = expert_id;
            e.last_access  = lr.token_counter;
            e.access_count = 0;
            e.loaded_at    = lr.token_counter;
            e.occupied     = true;
            lr.slot_of[expert_id] = slot;
            st->total_touched++;

            const size_t off = (size_t) expert_id;
            for_each_tensor(lr, [&](void * base, size_t stride) {
                safe_madvise((uint8_t *) base + off * stride, stride, MADV_WILLNEED);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// release()
// ---------------------------------------------------------------------------

void llama_moe_residency_release(
        struct llama_moe_residency_state * st) {
    if (!st) return;
    for (auto & lr : st->layers) {
        for (auto & e : lr.cache) {
            if (!e.occupied) continue;
            const size_t off = (size_t) e.expert_id;
            for_each_tensor(lr, [&](void * base, size_t stride) {
                safe_madvise((uint8_t *) base + off * stride, stride, MADV_DONTNEED);
            });
            e.occupied = false;
        }
        for (auto & s : lr.slot_of) s = -1;
    }
    st->layers.clear();
}

// ---------------------------------------------------------------------------
// log_stats()
// ---------------------------------------------------------------------------

void llama_moe_residency_log_stats(
        const struct llama_moe_residency_state * st) {
    if (!st || !st->cfg.enabled) return;

    const uint64_t total = st->total_hits + st->total_misses;
    const double hit_rate = total > 0 ? double(st->total_hits) / double(total) : 0.0;

    LLAMA_LOG_DEBUG(
        "moe-residency: decodes=%llu touches=%llu hits=%llu misses=%llu evictions=%llu hit_rate=%.1f%%\n",
        (unsigned long long) st->decode_count,
        (unsigned long long) st->total_touched,
        (unsigned long long) st->total_hits,
        (unsigned long long) st->total_misses,
        (unsigned long long) st->total_evicted,
        hit_rate * 100.0);
}

// ---------------------------------------------------------------------------
// topk_from_stats()
// ---------------------------------------------------------------------------

bool llama_moe_residency_topk_from_stats(
        const struct llama_context * ctx,
        int k,
        std::vector<std::vector<int>> & out_top) {
    if (!ctx || k <= 0) return false;
    out_top.clear();

    const auto & model = ctx->get_model();
    const int n_layer = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;
    if (n_expert <= 0) return false;

    out_top.resize(n_layer);
    bool any_data = false;

    for (int il = 0; il < n_layer; ++il) {
        const auto * stats = ctx->get_expert_stats(il);
        if (!stats) continue;

        std::vector<std::pair<int, uint64_t>> ranked;
        ranked.reserve(n_expert);
        for (int e = 0; e < n_expert; ++e) {
            ranked.emplace_back(e, stats->activation_count[e]);
        }
        std::sort(ranked.begin(), ranked.end(),
            [](const auto & a, const auto & b) {
                return a.second > b.second;
            });

        std::vector<int> topk;
        topk.reserve(k);
        for (int i = 0; i < k && i < (int) ranked.size(); ++i) {
            if (ranked[i].second == 0) break;
            topk.push_back(ranked[i].first);
            any_data = true;
        }
        if (topk.empty()) {
            for (int i = 0; i < k; ++i) topk.push_back(i);
        }
        out_top[il] = std::move(topk);
    }

    return any_data;
}
