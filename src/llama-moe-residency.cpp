// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius

#include "llama-moe-residency.h"
#include "llama-moe-coact.h"

#include "llama.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cerrno>

#if defined(_WIN32)
// Windows has no madvise(); the calls below map onto the closest Win32
// working-set hints. WILLNEED becomes PrefetchVirtualMemory (pages the
// range in from the file), COLD/DONTNEED become VirtualUnlock, which
// trims the range from the process working set without invalidating the
// file-backed copy - a non-destructive approximation of MADV_COLD's
// page-level reclaim hint. VirtualUnlock is a coarser, process-wide
// mechanism and is a no-op (returns ERROR_NOT_LOCKED, which the shim
// treats as success) on memory that was never VirtualLock'd.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// Mirror the Linux <sys/mman.h> values so the call sites in this file
// use the same symbols on both platforms. Windows headers do not define
// MADV_*, hence the explicit redeclaration here.
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_COLD     20

static int getpagesize(void) {
    static int page_size = 0;
    if (page_size == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        page_size = (int) si.dwPageSize;
    }
    return page_size;
}

static int madvise(void * addr, size_t len, int advice) {
    switch (advice) {
        case MADV_WILLNEED:
            {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes  = len;
                if (!PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    errno = EINVAL;
                    return -1;
                }
                return 0;
            }
        case MADV_COLD:
        case MADV_DONTNEED:
            // VirtualUnlock on a range that was never VirtualLock'd is
            // effectively a no-op and reports ERROR_NOT_LOCKED, which we
            // treat as success (matches the Linux "best-effort hint"
            // contract for these advice values on file-backed mmaps).
            if (!VirtualUnlock(addr, len) && GetLastError() != ERROR_NOT_LOCKED) {
                errno = EINVAL;
                return -1;
            }
            return 0;
        default:
            errno = EINVAL;
            return -1;
    }
}
#else
#include <sys/mman.h>
#include <unistd.h>
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
// Updates the four resident counters in-place. EINVAL is tracked
// separately because it is the most common failure mode for advice
// values not applicable to the current mapping (e.g. MADV_FREE on a
// MAP_SHARED file-backed mapping - invalid per the Linux man page).
static void safe_madvise(void * base, size_t len, int advice,
                         const char * advice_name,
                         uint64_t & c_success,
                         uint64_t & c_failure,
                         uint64_t & c_einval,
                         uint64_t & c_invalid_map,
                         bool log_failures) {
    if (!base || len == 0) {
        c_invalid_map++;
        return;
    }
    uintptr_t p = reinterpret_cast<uintptr_t>(base);
    uintptr_t page_start = p & ~(uintptr_t(getpagesize()) - 1);
    uintptr_t end = p + len;
    uintptr_t page_end = (end + uintptr_t(getpagesize()) - 1) & ~(uintptr_t(getpagesize()) - 1);
    size_t aligned_len = page_end - page_start;
    if (aligned_len == 0) {
        c_invalid_map++;
        return;
    }
    int rc = madvise(reinterpret_cast<void *>(page_start), aligned_len, advice);
    if (rc == 0) {
        c_success++;
        return;
    }
    int e = errno;
    c_failure++;
    if (e == EINVAL) c_einval++;
    if (log_failures) {
        LLAMA_LOG_WARN(
            "moe-residency: madvise(%s) failed: addr=%p len=%zu errno=%d (%s)\n",
            advice_name ? advice_name : "?",
            reinterpret_cast<void *>(page_start),
            aligned_len, e, strerror(e));
    }
}

template <typename Fn>
static void for_each_tensor(const llama_moe_layer_residency_internal & lr, Fn fn) {
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
    const int n_expert_used = hparams.n_expert_used();
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
            // MADV_COLD (Linux 5.4+): mark the page cache copy as a more
            // probable reclaim target under memory pressure. The page is
            // *not* invalidated - if the kernel keeps it, a future touch
            // is still a page-cache hit; if it reclaims, a re-fault is
            // required but the file-backed mmap serves it.
            //
            // Previously this code used MADV_FREE, which is invalid for
            // MAP_SHARED file-backed mappings (the only kind the model
            // loader creates - see llama-mmap.cpp) and was silently
            // returning EINVAL. The earlier "MADV_FREE fixed the
            // regression" reading was actually the kernel correctly
            // no-op'ing the advice; throughput came back because the
            // residency layer stopped interfering. MADV_COLD is the
            // right knob: it's a documented, non-destructive hint
            // valid for this mapping type.
            for_each_tensor(lr, [&](void * base, size_t stride) {
                safe_madvise(reinterpret_cast<uint8_t *>(base) + eoff * stride,
                             stride, MADV_COLD, "MADV_COLD",
                             st->advice_success, st->advice_failure,
                             st->advice_einval, st->invalid_mapping,
                             st->cfg.log_advice_failures);
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
        safe_madvise(reinterpret_cast<uint8_t *>(base) + off * stride,
                     stride, MADV_WILLNEED, "MADV_WILLNEED",
                     st->advice_success, st->advice_failure,
                     st->advice_einval, st->invalid_mapping,
                     st->cfg.log_advice_failures);
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
                safe_madvise(reinterpret_cast<uint8_t *>(base) + off * stride,
                             stride, MADV_WILLNEED, "MADV_WILLNEED",
                             st->advice_success, st->advice_failure,
                             st->advice_einval, st->invalid_mapping,
                             st->cfg.log_advice_failures);
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
                safe_madvise(reinterpret_cast<uint8_t *>(base) + off * stride,
                             stride, MADV_DONTNEED, "MADV_DONTNEED",
                             st->advice_success, st->advice_failure,
                             st->advice_einval, st->invalid_mapping,
                             st->cfg.log_advice_failures);
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
    const double policy_hit_rate = total > 0
        ? double(st->total_hits) / double(total) : 0.0;
    const uint64_t advice_total = st->advice_success + st->advice_failure;
    const double advice_ok = advice_total > 0
        ? double(st->advice_success) / double(advice_total) : 0.0;

    // Note: hit_rate is the SOFTWARE POLICY hit rate, not physical residency.
    // advice_einval counts madvise calls the kernel rejected (e.g. on
    // mapping types that don't support the advice). If einval > 0 here
    // the policy is not actually doing anything.
    LLAMA_LOG_INFO(
        "moe-residency: decodes=%llu touches=%llu policy_hits=%llu policy_misses=%llu "
        "evictions=%llu policy_hit_rate=%.1f%% "
        "madvise: ok=%llu fail=%llu einval=%llu invalid_map=%llu ok_ratio=%.1f%%\n",
        (unsigned long long) st->decode_count,
        (unsigned long long) st->total_touched,
        (unsigned long long) st->total_hits,
        (unsigned long long) st->total_misses,
        (unsigned long long) st->total_evicted,
        policy_hit_rate * 100.0,
        (unsigned long long) st->advice_success,
        (unsigned long long) st->advice_failure,
        (unsigned long long) st->advice_einval,
        (unsigned long long) st->invalid_mapping,
        advice_ok * 100.0);
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

// ---------------------------------------------------------------------------
// debug_sample_residency()
// ---------------------------------------------------------------------------
// Linux-only: walks the current R+F cache for each layer and calls
// mincore() on each expert's pages to report actual physical residency
// (the kernel page cache). This is the only way to verify that the
// software policy is actually translating into the residency behavior
// the policy claims. Off by default; gated by --moe-residency-debug.
// Returns the number of experts sampled.
//
// Reports each expert's policy_state (HOT/COLD/UNKNOWN), resident_pages,
// total_pages, and residency_ratio. For each tensor in the expert
// (gate/up/down) we sample up to --moe-residency-debug-interval
// max-pages to keep the call cheap. The result is the AVERAGE across
// the three tensors, weighted by their page counts.
#ifdef __linux__
#include <sys/mman.h>

int llama_moe_residency_debug_sample(
        const struct llama_moe_residency_state * st,
        int max_pages_per_tensor) {
    if (!st || !st->cfg.enabled) return 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return 0;

    int sampled = 0;
    long long total_resident = 0;
    long long total_pages    = 0;

    for (const auto & lr : st->layers) {
        for (const auto & e : lr.cache) {
            if (!e.occupied) continue;
            const int expert_id = e.expert_id;
            const size_t off = (size_t) expert_id;
            long long exp_resident = 0;
            long long exp_pages    = 0;
            int tensors_seen = 0;
            for_each_tensor(lr,
                [&](void * base, size_t stride) {
                    if (!base || stride == 0) return;
                    uint8_t * p = reinterpret_cast<uint8_t *>(base) + off * stride;
                    uintptr_t page_start = reinterpret_cast<uintptr_t>(p) & ~(uintptr_t(page_size) - 1);
                    size_t n_pages = (stride + page_size - 1) / page_size;
                    if ((size_t) max_pages_per_tensor > 0 &&
                        n_pages > (size_t) max_pages_per_tensor) {
                        n_pages = max_pages_per_tensor;
                    }
                    std::vector<unsigned char> vec(n_pages, 0);
                    int rc = mincore(reinterpret_cast<void *>(page_start),
                                     n_pages * page_size, vec.data());
                    if (rc != 0) {
                        // ENOMEM = not resident, EAGAIN = kernel busy.
                        // Both count as not-resident for the ratio.
                        if (errno == ENOMEM) {
                            // all not-resident
                        } else if (errno == EAGAIN) {
                            // sample could not be completed; treat as
                            // unknown - skip this tensor
                            return;
                        } else {
                            return;
                        }
                    }
                    for (size_t i = 0; i < n_pages; ++i) {
                        // mincore returns 1 in LSB if page is resident.
                        if (vec[i] & 0x01) exp_resident++;
                    }
                    exp_pages    += (long long) n_pages;
                    tensors_seen++;
                });
            if (tensors_seen == 0 || exp_pages == 0) continue;
            const double ratio = (double) exp_resident / (double) exp_pages;
            const char * state = "UNKNOWN";
            // Policy hint: "HOT" means the R+F policy predicted this
            // expert would be accessed (access_count > 0); "COLD" means
            // the policy didn't predict it. This is NOT the same as
            // physical eviction - the actual residency ratio measured
            // by mincore() above may still show pages resident even
            // when state is "COLD" (the kernel may not have reclaimed
            // them yet). "WARM" is never assigned by this code; the
            // name is a legacy label from an earlier version.
            if (e.access_count > 0) state = "HOT";
            else                     state = "COLD";
            LLAMA_LOG_INFO(
                "moe-residency-debug: layer=%d expert=%d policy=%s "
                "resident=%lld/%lld ratio=%.2f\n",
                (int) lr.model_layer, expert_id, state,
                exp_resident, exp_pages, ratio);
            total_resident += exp_resident;
            total_pages    += exp_pages;
            sampled++;
        }
    }
    if (sampled > 0 && total_pages > 0) {
        LLAMA_LOG_INFO(
            "moe-residency-debug: aggregate over %d experts: "
            "resident=%lld/%lld ratio=%.2f\n",
            sampled, total_resident, total_pages,
            (double) total_resident / (double) total_pages);
    }
    return sampled;
}

#else  // !__linux__

int llama_moe_residency_debug_sample(
        const struct llama_moe_residency_state * /*st*/,
        int /*max_pages_per_tensor*/) {
    return 0;
}

#endif