// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 fewtarius
//
// MoE expert residency tracker for CachyLLama.
//
// Phase 1: tracks which MoE experts are "hot" per layer via a per-layer LRU.
// Uses madvise(MADV_WILLNEED / MADV_DONTNEED) on the existing mmap'd model
// regions to keep hot experts paged in and evict cold ones.
//
// This approach does NOT rewrite tensor->data (the proper Hypura-style Phase
// 2 path). It works by managing which pages of the mmap'd model file are
// resident in the OS page cache.
//
// Limitations:
//   - Does not free virtual address space; the model still occupies VAs.
//     Linux overcommit generally handles this for read-only mappings.
//   - Does not reduce mmap footprint; only controls physical residency.
//   - madvise is advisory on Linux; the kernel may ignore under pressure.

#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_set>
#include <vector>

struct ggml_tensor;

// Per-layer residency state for one MoE layer's expert tensors.
struct llama_moe_layer_residency_internal {
    // Model layer index (used to look up the corresponding expert_stats entry
    // in llama_context). May differ from the vector position in
    // llama_moe_residency_state::layers if the model has non-MoE layers.
    int model_layer = -1;

    // Pointers to the model tensors (gate / up / down, plus fused gate_up
    // for architectures that combine them, e.g. gemma4).
    ggml_tensor * t_gate    = nullptr;
    ggml_tensor * t_up      = nullptr;
    ggml_tensor * t_down    = nullptr;
    ggml_tensor * t_gate_up = nullptr;     // optional

    // Per-expert byte stride = tensor->nb[2]. All tensors in this layer
    // share the same per-expert count (= model n_expert) but may have
    // different strides due to layout (down is transposed).
    size_t gate_stride    = 0;
    size_t up_stride      = 0;
    size_t down_stride    = 0;
    size_t gate_up_stride = 0;

    // Number of experts in this layer (= model.hparams.n_expert).
    int n_expert = 0;

    // Recency+frequency cache (Phase 2). Replaces the Phase 1 deque-based LRU.
    // Each entry tracks when it was last accessed and how many times.
    // Eviction score (lower = more evictable):
    //   0.5 * (1 / (1 + current_token - last_access)) +
    //   0.5 * (access_count / (1 + current_token - loaded_at))
    // This addresses the FlashMoE finding that pure LRU evicts hot
    // experts 34% of the time.
    struct cache_entry {
        int     expert_id      = -1;
        uint64_t last_access   = 0;   // token counter at last touch
        uint64_t access_count  = 0;   // total touches since loaded
        uint64_t loaded_at     = 0;   // token counter when first loaded
        // Whether this slot is occupied (expert_id != -1).
        bool    occupied       = false;
    };
    std::vector<cache_entry> cache;            // size = max_resident_per_layer
    // Reverse map: expert_id -> slot index in `cache`, or -1 if not loaded.
    std::vector<int>        slot_of;           // size = n_expert
    // Token counter incremented per touch_layer_selection. Used to compute
    // recency and frequency scores.
    uint64_t                token_counter = 0;

    uint64_t hits   = 0;
    uint64_t misses = 0;
};

// Internal config (C++-side). The public API uses the POD
// `llama_moe_residency_config` declared in llama.h.
struct llama_moe_residency_internal_cfg {
    bool   enabled               = false;
    size_t max_resident_per_layer = 16;
    bool   prewarm_on_init       = true;
    int    prewarm_top_k         = 8;
    bool   log_per_decode        = true;

    // When true, safe_madvise() logs the first EINVAL/ENOSYS per (advice,
    // layer, expert) tuple and increments advice_failure counters. Set
    // false for production builds where the failures are expected and the
    // log volume would be excessive.
    bool   log_advice_failures   = true;

    // Linux-only debug knobs. Off by default; --moe-residency-debug.
    // debug_sample_interval = 0 disables the periodic mincore() sample.
    int    debug_sample_interval = 0;
    int    debug_max_pages       = 32;
};

// Aggregate residency state. One entry per MoE layer.
struct llama_moe_residency_state {
    llama_moe_residency_internal_cfg cfg;
    std::vector<llama_moe_layer_residency_internal> layers;
    int n_layers = 0;
    int n_expert = 0;
    int n_expert_used = 0;

    uint64_t total_hits    = 0;
    uint64_t total_misses  = 0;
    uint64_t total_evicted = 0;
    uint64_t total_touched = 0;
    uint64_t decode_count  = 0;

    // madvise() observability. advice_success/advice_failure are counted
    // per call to safe_madvise(); advice_einval counts the calls that the
    // kernel rejected as not applicable to the mapping type (e.g. MADV_FREE
    // on a MAP_SHARED file-backed mapping). invalid_mapping counts calls
    // skipped because the region wasn't page-alignable.
    uint64_t advice_success       = 0;
    uint64_t advice_failure       = 0;
    uint64_t advice_einval        = 0;
    uint64_t invalid_mapping      = 0;

    // Circuit breaker: when the first madvise() call returns ENOMEM, the
    // kernel is telling us the system is under memory pressure and can't
    // honor advisory hints. On UMA APUs (Flip 7840U, Strix) where the
    // model file is mmap'd into the GPU-visible budget (VRAM+GTT shares
    // DRAM), this is the steady-state for a model that fills most of
    // the budget. Issuing further madvise() calls is pure syscall
    // overhead with no benefit, so we set madvise_disabled_due_to_pressure
    // and skip future calls. The LRU software policy still tracks
    // touches/evictions for observability, but the kernel is left to
    // manage the page cache on its own. Cleared on release() so a
    // rebuilt state can re-attempt.
    bool madvise_disabled_due_to_pressure = false;
    uint64_t pressure_failure_count = 0;  // ENOMEM count before breaker tripped
};

// Build the residency state from a loaded MoE model. Returns true and
// populates `out` if the model is MoE; returns false and leaves `out`
// empty otherwise (caller should treat the model as non-MoE).
//
// Caller must call llama_moe_residency_release() before destroying `out`.
//
// Reads n_expert / n_expert_used / layer tensors via llama_model accessor
// functions; does NOT mutate the model.
bool llama_moe_residency_build(
        const struct llama_model * model,
        struct llama_moe_residency_internal_cfg cfg,
        struct llama_moe_residency_state * out);

// Pre-warm top-K experts per layer. `top_experts` may be null, in which
// case experts 0..cfg.prewarm_top_k-1 are touched for each layer.
// `top_experts[layer]` must contain at most cfg.prewarm_top_k expert IDs.
// No-op if cfg.prewarm_on_init is false.
void llama_moe_residency_prewarm(
        struct llama_moe_residency_state * st,
        const int * const * top_experts);    // [n_layers][<=prewarm_top_k]

// Touch all experts in `expert_ids` for `model_layer`. The residency state's
// layer entry is found by matching model_layer (NOT the vector position,
// which may differ for models with mixed dense/MoE layers).
void llama_moe_residency_touch_layer_selection(
        struct llama_moe_residency_state * st,
        int model_layer,
        const int32_t * expert_ids,
        int n_expert_ids);

// Mark expert `expert_id` of the residency layer at vector index `layer_idx`
// as hot. Triggers eviction of cold experts when the per-layer LRU exceeds
// cfg.max_resident_per_layer.
//
// `was_already_loaded` (out param, may be null): set to true if the expert
// was already in the LRU (cache hit), false if newly added (cache miss).
void llama_moe_residency_touch(
        struct llama_moe_residency_state * st,
        int layer_idx,
        int expert_id,
        bool * was_already_loaded);

// Tear down state. Phase 1 is a no-op since we don't allocate separate buffers.
void llama_moe_residency_release(
        struct llama_moe_residency_state * st);

// Log a summary line of the current stats. Safe to call repeatedly.
void llama_moe_residency_log_stats(
        const struct llama_moe_residency_state * st);

// Helper: build a list of [n_layers][k] top expert IDs by activation count
// for use with llama_moe_residency_prewarm(). Reads from llama_context's
// expert_stats. Returns true if any layer has >= k observations; false
// otherwise (caller should pass null to prewarm for default behavior).
//
// `out_top[layer][i]` must be freed by the caller; this function does not
// allocate the outer array (caller owns llama_moe_residency_topk_buf).
bool llama_moe_residency_topk_from_stats(
        const struct llama_context * ctx,
        int k,
        std::vector<std::vector<int>> & out_top);

// Linux-only: sample each tracked expert's pages via mincore() and log
// the physical residency ratio alongside the policy state. Returns the
// number of experts sampled. No-op on non-Linux platforms (returns 0).
// max_pages_per_tensor caps the pages sampled per tensor to keep the
// call O(1) - 0 means no cap.
int llama_moe_residency_debug_sample(
        const struct llama_moe_residency_state * st,
        int max_pages_per_tensor);

// Platform-agnostic madvise wrapper. Returns 0 on success, -1 on failure
// with errno set. `advice` accepts the MADV_WILLNEED / MADV_DONTNEED /
// MADV_COLD symbols from <sys/mman.h> on Linux; on Windows the
// implementation maps them onto PrefetchVirtualMemory / VirtualUnlock
// as a non-destructive approximation.
//
// The enum values mirror the Linux <sys/mman.h> constants so the same
// advice integer works on both platforms. Callers in production pass
// the standard MADV_* symbols (provided by <sys/mman.h> on Linux and
// by an internal shim on Windows) - the enum is for the test suite.
enum llama_moe_residency_advice {
    LLAMA_MOE_RESIDENCY_MADV_WILLNEED = 3,  // mirror Linux MADV_WILLNEED
    LLAMA_MOE_RESIDENCY_MADV_DONTNEED = 4,  // mirror Linux MADV_DONTNEED
    LLAMA_MOE_RESIDENCY_MADV_COLD     = 20, // mirror Linux MADV_COLD
};
LLAMA_API int llama_moe_residency_madvise(void * addr, size_t len, int advice);

// Returns the OS page size in bytes. Wrapper around getpagesize() that
// works on both POSIX and Windows. Exposed for the test suite.
LLAMA_API int llama_moe_residency_pagesize(void);