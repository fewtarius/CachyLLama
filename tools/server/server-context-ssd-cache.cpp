// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
// Server Context SSD Cache Integration

#include "server-context-ssd-cache.h"

#include "server-context.h"
#include "server-task.h"
#include "llama.h"

#include <vector>

namespace llama {

uint64_t server_ssd_cache::store(uint32_t slot_id,
                                 struct llama_context* ctx,
                                 const common_prompt_checkpoint& ckpt,
                                 const llama_token* tokens,
                                 size_t tokens_size,
                                 uint32_t turn_id)
{
    if (!cache_ || !ctx || !ckpt.data_tgt.data()) return 0;

    // Compute full state (recurrent + KV cache) for cold-start recovery.
    // The in-memory ckpt.data is PARTIAL_ONLY (recurrent only), but SSD needs
    // the full state so the attention KV cache is available after cold restart.
    const size_t full_size = llama_state_seq_get_size_ext(ctx, slot_id, 0);
    std::vector<uint8_t> full_data(full_size);
    const size_t n_written = llama_state_seq_get_data_ext(ctx, full_data.data(), full_size, slot_id, 0);
    if (n_written != full_size) {
        LOG_WRN("SSD cache: full state serialization size mismatch (expected %zu, got %zu)\n",
                 full_size, n_written);
        return 0;
    }

    return kv_ssd_store(cache_, slot_id,
                        full_data.data(), full_data.size(),
                        ckpt.pos_min, ckpt.pos_max,
                        ckpt.n_tokens, turn_id,
                        (const uint32_t*)tokens, tokens_size,
                        cache_->compat_hash);
}

bool server_ssd_cache::load(uint64_t checkpoint_id,
                            struct llama_context* ctx,
                            int32_t& out_pos_min,
                            int32_t& out_pos_max,
                            uint64_t& out_n_tokens)
{
    if (!cache_ || !ctx || checkpoint_id == 0) return false;

    const kv_ssd_checkpoint* meta = kv_ssd_get_meta(cache_, checkpoint_id);
    if (!meta) return false;

    // Load data from cache (promotes to hot tier)
    std::vector<uint8_t> data;
    if (!kv_ssd_load(cache_, checkpoint_id, data)) return false;

    // Restore full state (recurrent + KV cache) without PARTIAL_ONLY
    size_t bytes_restored = llama_state_seq_set_data_ext(
        ctx,
        data.data(),
        data.size(),
        meta->slot_id,
        0);  // no PARTIAL_ONLY: restore both KV cache and recurrent state

    if (bytes_restored == 0) {
        LOG_WRN("SSD cache: llama_state_seq_set_data_ext failed for checkpoint %lu\n",
                 (unsigned long)checkpoint_id);
        return false;
    }

    out_pos_min = meta->pos_min;
    out_pos_max = meta->pos_max;
    out_n_tokens = meta->n_tokens;
    return true;
}

uint64_t server_ssd_cache::find_match(const llama_token* tokens, size_t tokens_size, uint32_t current_turn,
                                        uint64_t max_n_tokens, int32_t n_past) {
    if (!cache_) return 0;
    return kv_ssd_find_match(cache_, (const uint32_t*)tokens, tokens_size, current_turn,
                             max_n_tokens, n_past);
}

uint64_t server_ssd_cache::find_by_slot(uint32_t slot_id, uint64_t min_tokens, uint32_t current_turn) {
    if (!cache_) return 0;
    return kv_ssd_find_by_slot(cache_, slot_id, min_tokens, current_turn);
}

void server_ssd_cache::on_turn_complete(uint32_t turn_id) {
    if (cache_) {
        kv_ssd_on_turn_complete(cache_, turn_id);
    }
}

} // namespace llama