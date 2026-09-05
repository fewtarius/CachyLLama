# Context checkpoint system

The server maintains a per-slot ring buffer of in-memory KV cache snapshots ("context checkpoints") used to skip prompt reprocessing on cached turns (LCP / f_keep optimization). Four interlocking pieces — change one, re-verify the others.

## 1. Two producer paths feeding one `std::list<common_prompt_checkpoint>`

- **`create_checkpoint()`** — mid-prompt snapshots fired during prefill when a batch starts a user message (or `--checkpoint-near-end` is set and we're near the prompt end). Uses the live KV cache's `pos_min` / `pos_max` from `llama_memory_seq_pos_*`, skipping if checkpoints are too close (`checkpoint_min_step`, default 8192).
- **`deferred_create_final_checkpoint()`** — after the first generation token lands, captures a full prompt snapshot at `pos_min = 0`, `pos_max = prompt_n_tokens - 1`. Runs asynchronously so the SSD write doesn't block decode.

Both add with `emplace_back`, so list position equals insertion order.

## 2. Insertion-order ring buffer eviction

When the list reaches `n_ctx_checkpoints` (default 32), the ring buffer pops the FRONT (oldest) and appends the new entry at the back.

Why insertion-order, not "highest pos_min": deferred finals all carry `pos_min == 0`, so the old strict-greater-than comparator always picked `begin()` and recycled the oldest entry every time — a single-slot FIFO with N-1 dead entries. Insertion order breaks the tie and gives a true round-robin across conversation snapshots. The acceptance predicate filters by `pos_min` / `pos_max` regardless of list position, so cycling doesn't affect matching.

Cold-start mid-prompts (created on a fresh slot where `pos_min_thold == 0` makes the first batch's `pos_min` equal 0 too) share the same `pos_min == 0` signature as deferred finals. They're valid LCP snapshots either way — they just consume one of the N ring buffer slots.

## 3. SWA-skipped entries persist across turns

In `get_available()` there's an SWA invalidation block that erases checkpoints whose `pos_max > pos_next` for non-deferred-final snapshots. Deferred-final snapshots (`pos_min == 0`) are preserved across turns. Without this guard, the SWA step would erase every prior deferred final and the ring buffer would never accumulate past 2 entries.

Genuine SWA invalidation still happens inside the LCP acceptance predicate:

```cpp
if (n_swa > 0 && cur.pos_max > pos_next) return false;
```

That predicate is the right place to filter by SWA coverage; the `get_available()` guard exists only to keep the buffer populated, not to make SWA-correctness decisions.

**Don't add a redundant erase here based on SWA coverage** — that path will self-conflict.

## 4. Memory budget (`_ckpt_memory_budget()`)

The two producer paths handle size- and count-based eviction in **different orders**:

- **`deferred_create_final_checkpoint()`** runs size-based eviction *before* count-based eviction. Once a checkpoint exceeds the per-call budget it is dropped, so the count cap never sees it.
- **`create_checkpoint()`** runs count-based eviction (insertion-order pop_front) *before* size-based eviction. The count cap is enforced first; only entries that survive count eviction are candidates for size eviction. It also runs an additional pre-pass that erases checkpoints whose `n_tokens` falls within `checkpoint_min_step` of a prior checkpoint on a different task — those are noise, not state.

The size budget function is the same in both paths:

```cpp
size_t _ckpt_memory_budget() const {
    const size_t default_limit = (size_t)2 * 1024 * 1024 * 1024;  // 2 GiB floor
    if (params_base.n_ctx_checkpoints <= 0) return default_limit;
    // 400 MiB per configured checkpoint = 200 MiB working set * 2 headroom.
    const size_t per = (size_t)params_base.n_ctx_checkpoints * 400 * 1024 * 1024;
    return std::max(default_limit, per);
}
```

The budget **floors** (not caps) at 2 GiB and scales upward with `n_ctx_checkpoints` so the auto-scaled count from `llama-ai/scripts/optimize.sh` (8 base + 1 per 8K above 65K context, capped at 32) actually fires. An earlier version coupled it to `cache_ram` (1%) and unconditionally capped checkpoints at 3-4 — the auto-scaling was a no-op.

Worst case at 32 checkpoints: 32 × 400 MiB = 12.8 GiB (the `std::max` floors at 2 GiB so small checkpoint counts still get a 2 GiB working set as a minimum). For the default 32 checkpoints at typical q8_0 KV (~50 MiB each ~= 1.6 GiB total), the budget is the 12.8 GiB cap, well above the working set.

## 5. LCP acceptance predicate

The walk in `get_available()` / `decode-input` is reverse-iteration (newest first). The first qualifying entry wins. Both deferred finals (`pos_min = 0`) and early-mid-prompts (`pos_min < pos_min_thold`) are accepted; mid-prompts whose `pos_min` falls past `pos_min_thold` but haven't been SWA-shifted are also accepted.

## Auto-scaling (from `llama-ai/scripts/optimize.sh`)

```bash
base_ctx = 65536
base_cp  = 8
scale_per = 8192
max_cp   = 32
if [[ $ctx -gt $base_ctx ]]; then
    extra = (ctx - base_ctx) / scale_per
    SOLVER_CHECKPOINTS = base_cp + extra
fi
[[ $SOLVER_CHECKPOINTS -gt $max_cp ]] && SOLVER_CHECKPOINTS = max_cp
```

| Context | Checkpoints |
| ------- | ----------- |
| <= 65 K | 8           |
| 98 K    | 12          |
| 131 K   | 16          |
| 196 K   | 24          |
| 262 K   | 32 (capped) |

Verified on Nimo (Strix Halo) with Laguna-S-2.1 Q5_K_XL at 131K context: ring buffer saturates at 16, then cycles oldest-first across turns 1..16, 1..16, ... — true round-robin. f_keep climbs monotonically (0.488 -> 0.949 across 20 turns), 18 ckpt-restored events after the first cold turn.
