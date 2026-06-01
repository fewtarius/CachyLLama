# User Isolation and Scheduling Design

## Status

Draft. Not implemented. Targets the conversation-aware SSD cache work in
`common/kv-ssd-cache.{h,cpp}` and `tools/server/`.

## Problem

Current isolation is content-derived: `conv_hash` is FNV-1a of the first 1024
task tokens. This has four problems:

1. **Identity instability.** Edit the system prompt and the first 1024 tokens
   change, `conv_hash` rotates, and all warm checkpoints for that "user" become
   unreachable except via the fuzzy continuation fallback.
2. **No operator-declared boundary.** The server operator cannot enforce "this
   is user X" - the boundary is whatever the prompt looks like today.
3. **PII in the hash key.** Tokens may contain names, emails, identifiers.
   The hash itself is not reversible but the *content* drives the routing key.
4. **No scheduling isolation.** A single noisy tenant can fill every slot
   (`n_parallel`). The DeepSeek model exposes per-`user_id` concurrency caps
   for exactly this reason.

## Goal

Introduce a first-class `user_id` that:

- Replaces or augments the content-derived `conv_hash` for KV cache routing.
- Carries a per-`user_id` concurrency cap (scheduling isolation).
- Carries a per-`user_id` log key for content safety auditing.
- Is wired through the OpenAI and Anthropic request surfaces.
- Falls back gracefully to content-derived routing when `user_id` is absent.
- Is backwards compatible. Existing deployments with no `user_id` keep working
  via `conv_hash`.

## The Three Isolation Dimensions

| Dimension | DeepSeek term | Current state | Target state |
|---|---|---|---|
| Identity | `user_id` (regex `[a-zA-Z0-9\-_]+`, max 512) | None (content-derived only) | First-class request field |
| KVCache | "KVCache Isolation" | `conv_hash` per conversation dir | `user_id` dir; `conv_hash` is the fuzzy fallback |
| Scheduling | "Scheduling Isolation" | None, global `n_parallel` | `user_id` slot cap, operator-configurable |
| Content safety | "Content Safety Isolation" | None | Per-`user_id` log key in server logs |
| Concurrency | Account-level | `n_parallel` only | `n_parallel` (account) + per-`user_id` cap |

## Request Surface

### OpenAI Chat Completions

Top-level field on the request body, matching OpenAI's own `user` parameter
naming convention would be tempting but conflicts with OpenAI's own field
semantics. Use a llama-specific key to avoid collision:

```json
{
  "model": "...",
  "messages": [...],
  "llama_user_id": "tenant-42-user-7"
}
```

OpenAI SDK callers pass it through `extra_body` as DeepSeek documents:

```python
client.chat.completions.create(
    model="...",
    messages=[...],
    extra_body={"llama_user_id": "tenant-42-user-7"},
)
```

### Anthropic Messages

Anthropic already supports `metadata` as a free-form object. We extract
`metadata.user_id` (existing wiring at `tools/server/server-chat.cpp:543` is
orphaned - we parse it and never use it). Promote that field and rename to
`llama_user_id` for consistency, or keep `user_id` inside `metadata` for
Anthropic callers who already use it that way. Decision: keep `metadata.user_id`
as the Anthropic path since it is what the client library natively supports,
and accept either path on the wire.

### Validation

```
regex:    ^[a-zA-Z0-9\-_]+$
max len:  512
empty:    treat as anonymous bucket (still isolated, still subject to caps)
```

Reject with HTTP 400 on invalid input. Reject with HTTP 429 when the
`user_id` is at its concurrency cap.

## KVCache Routing

### Precedence

When `user_id` is present and valid:

1. Route KV cache to `{SSD_PATH}/{fnv1a(user_id)}/`
2. Cold-start lookup uses `user_id` as the directory key
3. Content-derived continuation matching is **disabled** within the
   `user_id` directory (no fuzzy cross-`user_id` lookups - privacy guarantee)
4. Content-derived continuation matching **remains enabled** for the
   anonymous bucket and for the global fallback path

When `user_id` is absent or empty:

1. Route to the anonymous bucket `{SSD_PATH}/_anonymous/`
2. Cold-start lookup uses the anonymous bucket
3. Content-derived continuation matching is enabled inside the anonymous
   bucket and across anonymous buckets on global fallback

Rationale: explicit `user_id` is a privacy declaration. We must not silently
mix KV state across declared identities. The anonymous bucket keeps the
content-derived behaviour for callers who have not opted in.

### Compatibility With Existing Deployments

`conv_hash` (content-derived) is preserved as a separate code path. It is
still used:

- When `user_id` is not provided.
- As the directory naming scheme for the anonymous bucket if we choose to
  preserve the existing `conv_hash`-based names (recommended for upgrade
  safety: see "Migration" below).

`user_id` and `conv_hash` are not mixed. A request either has a `user_id` and
uses the `user_id` directory, or it does not and uses `conv_hash`.

### Migration

The SSD cache directory layout is already
`{SSD_PATH}/{conv_hash_hex}/`. We have two options for the `user_id` scheme:

A. **Hash the `user_id` the same way** (`fnv1a(user_id)`). Existing dirs
   are named by the same hash, so a `user_id` that happens to hash to the
   same value as a `conv_hash` collides. Operators would need to use a
   different hash function for `user_id` (e.g. SHA-256 truncated, or a
   separate namespace prefix).

B. **Namespace prefix.** `user_id` directories are
   `{SSD_PATH}/u/{fnv1a(user_id)}/`. `conv_hash` directories stay at
   `{SSD_PATH}/{fnv1a(conv_hash)}/`. No collision. Clean separation.

Recommended: **B**. Operators upgrading from a content-only deployment keep
all existing checkpoints. New `user_id` traffic lands in a separate
namespace. We can add a migration tool later if anyone needs to claim an
anonymous cache under a `user_id`.

## Scheduling Isolation

### Configuration

```
--max-concurrent-per-user <N>   # 0 = unlimited (default)
```

Setting this enables per-`user_id` slot accounting. When a request arrives
with a `user_id`, the slot allocator checks the current count of in-flight
requests for that `user_id`. If the count is at or above `N`, the request is
rejected with HTTP 429 and a `Retry-After` header.

The anonymous bucket is subject to the same cap under the bucket name
`_anonymous`, so a deployment that did not previously have a cap does not
get one for free by opting out of `user_id`.

### Slot Allocator Changes

`server_context::get_available_slot` (`tools/server/server-context.cpp:1332`)
becomes `user_id`-aware:

1. Free slots that match the current `user_id` are preferred (cache locality).
2. Among matching free slots, the slot with the highest prompt similarity is
   preferred (existing LCP logic).
3. If no matching slot is free, fall back to a free slot from any `user_id`
   only if it would not push the requesting `user_id` over its cap.
4. If no slot can be allocated without breaking the cap, return `nullptr`
   and the caller emits HTTP 429.

This is a small change to the slot loop and an addition to the slot
metadata (`slot.user_id` field, cleared on release).

### Per-Slot State

Add to `server_slot`:

```cpp
std::string user_id;  // empty if anonymous
```

Set when the slot is assigned. Cleared when the slot is released. Used for
both the routing check and the per-user concurrency counter.

### Concurrency Counter

A `std::unordered_map<std::string, int>` guarded by the existing slot mutex
tracks the in-flight count per `user_id`. Increment on slot assign,
decrement on release. The `_anonymous` key holds the count for requests
without a `user_id`. The total across all keys never exceeds `n_parallel`.

## Content Safety

Add a structured log line on request completion:

```
SRV_INF("request_complete user_id=%s tokens_in=%d tokens_out=%d ...")
```

This is the minimum bar. Operators can grep per `user_id` for audit. The
content itself is not logged - this is a routing key for log filtering.

If deeper content safety tooling is needed later (per-user allow/deny lists,
toxicity scoring integration) that is a separate feature. This design only
adds the log key.

## CLI Parameters

| Flag | Default | Effect |
|---|---|---|
| `--max-concurrent-per-user N` | 0 (unlimited) | Per-`user_id` slot cap. Enforced for both explicit and anonymous buckets. |

The SSD cache parameters (`--cache-ssd-path`, `--cache-ssd-max-conversations`,
etc.) are unchanged. The `user_id` scheme is automatic when SSD caching is
enabled.

## Backwards Compatibility

- No `user_id` provided: behaviour identical to current. Anonymous bucket
  used. `conv_hash` and continuation matching still work.
- Existing SSD cache directories: untouched. `user_id` traffic lands in a
  separate `u/` namespace.
- `metadata.user_id` parsing: the orphaned extraction at
  `server-chat.cpp:543` is completed - the field is now actually routed.
- Slot allocation: unchanged when `--max-concurrent-per-user` is 0.

## Implementation Outline

Roughly 4-5 focused commits:

1. **`common/kv-ssd-cache`**: add `u/` namespace support, accept either a
   raw 64-bit hash or a string `user_id`. Document the precedence rules.
2. **`server-task` / `server-chat`**: add `user_id` field to `server_task`,
   extract from `metadata.user_id` (Anthropic) and `llama_user_id` /
   `extra_body.llama_user_id` (OpenAI). Validate the regex and length.
3. **`server-context`**: thread `user_id` from task into the slot, use it
   in `get_available_slot` for cache lookup and concurrency cap. Add the
   per-user counter map.
4. **`common/arg`**: add `--max-concurrent-per-user`.
5. **HTTP 429 path**: when no slot is available due to the per-user cap,
   return 429 with a `Retry-After` header. Currently the server just queues
   the task; we need a fast-fail path for the cap case.

## Open Questions

- **Default cap for the anonymous bucket.** If `--max-concurrent-per-user`
  is set, do we apply it to `_anonymous`? Recommended yes, for fairness,
  but some operators may want anonymous to be uncapped and only declared
  `user_id`s to be capped.
- **Per-model caps.** DeepSeek does per-model concurrency. We have one
  model per server, so this is not relevant today. If multi-model routing
  lands later it will need to be revisited.
- **`Retry-After` semantics.** Streaming requests mid-flight when the cap
  drops the next request. Not a real concern for the cap itself (it only
  blocks new requests) but worth a test.

## References

- DeepSeek docs: <https://api-docs.deepseek.com/quick_start/rate_limit>
- Current `conv_hash` definition: `common/kv-ssd-cache.cpp:41`
- Page manager: `tools/server/server-context-page-manager.{h,cpp}`
- Orphaned `metadata.user_id` extraction: `tools/server/server-chat.cpp:543`
- Slot allocation: `tools/server/server-context.cpp:1332`
