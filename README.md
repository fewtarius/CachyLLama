# CachyLLama

A performance-focused fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for
running local LLM inference on AMD APU hardware — integrated GPUs, handhelds,
and any system with shared memory. CachyLLama tracks upstream `llama.cpp`
closely but diverges freely where performance is on the line: we cherry-pick
from upstream PRs, borrow from other forks, and ship our own optimizations.
If a change makes inference faster on lower-spec hardware, it has a home here
regardless of where it originated.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Upstream: ggml-org/llama.cpp](https://img.shields.io/badge/upstream-ggml--org%2Fllama.cpp-orange.svg)](https://github.com/ggml-org/llama.cpp)
[![Parent project: fewtarius/llama-ai](https://img.shields.io/badge/parent-fewtarius%2Fllama--ai-green.svg)](https://github.com/fewtarius/llama-ai)

[CachyLLama parent project](https://github.com/fewtarius/llama-ai) / [CLIO agentic client](https://github.com/SyntheticAutonomicMind/CLIO) / [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)

---

## What CachyLLama is

CachyLLama is the C++ inference engine in the
[CachyLLama ecosystem](https://github.com/fewtarius/llama-ai). It is a fork of
`llama.cpp` that tracks upstream master through periodic merges, then layers on
a wide surface of performance work — some developed in-house, much adapted from
upstream PRs and other forks.

If you want the runner scripts, GPU/CPU detection, benchmark harness, and
end-to-end install, use the
[parent project](https://github.com/fewtarius/llama-ai). If you want just the
inference engine with the persistent KV cache, MoE expert residency, Lightning
Indexer, and the rest of the CachyLLama performance stack, you are in the right
place.

## What makes CachyLLama different

Where upstream tracks raw model execution, CachyLLama tracks *end-to-end
agentic throughput* — the metric that matters when you are running 18–30K-token
prompts on 5–20 t/s hardware. We are faster on the critical path: Lightning
Indexer fused ops, DSV4 hyper-connection fusion, quantized-KV flash-attention
dequant-once, and APU-tuned command submission all shave time off the parts of
the pipeline that dominate on shared-memory hardware. We support the same
quantization types and model architectures as upstream — the difference is in
how fast they run and how much of the prompt gets cached, not in what runs.

### Persistent SSD-backed KV cache

Conversation state survives server restarts and power cycles. Three-tier
architecture keeps active conversations in RAM, demotes idle ones to disk, and
restores from disk on cold start:

- **Hot** — current session, in RAM. Instant restore within the same conversation.
- **Warm** — previous sessions, same server run. In RAM until memory pressure
  pushes them to cold.
- **Cold** — on disk. Survives server restarts.

Tiers promote and demote automatically based on turn activity. Per-conversation
ring buffers prevent unbounded disk growth. Kernel readahead
(`posix_fadvise` on Linux, `readahead` on macOS) overlaps SSD I/O with CPU work.
Conversation-hash and model-compatibility-hash checks prevent mismatched
checkpoint restoration.

**System prompt cache.** A global, cross-conversation cache keyed on the first N
tokens of any prompt. First eval writes the state; subsequent requests skip the
entire system prompt re-evaluation. Works for both standard transformer and
hybrid (MoE/SSM) models — the per-position recurrent state in the state file
means a state saved after the full prompt can be restored with `n_past` capped
to the system prompt boundary. Default: 8 entries per model, 30 days unused
before expiry.

**Hybrid MoE checkpoint restore.** Hybrid architectures (Qwen3.5/3.6, Gemma 4,
GLM-4.7) combine attention cells with recurrent state, and the recurrent state
covers all positions in the prompt regardless of how the attention cells are
split. CachyLLama tracks recurrent state separately from attention cells, uses
`n_tokens`-based matching when searching for checkpoints, and exposes
`llama_memory_seq_rm_attn_only` to clear attention cells without disturbing
recurrent state. MLA support for DeepSeek2/DeepSeek3 is included.

### Host-memory prompt cache (`--cache-ram`)

An in-memory ring of serialized KV state blobs, designed for the
**agentic-interleaving** workload: a single agent makes multiple divergent
auxiliary calls (keyword extraction, summarization, sub-questions) before
returning to the main thread. The cache holds prior prompts so the next
auxiliary call can find a divergent-prefix match instead of reprocessing the
full context from scratch.

**Only useful with `--parallel > 1`.** With a single slot, the only slot in
the system is the one being saved and immediately reloaded, so the cache can
never hold state between requests — the just-saved entry is consumed by the
load step in the same call, and the cache ends up empty. CachyLLama skips the
save+load round-trip when `n_parallel <= 1` so the work doesn't happen at all
(no VRAM↔RAM round-trip, no 1-second-per-turn stall). For multi-slot
configurations the cache works as designed: idle slots accumulate divergent
state that the LCP matcher uses to hot-swap into a fresh task.

In single-slot deployments, the in-memory checkpoint ring
(`slot.prompt.checkpoints`, the `deferred_create_final_checkpoint` system)
already covers the same problem space at zero cost — it holds raw KV slices
in VRAM, and the LCP match at slot selection restores them with a same-VRAM
copy rather than a VRAM↔RAM round-trip.

### MoE expert management

Three subsystems work together to run models larger than physical memory:

**Expert activation tracking.** HTTP endpoints (`/expert-stats`,
`/expert-tracking`) plus a C API (`llama_expert_tracking_enable`,
`llama_expert_stats_get`, `llama_model_n_expert`, `llama_model_n_expert_used`)
for reading per-layer expert activation counts and per-decode selections in real
time. The activations feed the residency subsystem.

**Expert SSD residency.** Enabled by `--moe-expert-residency`. The runtime tracks
which experts fire per layer (recency + frequency), uses `madvise(MADV_WILLNEED)`
to prefetch hot expert pages into RAM, and `madvise(MADV_FREE)` to lazily release
cold ones back to the mmap'd file. Lets models whose total footprint exceeds
physical RAM run at near-RAM speed — active ~3% of weights stay paged in, the
rest stays on disk and only pays SSD latency on cold misses. Validated hit rates:
95.5% on Qwen3.6-35B-A3B, 89.5% on GLM-4.7-Flash, 83% on gpt-oss-20b.

**Expert host-RAM offload (`--cpu-moe`).** Routes MoE FFN expert tensors to host
RAM via `tensor_buft_overrides` (LLM_FFN_EXPS_REGEX) while keeping
attention/embedding layers on the GPU. Combined with residency, lets you run
30–90 GB Q5_K_XL / Q8_K_XL MoE quants on hardware with 32 GB or less of unified
memory — the rest stays mmap'd on SSD, only the LRU-resident expert subset lives
in RAM.

**Co-activation tracking.** A per-layer co-activation matrix persists across
sessions at `~/.cachylla/coactivation/{model}.json`. FlashMoE showed pure LRU
evicts hot experts 34% of the time; the matrix informs future prewarm ordering
so jointly-active experts land together.

### User isolation

`user_id` as a first-class request parameter. Routes checkpoints to a `u/`
namespace on disk. Per-user concurrency cap with HTTP 429 enforcement. Slot
affinity (allocation prefers slots already owned by the requesting user for
cache locality). OpenAI SDK compatible via `extra_body`.

### Vulkan performance stack (AMD/RADV)

CachyLLama carries a wide surface of Vulkan optimization work, all env-gated
with opt-out defaults:

**DeepSeek-V4 Lightning Indexer.** A custom fused op for DeepSeek's compressed-KV
attention. The shader dequantizes K into shared memory once per block, reuses it
across all heads, and walks Q per head in the inner loop. Verified 108/108 on
`test-backend-ops` on Strix Halo (gfx1151, RDNA3.5). Two coopmat variants are
committed but not yet wired up — tensor-core acceleration when the dispatch is
ready.

**DeepSeek-V4 hyper-connection fused ops (DSV4_HC\_{PRE,COMB,POST}).** Three
shaders replace softmax-scale-iterate sequences. Hardcoded HC=4. Measured on
DSV4-Flash IQ3_XXS, Nimo (RADV Strix Halo), 2405-token prompt: prefill
144.6 → 168.3 t/s (+16.4%), decode at 2.4k ctx 8.2 → 11.6 t/s (+41.1%).
Merged upstream (commit `ccbc17862`); CachyLLama carries the tuning.

**Quantized-KV flash-attention dequant-once.** Backports
[ggml-org/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494)
with a CachyLLama memory gate. The coopmat1 FA path re-dequantizes the
quantized K/V cache inside every Q workgroup on every prefill step — fine on a
discrete GPU with a big L2, brutal on shared-memory UMA. The fix dequantizes
and transposes into a per-head-contiguous f16 scratch once per layer, then runs
the coalesced f16 FA. Runtime controls: `GGML_VK_NO_FA_SCRATCH_TRANSPOSE=1`
(disables), `GGML_VK_FA_SCRATCH_SAFETY_MB=N` (headroom, default 1024 MiB),
`GGML_VK_FA_SCRATCH_FORCE=1` (bypass host-RAM check).

**APU `nodes_per_submit` auto-lower.** Defaults to 8 on UMA devices (iGPU/APU,
RDNA3 Phoenix) to stay under the 2s amdgpu `lockup_timeout`, keeps 100 on
discrete GPUs. Override via `GGML_VK_NODES_PER_SUBMIT=N`.

**Subgroup size pinning.** Pins a 32-wide subgroup for coopmat1 FA where
narrowing is free on RDNA3 (wave64 hardware, 32-wide logical).

**Concat transpose, MMID row-lists.** Infrastructure shaders for delta-net
dim-0 concat and grouped-GEMM redesign, both env-gated and enabled by default.

### CPU ISA auto-detection

The upstream Vulkan build defaults to `GGML_NATIVE=OFF` and `GGML_AVX512=OFF`,
leaving AVX-512 code paths compiled out on Zen 4 hardware that supports them.
CachyLLama's parent project `scripts/detect-gpu.sh` reads `/proc/cpuinfo` (or
`sysctl` on macOS), picks the highest ISA level the CPU supports (`avx512_bf16`
on Zen 4, `avx512_vnni` on Sapphire Rapids, `avx2` on Zen 3, etc.), and
`scripts/rebuild.sh` wires the matching cmake flags into the build.

### DFlash and Laguna model support

**DFlash** (`src/models/dflash.cpp`): A generic framework for target-
architecture-specific decoder contracts. Gated via `dflash.decoder_arch` metadata
(`LLM_KV_DECODER_ARCH`). Currently supports `"laguna"`.

**Laguna-S-2.1** (`src/models/laguna.cpp`): sigmoid-routed MoE with score-
correction bias, one shared expert, softplus attention output gate, QK-norm,
and per-layer-type RoPE. Supports both XS.2 (hybrid full/SWA with per-head
gate) and M.1 (full-attention with per-element gate) variants.

### `common::host_available_ram()`

Extracted from duplicate implementations in `kv-ssd-cache.cpp` and
`kv_page_manager.cpp` into `common/host-ram.{h,cpp}`. Provides a cross-platform
available-RAM query used by the FA scratch gate and auto-sizing code paths.

---

## Benchmarks

### Strix Halo (Nimo Axis N161)

Radeon 8060S, Vulkan backend. Speedup is prompt eval speedup: cold `prompt_ms`
divided by warm `prompt_ms` (with warm path restoring the prefix from SSD rather
than re-evaluating it). Full data in the
[parent project benchmarks](https://github.com/fewtarius/llama-ai#benchmarking).

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS (256x8.4B MoE, Lightning Indexer) | 97.0s | 0.34s | **282x** | 6.3-17.5 |
| MiniMax-M2.7 Q2_K_XL (256x4.9B MoE) | 60.3s | 3.19s | **19x** | 12.0-17.2 |
| Qwen3-235B-A22B Thinking-2507 IQ2_M (235B-A22B MoE) | 100.5s | 1.04s | **96x** | 10.6-13.6 |
| Qwen3.5-122B-A10B Q5_K_M (122B-A10B MoE, Mamba hybrid) | 52.9s | 0.31s | **171x** | 9.5-10.6 |
| Qwen3.5-122B-A10B UD-Q4_K_XL (122B-A10B MoE, Mamba hybrid) | 53.5s | 0.28s | **189x** | 23.5-32.7 |
| gpt-oss-120b Q8_K_XL (120B MoE) | 25.0s | 0.24s | **104x** | 18.2-19.9 |
| Qwen3-Coder-Next Q8_K_XL (512x2.5B MoE, Mamba hybrid) | 48.5s | 0.22s | **221x** | 18.9-19.9 |
| Qwen3.6-35B-A3B Q8_K_XL (35B-A3B MoE, Mamba hybrid) | 20.5s | 0.14s | **149x** | 14.2-17.5 |
| GLM-4.7-Flash Q8_K_XL (64x2.6B MoE, MLA) | 85.5s | 0.18s | **478x** | 13.9-20.3 |
| gemma-4-26B-A4B Q5_K_M (26B-A4B MoE, sliding window) | 16.7s | 0.14s | **123x** | 14.0-14.7 |
| Qwen3.6-27B Q8_K_XL (27B dense, Mamba hybrid) | 85.3s | 0.36s | **238x** | 4.7-4.8 |
| Laguna-S-2.1 Q4_K_XL (256x4.5B MoE, DFlash target) | 50.0s | 0.48s | **104x** | 11.2-16.9 |
| gpt-oss-20b Q6_K_XL (20B MoE) | 11.1s | 0.10s | **112x** | 22.3-23.8 |

### Ayaneo Flip KB

Radeon 780M, 32GB unified memory, Vulkan backend.

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| gemma-4-26B Q5_K_M (26B, dense) | 63.8s | 0.46s | **138.4x** | 14-17 |
| gpt-oss-20b Q6_K_XL (20B, 3.6B active MoE) | 45.9s | 0.24s | **193.5x** | 25-31 |
| Qwen3.6-35B-A3B Q4_K_XL (35B, 3B active MoE) | 70.9s | 0.38s | **187.9x** | 19-24 |

### Real-world CLIO performance (Strix Halo)

Qwen3.6-35B Q8_K_XL, 196K context, 32 threads, Vulkan backend. CLIO evaluates a
project: read files, check git history, run commands, write a final analysis.

| Turn | Prompt tokens | TTFT | Gen t/s |
|------|---------------|------|---------|
| T0 | 21,336 | 26.5s | 36.5 |
| T1 | 1,623 | 2.8s | 37.9 |
| T2 | 5,152 | 8.0s | 37.7 |
| T3 | 4,428 | 7.5s | 38.2 |

**Total: 88 seconds** (32,539 prompt tokens, 1,656 generated). First turn
evaluates the full system prompt cold. Every subsequent turn uses in-memory
checkpoints to restore the cached prefix, evaluating only the new content.
Generation holds steady at 36–38 t/s across all turns.

---

## CachyLLama CLI flags

### SSD cache

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-ssd PATH` | (off) | Enable SSD-backed KV cache |
| `--cache-ssd-checkpoints N` | 64 | Max checkpoints per slot |
| `--cache-ssd-hot-window N` | 16384 | Always-keep window in tokens |
| `--cache-ssd-warm-window N` | 32768 | Keep-in-RAM window in tokens |
| `--cache-ssd-max-cold N` | 0 | Max cold tier checkpoints (0 = unlimited) |
| `--cache-ssd-page-size N` | 1024 | Tokens per page (512 / 1024 / 2048) |
| `--cache-ssd-max-conversations N` | 16 | Max conversation directories |
| `--cache-ssd-hot-ram N` | auto | Hot tier RAM budget in MiB (0 = auto) |
| `--cache-ssd-warm-ram N` | auto | Warm tier RAM budget in MiB (0 = auto) |
| `--cache-ssd-cold-maxsize N` | 0 | Global cap on total cold tier bytes across all conversations in MiB (0 = unlimited). When exceeded, oldest conversations are evicted as whole directories. |

### System prompt cache

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-ssd-system-prompts N` | 8 | Max global system prompt entries cached for reuse across conversations |
| `--cache-ssd-system-max-days N` | 30 | Expire system prompt entries unused for N days |

### Host-memory prompt cache

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-ram N` | 8192 | Host-memory prompt cache size in MiB. Only useful with `--parallel > 1`; with a single slot the save+load round-trip is a no-op (skipped automatically). Set to 0 to disable. `-1` for no limit. |
| `--cache-idle-slots` | enabled | Save idle slots to the prompt cache on new task. Requires `--cache-ram > 0`. With `--kv-unified`, idle slots are also cleared to make room for new tasks. |
| `--parallel N` (`-np`) | 1 | Number of server slots. Each slot holds its own KV cache; multiple slots let several conversations run concurrently without blocking each other. CachyLLama's parent launcher (`llama-run.sh`) defaults to 1; raise to 2+ to host multiple agent sessions. Per-slot KV memory cost is `n_ctx * layer_count * head_dim * 2 (K+V) * dtype_bytes`. |

### User isolation

| Flag | Default | Description |
|------|---------|-------------|
| `--max-concurrent-per-user N` | 0 | Per-user slot cap (0 = unlimited) |

When the cap is hit, the server returns HTTP 429:

```json
{
  "error": {
    "code": 429,
    "message": "User 'tenant-42-user-7' has reached the concurrent request limit (2)",
    "type": "rate_limit_error"
  }
}
```

To identify a request, pass `llama_user_id` in the request body. OpenAI SDK
callers use `extra_body={"llama_user_id": "..."}`. Validated to
`^[a-zA-Z0-9\-_]+$` with a 512-char ceiling.

### Vulkan APU/iGPU tuning

| Flag | Default | Description |
|------|---------|-------------|
| `GGML_VK_NODES_PER_SUBMIT` | auto | Override automatic `nodes_per_submit` (lower values feed RDNA3 iGPUs more frequently) |
| `GGML_VK_DISABLE_LIGHTNING_INDEXER` | (unset) | Set to `1` to disable the DeepSeek-V4 Lightning Indexer fused op |
| `GGML_VK_DISABLE_DSV4_HC[_COMB\|_PRE\|_POST]` | (unset) | Set to `1` to disable individual DSV4 hyper-connection fused ops |
| `GGML_VK_CONCAT_TRANSPOSE` | (unset) | Set to `0` to disable the concat_transpose shader |
| `GGML_VK_FA_KV_CONTIG` | (unset) | Set to `1` to enable the f16 KV contiguize pass for FA prefill |
| `GGML_VK_MMID_F16B` | (unset) | Set to `1` to enable f16-B mul_mat_id path (env-gated, experimental) |
| `GGML_VK_MMID_WAVE32` | (unset) | Set to `1` to probe 32-wide mul_mat_id on wave64 hardware |

### Quantized-KV flash-attention scratch control (Strix Halo)

| Flag | Default | Description |
|------|---------|-------------|
| `GGML_VK_NO_FA_SCRATCH_TRANSPOSE` | (unset) | Set to `1` to disable the dequant+transpose scratch entirely |
| `GGML_VK_FA_SCRATCH_SAFETY_MB` | 1024 | MiB of host-RAM headroom the scratch allocator requires before allocating |
| `GGML_VK_FA_SCRATCH_FORCE` | (unset) | Set to `1` to bypass the host-RAM check |

The gate runs at every prefill step when K and V are both quantized (q8_0 or
q4_0), N is at least 64 (prefill, not decode), and the tensors are contiguous.
If `required_scratch + GGML_VK_FA_SCRATCH_SAFETY_MB > MemAvailable`, the slow
coopmat1 path runs instead and a one-time warning is logged. The scratch size
grows with context: ~256 MiB at 128k for head_dim 128 (Qwen3-Coder-30B-A3B),
~512 MiB at 128k for head_dim 256 (Qwen3.6-35B-A3B). Tracked upstream as
[ggml-org/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494).

### MoE expert residency / offload

| Flag | Default | Description |
|------|---------|-------------|
| `--moe-expert-residency` / `--no-moe-expert-residency` | disabled | Master switch. Tracks MoE expert activations and uses `madvise` to keep hot experts paged into RAM and cold ones released back to the mmap'd file. Requires `--load-mode mmap` (default). |
| `--moe-resident-per-layer N` | 32 | Max experts kept hot per MoE layer (per-layer LRU size). Must be > 0. |
| `--moe-prewarm-top-k N` | 16 | Experts to prewarm per layer at startup. Set to 0 to disable prewarm. |
| `--moe-residency-debug` `[on\|off]` | off | Linux only. Periodically call `mincore()` on each tracked expert and log the physical residency ratio alongside the software policy state. For development and correctness verification, not production. |
| `--moe-residency-debug-interval N` | 64 | Linux only. Decodes between `mincore()` samples. The `mincore()` call costs O(experts) per sample; tune this to balance observability against overhead. |
| `--cpu-moe` | disabled | Route all MoE FFN expert tensors to host RAM via `tensor_buft_overrides` (LLM_FFN_EXPS_REGEX). Combine with `--moe-expert-residency` on memory-constrained hardware to mmap the full weights from SSD while keeping only the LRU-resident expert subset in RAM. |
| `--n-cpu-moe N` | 0 | Route only the first N MoE layers' expert weights to host RAM (0 = use `--cpu-moe` value). |

Environment-variable equivalents: `LLAMA_ARG_MOE_EXPERT_RESIDENCY`,
`LLAMA_ARG_MOE_RESIDENT_PER_LAYER`, `LLAMA_ARG_MOE_PREWARM_TOP_K`,
`LLAMA_ARG_MOE_RESIDENCY_DEBUG`, `LLAMA_ARG_MOE_RESIDENCY_DEBUG_INTERVAL`,
`LLAMA_ARG_CPU_MOE`, `LLAMA_ARG_N_CPU_MOE`.

Hit rate and latency data per model is in
[`docs/moe-expert-residency.md`](docs/moe-expert-residency.md).

---

## MoE expert tracking API

### `GET /expert-stats`

Per-layer expert activation counts, frequencies, and token counts:

```json
{
  "n_expert": 256,
  "n_expert_used": 8,
  "total_tokens": 1500,
  "tracking_enabled": true,
  "layers": [
    {
      "layer": 0,
      "activations": [
        {"expert": 42, "count": 150, "frequency": 0.0125},
        {"expert": 7,  "count": 148, "frequency": 0.0123}
      ]
    }
  ]
}
```

### `POST /expert-tracking`

Enable/disable tracking and optionally reset counters:

```json
{"enabled": true, "reset": true}
```

### C API

```c
llama_expert_tracking_enable(ctx, true);

// Per-layer stats (returns 0 on success, -1 if tracking disabled)
struct llama_expert_stats stats;
llama_expert_stats_get(ctx, /*layer=*/0, &stats);

// Reset all counters
llama_expert_stats_reset(ctx);

// Model-level constants
int32_t n_expert      = llama_model_n_expert(model);
int32_t n_expert_used = llama_model_n_expert_used(model);
```

The `llama_expert_stats` struct exposes `activations[]` (per-expert count and
frequency) and `token_count` for the layer.

### MoE expert residency C API

```c
// Configure and enable residency
struct llama_moe_residency_config cfg = llama_moe_residency_config_default();
cfg.max_resident_per_layer = 32;  // max experts kept hot per layer
cfg.prewarm_top_k   = 16;      // experts to prewarm at startup
llama_moe_residency_enable(ctx, &cfg);

// Read stats
struct llama_moe_residency_stats rs;
llama_moe_residency_stats_get(ctx, &rs);

// Disable (releases all MADV_WILLNEED pages)
llama_moe_residency_disable(ctx);
```

---

## Building CachyLLama

For end-to-end install (runner scripts, GPU detection, benchmark harness, GTT
configuration), use the [parent project](https://github.com/fewtarius/llama-ai):

```bash
git clone --recurse-submodules https://github.com/fewtarius/llama-ai.git
cd llama-ai
./scripts/rebuild.sh
```

To build just the inference engine from this repo:

```bash
cmake -B build
cmake --build build --config Release -j$(nproc)
```

All standard `llama.cpp` build options are supported. CachyLLama adds no new
build flags — everything is runtime config via the CLI flags above.

---

## Documentation

- [CachyLLama parent project](https://github.com/fewtarius/llama-ai) — runner
  scripts, GPU detection, benchmarks, end-to-end install
- [CLIO](https://github.com/SyntheticAutonomicMind/CLIO) — agentic AI client
  optimized for CachyLLama's persistent cache
- [User isolation design](docs/development/user-isolation-design.md) —
  architecture for the `user_id` / `u/` namespace / `--max-concurrent-per-user`
  features
- [MoE expert residency](docs/moe-expert-residency.md) — `--moe-expert-residency`
  / `--cpu-moe` mechanism, hit rate measurements, C API
- [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp) — the base project
  we fork from

---

## Relationship with upstream llama.cpp

CachyLLama is a fork of `llama.cpp`. The relationship is one of **loose
tracking**: we periodically merge upstream master into CachyLLama, then carry
our own divergent work on top. We do not aim for patch-by-patch parity.

**Upstream-first where it lands cleanly.** When an upstream PR addresses a
performance need (e.g. [PR #25494](https://github.com/ggml-org/llama.cpp/pull/25494)
quantized-KV FA prefill), we track it through merge. When it does not get
upstreamed, CachyLLama carries it indefinitely.

**Third-party borrowing.** We adapt performance work from other forks and PRs
— Nathanw114's flash-attention shader optimizations (15+ commits), gaetan-puleo's
Strix Halo RDNA3.5 ROCm tuning, and more. Each carry is documented in
[AGENTS.md](AGENTS.md) with upstream status and CachyLLama-specific additions.

**Our own work.** The persistent KV cache, MoE expert residency, Lightning
Indexer, DFlash/Laguna, user isolation, and the Vulkan APU tuning are all
CachyLLama originals or CachyLLama-led collaborations.

See [AGENTS.md](AGENTS.md) for the full patch-set table and development
conventions. The full patch-set status table is in
[docs/patch-set-status.md](docs/patch-set-status.md).

---

## License

**CachyLLama source code:** MIT (same as upstream `llama.cpp`, see
[LICENSE](LICENSE), Copyright (c) 2023-2026 The ggml authors). All CachyLLama
additions - the persistent KV cache, MoE expert residency, Lightning Indexer,
Vulkan shader work, DFlash/Laguna model support, user isolation, and the
`common::host_available_ram()` utility - are released under the same MIT terms.

**llama-ai parent project:** GPL-3.0-or-later for source, CC-BY-NC-SA-4.0 for
documentation (see the [parent project LICENSE](https://github.com/fewtarius/llama-ai/blob/main/LICENSE)).

**ROCm components:** Carry AMD's license (see `licenses/roc*` in the ROCm SDK
bundle downloaded by `scripts/rebuild.sh`).
