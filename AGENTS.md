# AGENTS.md

**Version:** 3.0
**Date:** 2026-09-05
**Purpose:** Technical reference for CachyLLama development (methodology in `.clio/instructions.md`)

---

## Project Overview

**CachyLLama** is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) focused on performance optimization for AMD APU hardware. It tracks upstream master through periodic merges, then layers on its own divergent work.

- **Languages:** C11 (ggml core), C++17 (llama engine, common utilities), GLSL (Vulkan shaders), Python (conversion scripts, GGUF tooling)
- **Build system:** CMake 3.14+
- **Architecture:** Modular C library with multi-backend hardware acceleration (CPU, Vulkan, CUDA, Metal, HIP, etc.) plus CachyLLama-specific subsystems
- **License:** MIT (same as upstream `llama.cpp`, Copyright (c) 2023-2026 The ggml authors)
- **Upstream:** <https://github.com/ggml-org/llama.cpp>
- **Parent project:** [fewtarius/llama-ai](https://github.com/fewtarius/llama-ai) — runner scripts, GPU detection, benchmark harness

### CachyLLama-specific subsystems

| Subsystem | Location | Description |
|-----------|----------|-------------|
| Persistent SSD-backed KV cache | `common/kv-ssd-cache.cpp`, `common/kv-ssd-system-cache.cpp`, `common/kv_page_manager.cpp` | Three-tier (hot/warm/cold) on-disk KV cache with conversation hashing |
| MoE expert residency | `src/llama-moe-residency.cpp` | madvise-based expert paging for models larger than RAM |
| MoE expert co-activation tracking | `src/llama-moe-coact.cpp` | Persists expert co-activation matrix for prewarm ordering |
| MoE expert activation tracking | C API in `include/llama.h` | Real-time per-layer expert activation counts via `/expert-stats` |
| User isolation | `include/llama.h`, server integration | `user_id` parameter, per-user concurrency cap, slot affinity |
| System prompt cache | `common/kv-ssd-system-cache.cpp` | Cross-conversation system prompt reuse with recurrent state |
| Hybrid MoE checkpoint restore | `src/llama-kv-cache*.cpp`, `src/llama-memory-recurrent.cpp` | Attention-only KV clearing, recurrent state preservation |
| DFlash framework | `src/models/dflash.cpp` | Generic decoder contract for target-architecture-specific drafting |
| Laguna-S-2.1 | `src/models/laguna.cpp` | Sigmoid-routed MoE with shared expert and softplus attention gate |
| DSV4 KV cache | `src/llama-kv-cache-dsv4.cpp` | DeepSeek-V4 sparse attention KV cache variant |
| DSA KV cache | `src/llama-kv-cache-dsa.cpp` | DeepSeek sparse attention KV cache variant |
| Hybrid memory types | `src/llama-memory-hybrid*.cpp`, `src/llama-memory-recurrent.cpp` | Hybrid SSM/attention memory management |
| Host RAM utility | `common/host-ram.h`, `common/host-ram.cpp` | Cross-platform available-RAM query |
| Vulkan FA scratch gate | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Quantized-KV FA dequant-once with host-RAM safety check |

---

## Quick Setup

```bash
# Clone (with submodules for ggml)
git clone --recurse-submodules https://github.com/fewtarius/CachyLLama.git
cd CachyLLama

# Build (Vulkan on Linux AMD, Metal on macOS, default CPU elsewhere)
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)

# Run the server
./build/bin/llama-server -m /path/to/model.gguf

# Run CLI chat
./build/bin/llama-cli -m /path/to/model.gguf
```

> For end-to-end usage (runner scripts, GPU detection, benchmark harness, GTT configuration), use the [parent project](https://github.com/fewtarius/llama-ai) which builds CachyLLama with the right cmake flags automatically.

---

## Architecture

```
                    include/llama.h (Public C API)
                          |
                    src/llama.cpp (API implementation)
                          |
    +----------+----------+----------+----------+----------+
    |          |          |          |          |          |
   src/       src/       src/       src/       src/       src/models/
   llama-     llama-     llama-     llama-     llama-     (per-arch)
   model      context    sampler    kv-cache   memory
    |                        src/llama-moe-residency.cpp
   common/                   src/llama-moe-coact.cpp
   (utilities)               common/kv-ssd-*.cpp
   common/host-ram.{h,cpp}   common/kv_page_manager.cpp
   common/arg.cpp            common/kv-ssd-system-cache.cpp
   common/sampling.cpp       common/preset.cpp
                             common/reasoning-budget.cpp
    ggml/ (Tensor library, Git submodule)
    |
    +----+----+----+----+----+----+----+----+
    |    |    |    |    |    |    |    |    |
   CPU CUDA Metal Vulkan SYCL HIP OpenCL ...
```

### Key modules

| Module | Purpose |
|--------|---------|
| `include/llama.h` | Public C API (includes CachyLLama extensions: MoE tracking, residency, user isolation) |
| `src/llama.cpp` | Core API implementation |
| `src/llama-model.cpp` | Model loading, architecture dispatch |
| `src/llama-context.cpp` | Inference context, graph evaluation, expert activation tracking, residency touch hook |
| `src/llama-kv-cache*.cpp` | KV cache implementations (standard, ISWA, MSA, DSV4, DSA, recurrent) |
| `src/llama-memory*.cpp` | Memory types (hybrid, recurrent, hybrid-iswa) |
| `src/llama-moe-residency.cpp` | MoE expert residency management |
| `src/llama-moe-coact.cpp` | MoE expert co-activation tracking |
| `src/models/*.cpp` | Per-model architecture implementations (130+ models) |
| `common/kv-ssd-cache.cpp` | Persistent SSD-backed KV cache |
| `common/kv-ssd-system-cache.cpp` | Cross-conversation system prompt cache |
| `common/kv_page_manager.cpp` | Page-level cache management |
| `common/host-ram.{h,cpp}` | Cross-platform available-RAM query |
| `common/arg.cpp` | CLI argument parsing (includes CachyLLama flags) |
| `common/sampling.cpp` | Token sampling strategies |
| `common/speculative.cpp` | Speculative decoding (Eagle3, DSpark, DFlash, MTP) |
| `ggml/` | Tensor library with backends (submodule) |

### Directory structure

| Path | Purpose |
|------|---------|
| `include/` | Public C API headers (`llama.h`, `llama-cpp.h`) |
| `src/` | Core llama library (model loading, inference, sampling, KV cache) |
| `src/models/` | Per-model architecture implementations |
| `src/llama-moe-residency.cpp` | MoE expert residency |
| `src/llama-moe-coact.cpp` | MoE expert co-activation tracking |
| `common/` | Shared utilities (arg parsing, sampling, chat, Jinja, PEG parser) |
| `common/kv-ssd-cache.cpp` | SSD-backed KV cache |
| `common/kv-ssd-system-cache.cpp` | System prompt cache |
| `common/kv_page_manager.cpp` | Page management |
| `common/host-ram.{h,cpp}` | Host RAM query |
| `ggml/` | ggml tensor library (submodule: backends, quantization, graph execution) |
| `tools/` | Executable tools (server, CLI, bench, quantize, perplexity) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tests/` | CTest-based C++ unit tests |
| `gguf-py/` | Python GGUF reader/writer library |
| `vendor/` | Vendored dependencies (cpp-httplib, nlohmann/json, miniaudio, stb) |
| `docs/` | Documentation (build guides, architecture, development) |
| `cmake/` | CMake modules and helpers |
| `ci/` | CI run scripts |
| `app/` | Unified `llama` binary target (intentionally not built in the default CachyLLama configuration) |
| `reference/` | Gitignored. Vendored reference checkouts for diff/audit (upstream `llama.cpp`, `strix-halo-llamacpp`) |

---

## Code Style

CachyLLama follows upstream `llama.cpp` conventions — do not deviate.

- **C++17** standard, **C11** for ggml core
- **4 spaces** indentation, no tabs
- **LF line endings**, UTF-8 encoding
- **Vertical alignment** for readability
- Brackets on same line: `if (cond) {`
- Pointer/reference alignment: `void * ptr`, `int & a`
- `snake_case` for functions, variables, and types
- Naming optimizes for **longest common prefix** (e.g., `number_small`, `number_big`)
- Sized integer types in public API: `int32_t`, `uint32_t`
- Declare structs as `struct foo {}` not `typedef struct foo {} foo`
- In C++ omit `struct`/`enum` keyword when unnecessary
- Avoid templates, fancy STL constructs — use basic `for` loops
- Keep it simple, minimal dependencies

**Formatting:** Use `.clang-format` (clang-tools v15+) when in doubt.

**EditorConfig:** Root `.editorconfig` enforces: spaces, indent 4, LF, UTF-8, trailing whitespace trimmed.

### CachyLLama-specific conventions

- CachyLLama additions are marked with `SPDX-License-Identifier: GPL-3.0-or-later` and `Copyright (c) 2026 fewtarius` at the top of each file
- Vulkan shader dispatch code that is env-gated should use the `GGML_VK_DISABLE_*` / `GGML_VK_*` naming convention
- CachyLLama C API functions in `include/llama.h` should use the `llama_` prefix and be grouped with other CachyLLama extensions (after the upstream API section)
- MoE-related code should reference `docs/moe-expert-residency.md` for architecture context
- **Do not** write `Assisted-by:` in commit messages — this is a fork, commits go directly to the CachyLLama git history

---

## Module Naming Conventions

| Prefix | Purpose | Examples |
|--------|---------|----------|
| `llama-*` | Core llama modules | `llama-model`, `llama-context`, `llama-sampler` |
| `ggml-*` | ggml backend modules | `ggml-cpu`, `ggml-cuda`, `ggml-metal`, `ggml-vulkan` |
| `test-*` | Test files | `test-backend-ops`, `test-tokenizer-0`, `test-sampling` |
| `llama-moe-*` | CachyLLama MoE subsystems | `llama-moe-residency`, `llama-moe-coact` |

Source files follow the pattern: `src/llama-{module}.cpp` / `src/llama-{module}.h`

---

## Testing

### C++ unit tests (CTest)

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

Run a specific test binary:

```bash
./build/bin/test-backend-ops
./build/bin/test-sampling
./build/bin/test-tokenizer-0
```

### CachyLLama-specific test coverage

| Test | What it covers |
|------|----------------|
| `test-backend-ops` | ggml operator consistency across backends — **must pass for Lightning Indexer, DSV4_HC, concat_transpose, and mul_mat_id changes** |
| `test-sampling` | Token sampling correctness |
| `test-tokenizer-0` | Tokenizer roundtrip tests |
| `test-chat-template` | Chat template rendering |
| `test-grammar-parser` | GBNF grammar parsing |
| `test-quantize-fns` | Quantization function correctness |

### Vulkan shader testing

The Lightning Indexer shader is validated by `test-backend-ops` on Strix Halo (gfx1151, RDNA3.5, Vulkan 1.4, RADV Mesa 26.2). Before changing `lightning_indexer.comp` or its dispatch in `ggml-vulkan.cpp`:

1. Run `test-backend-ops` and verify all cases pass (was 0/108 before the init-order fix — see [vulkan-init-order.md](docs/vulkan-init-order.md))
2. Test with DeepSeek-V4-Flash IQ3_XXS on a RADV APU to confirm no "Lightning Indexer is not supported" warning
3. The shader's `required_subgroup_size=32` must be passed via `ggml_vk_create_pipeline` on RDNA3 wave64 devices

### Python tests

```bash
cd gguf-py && python -m pytest tests/
python tests/test-tokenizer-0.py
```

### Benchmarking

Use `llama-bench` for parameter sweeps. For end-to-end cache performance benchmarks, use the parent project's `scripts/benchmark.sh` which drives `llama-server` via HTTP and measures cold/warm TTFT.

---

## Commit Format

CachyLLama uses its own git history as the canonical source of truth (no PR workflow — this is a fork). Commits are squash-merged into the main branch with descriptive messages.

```
<module>: <commit title>
```

Examples:

```
vulkan: fix Lightning Indexer init order (108/108 on Strix Halo)
```

```
common: extract host_available_ram() from kv-ssd-cache.cpp
```

```
models: add Laguna-S-2.1 support (decoder_arch = "laguna")
```

**Pre-commit checklist:**

- `ctest` passes for affected tests
- Format check: `git diff --name-only | grep -E '\.(c|cpp|h|hpp)$' | xargs clang-format --dry-run -Werror`
- No `TODO`/`FIXME` comments left
- No `ai-assisted/` handoff files staged (`git status` then `git reset HEAD ai-assisted/` if needed)
- Commit message describes what and why; no `Assisted-by:`

---

## Development Tools

```bash
# Build
cmake -B build && cmake --build build -j$(nproc)

# Test
cd build && ctest --output-on-failure

# Server
./build/bin/llama-server -m model.gguf

# CLI
./build/bin/llama-cli -m model.gguf

# Quantize
./build/bin/llama-quantize input.gguf output.gguf Q4_K_M

# Convert
python convert_hf_to_gguf.py /path/to/hf-model

# CI locally
./ci/run.sh

# Search code
grep -rn "pattern" src/ common/ include/
```

---

## Common Patterns

### Environment variables

| Prefix | Purpose |
|--------|---------|
| `GGML_VK_*` | Vulkan backend tuning (shaders, scratch, nodes_per_submit) |
| `GGML_VK_DISABLE_*` | Opt-out flags for individual Vulkan features |
| `LLAMA_ARG_*` | CLI flag equivalents for MoE residency offload |
| `LLAMA_SSD_*` | SSD cache configuration (defaults, not overrides) |

User overrides (that win over the solver in `llama-run.sh` in the parent project) use `*_OVERRIDE` suffix or are passed via CLI flags.

### Memory and storage guarantees

CachyLLama makes different kinds of claims about its optimizations, and they have different levels of support. Use this three-way distinction when documenting or troubleshooting.

**Things CachyLLama actually guarantees:**

- An expert in the R+F cache was selected by the R+F algorithm as a good candidate to keep resident.
- A checkpoint on disk has the on-disk format we wrote (atomic write succeeded).
- An explicit `user_id` never matches another `user_id`'s checkpoints (namespace hash is per-user).

**Things requested from the OS but not guaranteed:**

- That `MADV_WILLNEED` actually caused pages to be paged in. The kernel may already have them resident (so the call is a no-op), or may ignore the hint.
- That `MADV_COLD` actually caused the kernel to evict the page. The kernel decides under memory pressure.
- That the working set fits in physical RAM. We *try* to keep it there, but a competing workload can evict our pages regardless.

**Things observed on a particular machine:**

- The `policy_hit_rate` reported in the per-decode log — the LRU+R+F prediction accuracy. The number is correct but the implication ("the kernel kept our pages") is not.
- The aggregate residency ratio from `--moe-residency-debug` — the actual physical residency, measured via `mincore()`. This is the ground truth, but it is specific to the workload, hardware, and competing memory pressure at the time of measurement.

### Verifying residency is doing what it claims

Run the model with `--moe-residency-debug` (Linux only). The per-decode log line shows `policy_hit_rate` and the per-debug-interval line shows the `aggregate ... ratio` (real physical residency). The two should track each other within a few percent on hardware without competing memory pressure. If `policy_hit_rate` is high but `aggregate ratio` is low, the kernel is evicting pages we asked it to keep and the policy is not doing anything. If `advice_einval` in the per-decode summary is non-zero, the kernel is rejecting the `madvise()` advice outright and the policy is definitely not doing anything.

### Verifying SSD cache is doing what it claims

Check that the `kv-ssd` on-disk directory uses the expected `conv_hash` or SHA-256 `user_id` prefix (not a raw `user_id`). For atomic-write guarantees, kill the server with `kill -9` mid-checkpoint-write and verify that the prior valid index is recoverable on next startup. The `tests/test-kv-ssd-user-isolation` binary exercises both properties without needing a real model.

### Independently disabling optimizations

Each CachyLLama-specific optimization can be disabled in isolation so a regression can be bisected to a single cause. The flags:

- `--no-moe-expert-residency` — disables the MoE expert madvise layer. Tracking remains on; the R+F cache and the touch path are skipped.
- `--cache-ssd` / `--no-cache-ssd` — enables or disables the SSD checkpoint cache. The default is "auto" (enabled when the model is large enough to benefit).
- `--no-mmap` — implicit: requires `--no-moe-expert-residency` because the madvise layer operates on the mmap'd model file.
- `LLAMA_ARG_NO_FSYNC=1` — skips the fsync on checkpoint writes for lower write latency at the cost of losing the last checkpoint on crash. Use only for benchmarking, not production.
- `GGML_VK_NODES_PER_SUBMIT=N` — Vulkan command-buffer batching. The APU default is 8; discrete GPU default is 100. Override when debugging a specific backend issue.

If a workload regresses, the developer should be able to disable exactly one optimization and see the regression go away. If two optimizations share a flag, file a bug — the dependency should be broken.

---

## Maintenance Routines

These are recurring tasks performed periodically to keep CachyLLama's divergence from upstream manageable. When starting a session, check if any are due.

### Upstream merge

CachyLLama merges upstream `llama.cpp` master periodically via `git merge upstream/master`. Before merging:

1. Push to a backup branch: `git branch backup-before-rebase`
2. After the merge, re-check all CachyLLama carries for conflicts — especially `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (which accumulates shader dispatch additions) and `src/llama-arch.cpp` / `src/llama-model.cpp` (which gain per-model architecture entries)
3. The `patches/` directory in the parent project is deprecated — CachyLLama maintains its changes directly in git history

### Patch-set status re-validation

Each upstream merge can change which carries are still needed. Re-check [docs/patch-set-status.md](docs/patch-set-status.md) and:

- For "Merged upstream" rows: drop the local copy on the next merge, rebase local additions onto the upstream version.
- For "Not upstreamed" rows: keep carrying; verify the upstream status hasn't changed.
- For "Upstream added" rows: rebase CachyLLama tuning/gating onto the upstream version when the differences are small enough.

**Watch upstream #24127** (CUDA MMQ refactor) — it added `static_assert((I_) % 32 == 0)` to the CASE macro, so any new `rdna3_5` config must keep `I` as a multiple of 32.

### Adding a new model

1. Add architecture entries in `src/llama-arch.cpp` and `src/llama-arch.h`
2. Implement `src/models/{model}.cpp` following the pattern in [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)
3. Add GGUF metadata keys in `src/llama-model.cpp` if the model has custom hparams
4. Update `convert_hf_to_gguf.py` if the model needs conversion support
5. Run `test-backend-ops` to verify operator consistency

### Adding a new Vulkan shader

1. Add the `.comp` file in `ggml/src/ggml-vulkan/vulkan-shaders/`
2. Register it in `ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt`
3. Add dispatch logic in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
4. Gate behind an env var (`GGML_VK_DISABLE_*` or `GGML_VK_*`)
5. Run `test-backend-ops` to verify correctness

### CachyLLama-specific API additions

New public C API functions go in `include/llama.h` (after the upstream API section) and `src/llama.cpp`. Follow the existing `LLAMA_API` visibility convention. Document with inline comments that describe what, not why — git history handles why.

---

## Anti-Patterns

| Anti-pattern | Why it's wrong | What to do |
|--------------|----------------|------------|
| Adding third-party dependencies | Project minimizes deps intentionally | Use vendored libs in `vendor/` or implement inline |
| Using `typedef struct foo {} foo` | Project convention is `struct foo {}` | Declare as `struct foo {}` |
| Fancy template metaprogramming | Codebase avoids complex STL constructs | Use basic loops and simple patterns |
| Mixing unrelated changes in one commit | History should be scannable | One logical change per commit |
| Ignoring clang-format | Project has strict formatting rules | Run `clang-format`, respect `.editorconfig` |
| Committing handoff files | Session notes are internal | Keep `ai-assisted/` out of git |
| Writing `Assisted-by:` in commits | Fork history is our own | Use descriptive commit messages |
| Hardcoding line numbers in docs | Code shifts, docs go stale | Reference function/struct names, not line numbers |

---

## Key Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview, CLI flags, benchmarks |
| `docs/build.md` | Build instructions for all platforms/backends (upstream) |
| `docs/development/HOWTO-add-model.md` | Adding new model support |
| `docs/development/parsing.md` | PEG parser for model output |
| `docs/development/user-isolation-design.md` | User isolation architecture (`user_id` / `u/` namespace / `--max-concurrent-per-user`) |
| `docs/moe-expert-residency.md` | MoE expert residency mechanism, hit rates, C API |
| `docs/autoparser.md` | Auto-detecting model features (upstream) |
| `docs/ops.md` | ggml operator reference (upstream) |
| `docs/ops/Vulkan.csv` | Vulkan op support matrix (upstream) |
| `docs/vulkan-init-order.md` | Vulkan feature-flag init-order constraint (Lightning Indexer, DSV4_HC) |
| `docs/context-checkpoints.md` | Server context checkpoint ring buffer (LCP / f_keep) |
| `docs/patch-set-status.md` | Third-party carries and upstream status |
| `.clio/instructions.md` | Project methodology (Unbroken Method) |

---

*For project methodology and workflow, see `.clio/instructions.md`*
*For universal agent behavior, see system prompt*
