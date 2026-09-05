# Patch-set status

CachyLLama diverges from `upstream/master` by carrying third-party work. Re-evaluate this table when upstream merges or upstream PRs close — row status changes weekly.

**Convention:**
- "Merged upstream" — drop our copy on the next upstream merge.
- "Not upstreamed" — keep carrying; re-check upstream status each merge.
- "Upstream added" — upstream has the same feature; ours differs in tuning/gating. Keep both until the differences can be rebased onto the upstream version.

| Carry | Source | Upstream status | CachyLLama-specific additions |
|-------|--------|-----------------|------------------------------|
| Quantized-KV FA prefill dequant (Vulkan) | [Nathanw1014/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | PR open, under review by `jeffbolznv` | Host-RAM gate via `common::host_available_ram()`, three env vars (`GGML_VK_NO_FA_SCRATCH_TRANSPOSE`, `GGML_VK_FA_SCRATCH_SAFETY_MB`, `GGML_VK_FA_SCRATCH_FORCE`), printf-style warning |
| FA dequant-once to q4_0/q4_1/q5_0/q5_1 KV | Nathanw1014 carry | Not upstreamed | Extends the dequant-once path to four additional quant types. Env-gated via `GGML_VK_FA_DEQUANT_ALL=1` |
| FA contiguize strided f16 KV | Nathanw1014 carry | Not upstreamed | Env-gated via `GGML_VK_FA_KV_CONTIG=1` (default off). Falls back to native path if `required_scratch + safety > device-local capacity` |
| Coopmat1 FA P-fragment hoist | Nathanw1014 carry | Not upstreamed | Hoists the P-fragment load out of the `hsv_tile` loop. Measured +5% on Qwen3.6-35B-A3B prefill, Strix Halo |
| Coopmat1 FA Psh query-major | Nathanw1014 carry | Not upstreamed | Stores `Psh` query-major so the GEMM2 A load vectorizes |
| 32-wide subgroup pinning (coopmat1 FA) | Nathanw1014 carry | Not upstreamed | Pins `required_subgroup_size=32` where narrowing is free on RDNA3 wave64 |
| Bound command buffers by memory traffic | Nathanw1014 carry | Not upstreamed | Replaces flops-based ceiling with memory-traffic-based ceiling for UMA fairness |
| Concat transpose shader | Nathanw1014 carry | Not upstreamed | `concat_transpose.comp` for delta-net dim-0 concat. Env-gated via `GGML_VK_CONCAT_TRANSPOSE` (default ON) |
| MMID row-list prepass | Nathanw1014 carry | Not upstreamed | `mmid_row_lists.comp` for grouped-GEMM redesign. Stage 1 of 2 |
| MMID f16-B probe | Nathanw1014 carry | Not upstreamed | Env-gated via `GGML_VK_MMID_F16B=1` (default off) |
| MMID wave32 probe | Nathanw1014 carry | Not upstreamed | Env-gated via `GGML_VK_MMID_WAVE32=1` (default off) |
| MMID scale cache (q5_K, q4_K, superblock-amortized) | Nathanw1014 carry | Not upstreamed | Shared-memory scale cache for mul_mat_id |
| FA MMQ dot product fp32 scaling | Nathanw1014 carry | Not upstreamed | Scales the MMQ dot product in fp32 before narrowing for numerical stability |
| FA split-K reduce shader | Nathanw1014 carry | Not upstreamed | `flash_attn_split_k_reduce.comp` for split-K FA on large prompts |
| FA top-K selection shader | Nathanw1014 carry | Not upstreamed | `flash_attn_top_k.comp` for DeepSeek sparse FA |
| GATED_LINEAR_ATTN | Nathanw1014 carry | Not upstreamed | Implements `GGML_OP_GATED_LINEAR_ATTN` for gated linear attention |
| DeepSeek-V4 hyper-connection fused ops | [ggml-org/llama.cpp#26578](https://github.com/ggml-org/llama.cpp/pull/26578) | **Merged upstream** | Three shaders: `dsv4_hc_{pre,comb,post}.comp`. HC hardcoded to 4. Tunable with `GGML_VK_DISABLE_DSV4_HC[_COMB\|_PRE\|_POST]=1`. Measured: prefill +16.4%, decode +41.1% on DSV4-Flash IQ3_XXS, Nimo |
| DeepSeek-V4 Lightning Indexer | CachyLLama original | Not upstreamed | `lightning_indexer.comp`, `lightning_indexer_cm.comp`, `lightning_indexer_decode_cm.comp`. See [vulkan-init-order.md](vulkan-init-order.md) |
| DSV4 sparse FA gather-to-compact | CachyLLama original | Not upstreamed | Sparse top-k FA for DeepSeek V4 CSA shape. `flash_attn_top_k.comp` |
| FA flash-attn mask optimization | Nathanw1014 carry | Not upstreamed | `flash_attn_mask_opt.comp` for optimized attention mask handling |
| FA MMQ funcs shader | Nathanw1014 carry | Not upstreamed | `flash_attn_mmq_funcs.glsl` shared code for MMQ-based FA |
| Keep DeepSeek lightning-indexer K cache f16 | Nathanw1014 carry | Not upstreamed | Forces f16 key cache under quantized `-ctk` for Lightning Indexer correctness |
| Vulkan APU `nodes_per_submit` auto-lower | CachyLLama original | Not upstreamed (`ggml-vulkan.cpp` still hardcodes 100) | Defaults to 8 on UMA, 100 on discrete. `GGML_VK_NODES_PER_SUBMIT=N` override |
| Strix Halo RDNA3.5 tuning (ROCm/HIP) | gaetan-puleo carry | Upstream added `mmq-config-rdna3-5.cuh`; CachyLLama's has Strix Halo-specific tuning | `I = 64` in all MMQ CASE entries for upstream `#24127` `static_assert((I_) % 32 == 0)` |
| `common::host_available_ram()` | CachyLLama original | None | Extracted from duplicate implementations in `kv-ssd-cache.cpp` and `kv_page_manager.cpp` |
| DFlash framework | CachyLLama original | Not upstreamed | `src/models/dflash.cpp`. Generic decoder contract via `dflash.decoder_arch` metadata. Currently supports `"laguna"` |
| Laguna-S-2.1 | CachyLLama original | Not upstreamed | `src/models/laguna.cpp`. Sigmoid-routed MoE, shared expert, softplus attention gate, QK-norm, per-layer-type RoPE |

**CachyLLama focus downstream:** If a third-party carry lands upstream cleanly, the CachyLLama copy can be dropped on the next `merge upstream/master` and the local additions (memory gate, env overrides, follow-up fixes) rebased onto the upstream version. When a carry does not get upstreamed, CachyLLama carries it indefinitely — re-check upstream status each merge.

**Watch upstream #24127** (CUDA MMQ refactor): it added `static_assert((I_) % 32 == 0)` to the CASE macro, so any new `rdna3_5` config must keep `I` as a multiple of 32.
