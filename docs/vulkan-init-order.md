# Vulkan init-order: Lightning Indexer and DSV4_HC

On devices that enable `VK_EXT_subgroup_size_control` (Strix Halo, RDNA3 Phoenix, modern Mesa), the four Vulkan feature flags — `lightning_indexer`, `dsv4_hc_comb`, `dsv4_hc_pre`, `dsv4_hc_post` — were previously assigned **before** `subgroup_require_full_support` was finalized. The flag was set from `subgroup_size_control_features.computeFullSubgroups` several lines later, so all four evaluated to `false` at init time and the corresponding pipelines were never built. `supports_op` then returned `false` at runtime, causing fused operations to silently fall back to CPU with the warning:

```
layer N is assigned to device Vulkan0 but Lightning Indexer is assigned to
device CPU (usually due to missing support)
```

## Fix

The four flag assignments were moved to **after** the `subgroup_size_control_features.computeFullSubgroups` probe. Verified on Nimo (Strix Halo, gfx1151, Vulkan 1.4, RADV Mesa 26.2) with DeepSeek-V4-Flash IQ3_XXS — the warning is gone and the 15k-token prefill runs at 135-150 t/s.

## Why the order matters

`VK_EXT_subgroup_size_control` lets the application request a specific subgroup size (32, 64, etc.). Vulkan implementations only honor that request when `computeFullSubgroups` is true — otherwise the implementation is allowed to use a smaller subgroup size than requested, which silently breaks shaders that depend on the requested size for correctness.

The four CachyLLama-original fused ops (Lightning Indexer plus the three DSV4 hyper-connection shaders) all pin `required_subgroup_size = 32` and assume the subgroup size contract holds. Gating them on `computeFullSubgroups` ensures the request is actually granted before we say "yes, this device supports the op."

## Verification checklist

When changing Vulkan init order or adding a new env-gated fused op:

1. Run `test-backend-ops` and verify all ops pass (was 0/108 before the init-order fix on Strix Halo).
2. Test with DeepSeek-V4-Flash IQ3_XXS on a RADV APU to confirm no "Lightning Indexer is not supported" warning.
3. New ops that pin `required_subgroup_size = 32` must be added **after** the `subgroup_require_full_support` probe in `ggml-vulkan.cpp`, with their pipeline creation guarded by the same flag.
