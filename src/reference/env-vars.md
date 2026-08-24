# Environment Variables & Quick Reference

> **WIP:** vLLM and llama.cpp tables include Discord-sourced defaults — see
> [Verification status](./verification.md#referencemd).

A cheat-sheet of the settings that matter most when running ML workloads on gfx1030 / RDNA2.

## Key environment variables

| Variable | Example | What it does |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Makes non-Navi-21 RDNA2 cards run gfx1030 kernels. See [override guide](../setup/hsa-override.md). |
| `HIP_VISIBLE_DEVICES` | `0` | Restrict which GPUs a HIP program sees (hide an iGPU or pick one card). |
| `ROCR_VISIBLE_DEVICES` | `0` | Same idea, at the ROCr runtime level. |
| `PYTORCH_ROCM_ARCH` | `gfx1030;gfx1100;…` | Target arch(es) when **building** PyTorch/vLLM/extensions from source. The RDNA images bake `gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201`. |
| `AMDGPU_TARGETS` | `gfx1030` | Target arch(es) for CMake/HIP builds. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Force PyTorch to use rocBLAS instead of hipBLASLt (works around Navi 21 hipBLASLt gaps). |
| `ROCM_PATH` | `/opt/rocm` | Where ROCm is installed; used by many build systems. |
| `HSA_ENABLE_SDMA` | `0` | Occasionally needed to work around DMA issues on some setups. |

### vLLM on gfx1030 (`#vllm-rdna`)

| Variable | Example | What it does |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER` | `0` | Disable aiter fused kernels (community default on RDNA2). |
| `VLLM_RDNA_FORCE_FP16` | `1` | Force FP16 compute paths — avoids slow BF16 emulation on RDNA2. |
| `VLLM_USE_RDNA2_FA` | `1` | Enable native RDNA2 FlashAttention (`-extras` images). |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Use the v2 model runner (recommended on recent images). |
| `VLLM_DISABLED_KERNELS` | `ExllamaLinearKernel,TritonW4A16LinearKernel` | Force GPTQ onto `RDNA2W4A16LinearKernel`. |
| `VLLM_DISABLE_CUSTOM_ALL_REDUCE` | `1` | Disable custom all-reduce (stability on some topologies). |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Worker spawn method — avoids fork issues with ROCm. |
| `GPU_MAX_HW_QUEUES` | `2` | Limit HIP hardware queues (stability tuning). |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Enable AMD Triton FA fallback. |
| `PYTORCH_TUNABLEOP_ENABLED` | `0` / `1` | `0` for reproducible benches; `1` for runtime autotuning. |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | `0` | Disable hipBLASLt in tunableop (pair with `TORCH_BLAS_PREFER_HIPBLASLT=0`). |
| `VLLM_USE_AOT_COMPILE` | `0` | Disable AOT compile on multi-GPU if cache replay causes device-bound errors. |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | Disable torch.compile cache (pair with `VLLM_USE_AOT_COMPILE=0` for TP stability). |

### llama.cpp RDNA2 fork (`#llamacpp`)

| Variable | Example | What it does |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | **Required** to activate the V620/gfx1030 native RDNA2 profile. |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Avoid scratch reclaim issues on long-context runs. |
| `GGML_HIP_RDNA2_AUTO` | `1` | Enable automatic RDNA2 kernel selection. |
| `GGML_HIP_SAFE_STATE_IO` | `1` | Safer HIP state I/O (recommended on V620). |
| `GGML_TP_SHARDED_OUTPUT` | `1` | Sharded output head for tensor parallel (TP2+). |
| `GGML_CUDA_ALLREDUCE` | `nccl` | Use RCCL for tensor-parallel all-reduce (+10% tgen reported). |
| `GGML_HIP_GFX1030_P2P_ALLREDUCE` | `off` / `auto-expanded` | P2P all-reduce tuning; set `off` if RCCL misbehaves. |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` | Disable HIP graphs (some benchmark profiles use this). |
| `NCCL_P2P_LEVEL` | `PXB` / `PHB` | RCCL P2P topology level for all-reduce. |
| `NCCL_P2P_DISABLE` | `0` / `1` | Disable P2P in RCCL (fallback when topology is broken). |

### vLLM compilation / CUDA graphs

| Flag | Example | What it does |
| --- | --- | --- |
| `--compilation-config` | `'{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[]}'` | Enable CUDA graphs (preferred fast path on current images). |
| `--compilation-config` | `'{"mode":"NONE","cudagraph_mode":"FULL","compile_ranges_endpoints":[]}'` | Alternative graph mode without torch.compile. |
| `--compilation-config` | `'{"cudagraph_mode":"NONE"}'` | Disable graphs entirely. |
| `--enforce-eager` | — | Fallback: disable all graph capture. Use only when graphs crash. |

> `10.3.0` is the magic value for gfx1030 because the target decodes as
> `gfx` + `10` (major) `3` (minor) `0` (stepping) → `gfx1030`.

## V620 / gfx1030 PCI identity

Used by the [tuning](../tuning/power.md) scripts to match the right board:

| Board | PCI device | Subsystem | 4-tuple |
| --- | --- | --- | --- |
| Radeon PRO V620 (reference) | `1002:73a1` | `1002:0e34` | `1002:73a1:1002:0e34` |
| RX 6900 XT / 6800 (gfx1030) | `1002:73bf` | varies | — |

```sh
lspci -nn | grep '1002:73a1'   # find V620 reference boards
```

## Handy commands

```sh
rocminfo                              # full agent/GPU info; look for "Name: gfx1030"
rocminfo | grep -m1 -o 'gfx[0-9]*'    # just the target name
rocm-smi                              # live clocks, temps, VRAM, power, utilization
rocm-smi --showmeminfo vram           # VRAM usage
clinfo | grep -i board                # OpenCL board name
/opt/rocm/bin/rocminfo | grep -i wavefront   # confirm wave32 on RDNA2
```

## VRAM rules of thumb (LLMs)

| Card VRAM | Comfortable 4-bit model size |
| --- | --- |
| 16 GB (RX 6800/6800 XT/6900 XT/6950 XT) | 7B–13B, some 14B |
| 32 GB (PRO W6800 / V620) | up to ~30B–34B |

## FP16 vs BF16

RDNA2 has **no fast BF16**. Always prefer **FP16** for hot paths:

- PyTorch: pass `dtype=torch.float16`.
- vLLM: `--dtype float16` (see [Running vLLM](../vllm/overview.md)).
- The [`rdna2_extras`](../vllm/fork.md) fork adds quantized (W4A16 / FP8) RDNA2 kernels to cut VRAM and
  sidestep BF16 entirely.
