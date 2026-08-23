# Environment Variables & Quick Reference

A cheat-sheet of the settings that matter most when running ML workloads on gfx1030 / RDNA2.

## Key environment variables

| Variable | Example | What it does |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Makes non-Navi-21 RDNA2 cards run gfx1030 kernels. See [override guide](./hsa_override.md). |
| `HIP_VISIBLE_DEVICES` | `0` | Restrict which GPUs a HIP program sees (hide an iGPU or pick one card). |
| `ROCR_VISIBLE_DEVICES` | `0` | Same idea, at the ROCr runtime level. |
| `PYTORCH_ROCM_ARCH` | `gfx1030;gfx1100;…` | Target arch(es) when **building** PyTorch/vLLM/extensions from source. The RDNA images bake `gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201`. |
| `AMDGPU_TARGETS` | `gfx1030` | Target arch(es) for CMake/HIP builds. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Force PyTorch to use rocBLAS instead of hipBLASLt (works around Navi 21 hipBLASLt gaps). |
| `ROCM_PATH` | `/opt/rocm` | Where ROCm is installed; used by many build systems. |
| `HSA_ENABLE_SDMA` | `0` | Occasionally needed to work around DMA issues on some setups. |

> `10.3.0` is the magic value for gfx1030 because the target decodes as
> `gfx` + `10` (major) `3` (minor) `0` (stepping) → `gfx1030`.

## V620 / gfx1030 PCI identity

Used by the [tuning](./tuning_power.md) scripts to match the right board:

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
- vLLM: `--dtype float16` (see [Running vLLM](./vllm.md)).
- The [`rdna2_extras`](./vllm_fork.md) fork adds quantized (W4A16 / FP8) RDNA2 kernels to cut VRAM and
  sidestep BF16 entirely.
