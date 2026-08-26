# vLLM troubleshooting

## CUDA graph capture crashes {#cuda-graph-capture-crashes}

Symptom: crash at `GDN _output_projection all-reduce` / `_SimpleCData.__new__`, or OOM during graph capture.

**Root cause (fixed in current `rdna2_extras`):** TP comm wrappers were not `allow_in_graph`. Fork fix:
`fix(distributed): allow TP comm ops in torch.compile graph capture`.

**First, pull the latest image:**

```bash
docker pull blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0
```

Then try CUDA graphs (fast path):

```bash
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

Mount cache volumes — see [Configuration](../../vllm/configuration.md#cache-volumes-first-boot-is-slow).

**Fallback:** `--enforce-eager`

On multi-GPU AOT cache issues: `VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1`

Low throughput (~4–5 t/s on 27B)? Check for **older image** where AWQ still used Triton — see
[Quantization](../../vllm/quantization.md).

## First boot is extremely slow

Triton and torch.compile JIT on first run. Typical cache: `~/.triton/cache` (~3 GB),
`~/.cache/vllm/torch_compile_cache` (~700 MB).

## GPTQ/AWQ not using RDNA2 kernels

Check logs for `Using RDNA2W4A16LinearKernel`. If you see Triton/Exllama instead:

```bash
export VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel
```

Confirm `-extras` image from current `rdna2_extras`. See [Fork kernel dispatch](../../vllm/fork.md#kernel-dispatch-on-gfx1030).

## vLLM picks the wrong platform / doesn't see my Radeon

Use published [`blivioniag/vllm-rdna`](../../vllm/images.md) images with `patches/*rocm-platform*` fixes rather
than stock upstream builds.

## `Failed to infer device type` / `AMDSMI_STATUS_NOT_INIT`

Symptom: vLLM logs `ROCm platform is not available because no GPU is found` and
`AMDSMI_STATUS_NOT_INIT - Device not initialized` (often right after CUDA/NVML is also missing — that
part is expected on AMD).

Check the Docker device/group block **exactly** — `#vllm-rdna` hits this when `render` is missing:

```yaml
devices: [/dev/kfd, /dev/dri]
group_add: [video, render]    # both — video alone is not enough on many hosts
ipc: host
security_opt: [label=disable]  # or seccomp=unconfined on docker run
```

Host user still needs `render`/`video` as in [General troubleshooting](./general.md). Debug with
`VLLM_LOGGING_LEVEL=DEBUG`. If the same compose worked on an older tag, `docker pull` a known-good
image — a bad rebuild can also fail AMDSMI init.

## `ROCM_ATTN` hangs for hours (Triton compile)

AMD Triton flash-attention compile on gfx1030 can sit there for **hours** (RCCL vs Triton). On `-extras`,
use `--attention-backend RDNA_ATTN` and/or `VLLM_USE_RDNA2_FA=1` instead of `ROCM_ATTN`. See
[Configuration](../../vllm/configuration.md).

`FA_RDNA2` / `RDNA_ATTN` may not appear in the backend list on older `-extras` images or some GPTQ
models (logs only show Triton / ROCM / TurboQuant). Pull the latest `-extras` tag and confirm
`Using RDNA2W4A16LinearKernel` / native FA in startup logs. Qwen3.8-27B AWQ needs **head size 256**
on the fork — see [Quantization](../../vllm/quantization.md#int4-on-gfx1030-no-native-int4-alus).

## Multi-GPU RCCL hangs or cards drop offline

If TP works on one image and dies after a host ROCm bump, check the **ROCm version** before the
model. **7.2.1 through ~7.13** are reported to have a multi-card RCCL bug. Stay on **7.2.0** or
**7.14.0** — see [Installing ROCm](../../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).
