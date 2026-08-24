# Troubleshooting

Common errors on gfx1030 / RDNA2 and how to fix them.

## `torch.cuda.is_available()` returns `False`

1. Confirm ROCm sees the card: `rocminfo | grep -m1 -o 'gfx[0-9]*'`.
2. Confirm you installed the **ROCm** build of PyTorch (`torch.__version__` should end in `+rocmX.Y`).
   A default `pip install torch` gives you the CUDA build, which won't work on AMD.
3. Make sure your user is in the `render` and `video` groups: `groups | grep -E 'render|video'`.
   If not: `sudo usermod -aG render,video "$LOGNAME"` and re-login.

## `"no kernel image is available for execution on the device"`

Your card's target isn't gfx1030 and the library lacks kernels for it. Set the override:

```sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

See [HSA_OVERRIDE for RDNA2 Cousins](./hsa_override.md). If you build from source, build for your real
target with `PYTORCH_ROCM_ARCH` / `AMDGPU_TARGETS` instead.

## `hipErrorNoBinaryForGpu` / `Memory access fault`

- Usually the same root cause as above — wrong/missing arch. Apply the override or rebuild.
- A memory access fault can also mean a genuine OOM; check `rocm-smi` and reduce batch size / context.

## hipBLASLt errors

Navi 21 historically had gaps in hipBLASLt. Fall back to rocBLAS:

```sh
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

## BF16 is extremely slow

RDNA2 has no fast BF16. Force **FP16** everywhere (`--dtype float16` in vLLM, `dtype=torch.float16` in
PyTorch). See [Reference](./reference.md). The [`rdna2_extras`](./vllm_fork.md) images also add quantized
RDNA2 kernels (W4A16 / FP8) that avoid BF16 hot paths.

## vLLM CUDA-graph capture crashes (hybrid GDN models)

Symptom: crash during startup or first generation at `GDN _output_projection all-reduce` /
`_SimpleCData.__new__`, or OOM trying to allocate huge KV cache during graph capture.

**First, pull the latest image** — tags are refreshed in place and recent builds fix many graph issues:

```bash
docker pull blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0
```

Then try CUDA graphs (the fast path — do **not** default to `--enforce-eager`):

```bash
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

Alternative:

```bash
--compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL", "compile_ranges_endpoints": []}'
```

Be patient on first boot — Triton JIT + torch-compile cache warmup can take many minutes. Mount cache
volumes (see [Running vLLM](./vllm.md#cache-volumes-first-boot-is-slow)).

**Fallback** if graphs still fail after updating the image:

```bash
--enforce-eager
```

If throughput is still much lower than expected (~4–5 t/s on a 27B), check whether you're on AWQ (Triton
path) vs GPTQ (native `RDNA2W4A16LinearKernel`) — see
[Running vLLM](./vllm.md#quantization-gptq-vs-awq-on-gfx1030).

## vLLM first boot is extremely slow

Triton and torch.compile generate kernels on first run. This is normal — mount cache volumes and wait.
Typical cache sizes: `~/.triton/cache` (~3 GB), `~/.cache/vllm/torch_compile_cache` (~700 MB). Subsequent
boots reuse them. See [Running vLLM](./vllm.md#cache-volumes-first-boot-is-slow).

## vLLM GPTQ not using RDNA2 kernels

Check logs for `Using RDNA2W4A16LinearKernel`. If you see `TritonW4A16LinearKernel` or
`ExllamaLinearKernel` instead:

```bash
export VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel
```

Also confirm you're using an `-extras` image and a **GPTQ** model (AWQ won't hit this kernel). See
[The rdna2_extras Fork](./vllm_fork.md#kernel-dispatch-on-gfx1030).

## llama.cpp KV checkpoint crash on tensor split

Symptom: fatal error in `ggml-backend-meta.cpp` during warmup with tensor split enabled.

Fix: disable checkpoints with `--ctx-checkpoints 0`. This is a known issue on both stock llama.cpp and
the RDNA2 fork when using `--split-mode tensor`. See
[RDNA2-optimized fork](./llama_cpp_rdna2_fork.md#notable-limits).

## llama.cpp RCCL all-reduce fails (HIP "operation cannot be performed")

Symptom: `ggml_backend_cuda_comm_allreduce_nccl` crash, `NCCL WARN HIP failure`.

Try in order:

1. Confirm [PCIe P2P](./tuning_p2p.md) is actually working (`rocminfo --support` or P2P test).
2. Set `NCCL_P2P_LEVEL=PHB` (all cards on same root port) or `NCCL_P2P_DISABLE=1`.
3. On the RDNA2 fork: `GGML_HIP_GFX1030_P2P_ALLREDUCE=off` or `GGML_CUDA_ALLREDUCE=none` (slower but
   stable).
4. Check ACS — CPU root-port ACS can block GPU-direct P2P even with `pcie_acs_override` on PCH ports.

## vLLM picks the wrong platform / doesn't see my Radeon

Consumer Radeon cards sometimes aren't detected by stock vLLM's platform logic. The
[`vllm-rdna-docker`](./vllm_images.md) images apply `patches/*rocm-platform*` fixes for this — use a
published `blivioniag/vllm-rdna` image, or an `-extras` image, rather than a stock upstream build.

## The iGPU is being selected instead of my discrete card

RDNA2 laptops/APUs expose an iGPU that ROCm may enumerate. Pin the discrete card:

```sh
export HIP_VISIBLE_DEVICES=0     # index of your discrete GPU in `rocminfo`
```

## First diffusion / MIOpen run is very slow

MIOpen compiles and caches kernels on first use in `~/.cache/miopen`. The first run is slow; later runs
are fast. Don't delete that cache.

## Secure Boot blocks the amdgpu-dkms module

Either sign the module or disable Secure Boot. On a dev box, disabling Secure Boot is the quickest fix.

## Still stuck?

- Re-read the official [ROCm troubleshooting docs](https://rocm.docs.amd.com/).
- Check the [community resources](./resources.md) — Discords and repos below often have RDNA2-specific
  fixes.
- Open a pull request to add your fix here so the next person doesn't have to rediscover it.
