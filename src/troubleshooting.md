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
