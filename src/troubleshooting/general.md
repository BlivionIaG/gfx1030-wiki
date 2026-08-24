# General troubleshooting

## `torch.cuda.is_available()` returns `False`

1. Confirm ROCm sees the card: `rocminfo | grep -m1 -o 'gfx[0-9]*'`.
2. Confirm you installed the **ROCm** build of PyTorch (`torch.__version__` should end in `+rocmX.Y`).
3. Make sure your user is in the `render` and `video` groups: `groups | grep -E 'render|video'`.
   If not: `sudo usermod -aG render,video "$LOGNAME"` and re-login.

## `"no kernel image is available for execution on the device"`

Set the override:

```sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

See [HSA_OVERRIDE](../../setup/hsa-override.md). If building from source, use `PYTORCH_ROCM_ARCH` /
`AMDGPU_TARGETS` for your real target.

## `hipErrorNoBinaryForGpu` / `Memory access fault`

- Usually wrong/missing arch — apply the override or rebuild.
- Can also mean OOM; check `rocm-smi` and reduce batch size / context.

## hipBLASLt errors

```sh
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

## BF16 is extremely slow

Force **FP16** everywhere (`--dtype float16` in vLLM, `dtype=torch.float16` in PyTorch). See
[Environment variables](../../reference/env-vars.md).

## The iGPU is being selected instead of my discrete card

```sh
export HIP_VISIBLE_DEVICES=0     # index of your discrete GPU in `rocminfo`
```

## First diffusion / MIOpen run is very slow

MIOpen compiles kernels on first use in `~/.cache/miopen`. Later runs are fast.

## Secure Boot blocks the amdgpu-dkms module

Either sign the module or disable Secure Boot.
