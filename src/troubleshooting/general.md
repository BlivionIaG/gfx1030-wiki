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

## CPU governor hurts host-resident models

Symptom: Flash-Next (or any model with large **CPU-side** tables / offload) has weak prefill; VRAM-only
27B is fine.

On Intel `intel_pstate`, default `powersave` still boosts but **ramps lazily**. Bursty host work
(n-gram hash + gather from a multi-GB host table) finishes before the governor reacts. Community:
`performance` improved Flash-Next PP ~33% on a ~6k prompt; GPU-bound Qwen3.8-27B unchanged; idle
clocks still drop.

```sh
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Persist with a systemd oneshot if needed (resets on reboot). Skip this if every tensor stays in VRAM.

## Unsupported AMD GPU in the box "steals" ROCm

Symptom: env overrides look correct, but ROCm / vLLM acts as if only an old unsupported AMDGPU exists
(e.g. Polaris) and ignores the V620s.

`#general`: some ROCm code paths **punt the entire stack** when they see an unsupported AMD device,
instead of skipping that one card. Workarounds: remove/disable the unsupported device, or hide it with
`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` so only the gfx1030 cards remain visible.
