# HSA_OVERRIDE for RDNA2 Cousins

Only **Navi 21** cards report as `gfx1030`. The smaller RDNA2 dies use different LLVM targets:

| Card family | LLVM target | Die |
| --- | --- | --- |
| RX 6700 XT / 6750 XT / 6700 | `gfx1031` | Navi 22 |
| RX 6600 / 6600 XT / 6650 XT | `gfx1032` | Navi 23 |
| RX 6500 XT / 6400 | `gfx1034` | Navi 24 |
| RDNA2 iGPUs (Ryzen 6000/7000) | `gfx1035` / `gfx1036` | Rembrandt / Phoenix |

Many ROCm libraries only ship precompiled kernels for a subset of targets. Because every RDNA2 card
shares the same instruction set family, you can tell ROCm to treat your card as gfx1030 and reuse the
gfx1030 kernels.

## The override

```sh
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

`10.3.0` maps to `gfx1030`. Set it in the shell (or systemd unit / container env) **before** launching
any HIP/ROCm program:

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 python my_inference_script.py
```

To make it permanent for your user:

```sh
echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
```

## Why this works (and its limits)

- RDNA2 GPUs (gfx1030–gfx1036) are binary-compatible enough that gfx1030 kernels execute correctly on
  the smaller dies for the vast majority of ML ops.
- It is still a **workaround**. AMD does not officially validate it, and you may hit edge cases —
  particularly in hand-tuned assembly kernels or libraries that query the exact arch at runtime.
- Performance-tuned kernels (e.g. in rocBLAS/Tensile) were tuned for Navi 21's CU count and cache; on a
  smaller die they run correctly but may be sub-optimal.

## Building instead of overriding

If you compile a library yourself, prefer building for your **real** target so you get correctly-tuned
kernels, e.g.:

```sh
# Build for multiple RDNA2 targets at once
export PYTORCH_ROCM_ARCH="gfx1030;gfx1031;gfx1032"
# or for a single card
export AMDGPU_TARGETS=gfx1032
```

For llama.cpp:

```sh
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
```

## Verifying it took effect

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If PyTorch previously errored with `"no kernel image is available for execution on the device"` and now
prints `True`, the override is working.
