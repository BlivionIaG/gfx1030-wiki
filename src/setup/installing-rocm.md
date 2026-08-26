# Installing ROCm

`gfx1030` (Navi 21) is on AMD's **officially supported** list for recent ROCm releases on Linux, so in
most cases a stock install "just works" — no override hacks required.

> Always follow the official
> [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/) for your exact
> distro and ROCm version. The commands below are a convenience summary and may drift over time.

## Supported operating systems

ROCm's RDNA2 support targets specific LTS releases. As of recent ROCm versions that typically means:

- Ubuntu 22.04 / 24.04 LTS
- RHEL / Rocky 9.x
- Debian 12 (community, less tested)
- Fedora 43 (wiki-validated for [power](../tuning/power.md) / [P2P](../tuning/p2p.md)); Fedora Server
  44 + kernel 6.19 reported working for the V620 powerfix (community)

Check the [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
page for the version you plan to install.

## Ubuntu quick install (amdgpu-install)

```sh
# 1. Add the amdgpu package repository (replace VERSION with the ROCm release you want, e.g. 6.4.60400-1)
sudo apt update
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_VERSION_all.deb
sudo apt install ./amdgpu-install_VERSION_all.deb
sudo apt update

# 2. Install ROCm (compute use case)
sudo amdgpu-install --usecase=rocm

# 3. Add yourself to the render/video groups so you can access the GPU without root
sudo usermod -aG render,video "$LOGNAME"

# 4. Reboot
sudo reboot
```

## Verify the install

```sh
rocminfo                       # should list your Navi 21 card and "Name: gfx1030"
clinfo | grep -i 'Board\|gfx'  # OpenCL view
rocm-smi                       # live clocks, temps, VRAM, power
```

If `rocminfo` shows `Name: gfx1030`, you're done. If it shows `gfx1031`/`gfx1032`/etc., your card is a
smaller RDNA2 die — continue to [HSA_OVERRIDE for RDNA2 Cousins](./hsa-override.md).

## Multi-GPU: pin ROCm 7.2.0 or 7.14.0

For **more than one card**, stay on **ROCm 7.2.0** (not 7.2.1+) **or jump to 7.14.0**.
`#vllm-rdna` reports an **RCCL bug from 7.2.1 upward** that shows up as soon as you leave a single
GPU — tensor-parallel hangs, comm failures, or cards dropping offline. The published
[`vllm-rdna`](../vllm/running.md#image-matrix) images already sit on those two bases for that reason.

Do **not** "upgrade within 7.2.x" on a multi-GPU box. If you are already on a broken 7.2.1–7.13
userspace, rebuild or pull a **7.2.0** or **7.14.0** image rather than debugging RCCL on the
in-between releases.

## Notes & gotchas

- **Kernel driver:** ROCm relies on the `amdgpu` kernel module. Very new kernels sometimes ship a driver
  newer than your ROCm userspace expects; the `amdgpu-dkms` package from the amdgpu repo keeps them in
  sync.
- **Secure Boot:** if Secure Boot is enabled, the DKMS module must be signed or it will fail to load.
  The easiest path for a dev box is to disable Secure Boot.
- **Multiple GPUs / iGPU present:** if your CPU also has an RDNA2 iGPU, ROCm may enumerate it. Pin the
  discrete card with `HIP_VISIBLE_DEVICES` (see [Reference](../reference/env-vars.md)).
- **hipBLASLt:** some libraries assume hipBLASLt, which historically had gaps on Navi 21. If a workload
  complains, try `TORCH_BLAS_PREFER_HIPBLASLT=0` (PyTorch) or the workload's equivalent flag.
