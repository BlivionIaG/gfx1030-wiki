# Disabling ECC (Pro VRAM)

> **WIP:** Procedure is documented for **Radeon PRO W6800** in
> [lunnova's guide](https://lunnova.dev/articles/amdgpu-disabling-ecc/). `#vllm-rdna` and `#general`
> report the same ECC-on-by-default behavior on **V620**. Treat the two-reboot kernel-param path as
> **community** until you confirm `rocm-smi` on your own cards. See
> [Verification status](../reference/verification.md).

Workstation/data-center Navi 21 cards (**PRO V620**, **PRO W6800**) ship with **on-board ECC** enabled.
That costs about **7% of VRAM** (~2 GB on a 32 GB card). Consumer RX 6800 / 6900 cards have **no ECC**
and already show the full 16 GB.

| ECC | Typical `rocm-smi` VRAM (32 GB Pro card) |
|---|---|
| **On** (factory default) | ~30 700 MiB (~30 GB) |
| **Off** | ~32 768 MiB (full 32 GB) |

`#vllm-rdna` (pinned): extra **~2 GB per GPU** if you disable it. Some members keep ECC on for bit-flip
protection; others disable it for long-context KV cache.

## Check whether ECC is on

```bash
sudo dmesg | grep -i ecc
# "MEM ECC is active" / "GECC is enabled"  → ECC is on
# "MEM ECC is not presented"               → consumer card / no ECC

rocm-smi --showmeminfo vram
# ~30.7 GB total on a 32 GB Pro card → ECC is eating the rest
```

## Linux: `amdgpu.ras_enable=0` (two reboots)

There is no Radeon control-panel toggle on Linux. The community path (lunnova, 6.x kernels) is:

1. Add the kernel parameter **`amdgpu.ras_enable=0`**.
2. Reboot **twice**, leaving the parameter in place both times.

**GRUB (Ubuntu / Debian / Fedora):**

```bash
# Ubuntu/Debian: edit GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub
# Fedora: sudo grubby --update-kernel=ALL --args="amdgpu.ras_enable=0"
sudo nano /etc/default/grub
# …add amdgpu.ras_enable=0 to GRUB_CMDLINE_LINUX_DEFAULT, then:
sudo update-grub   # Ubuntu/Debian
# Fedora uses grubby above instead of update-grub
sudo reboot
```

After the **first** reboot, dmesg should mention that GECC will be disabled on the next boot:

```text
GECC will be disabled in next boot cycle if set amdgpu_ras_enable and/or amdgpu_ras_mask to 0x0
```

Reboot again. On the **second** boot you want:

```text
amdgpu: GECC is disabled
```

and `rocm-smi --showmeminfo vram` should report the full ~32 768 MiB.

**To turn ECC back on:** remove the kernel parameter and reboot twice.

> An older `amdgpu-no-ecc.patch` is **not** needed on 6.x kernels. Keep it only if you are still on 5.x
> and the two-reboot path does nothing — details in
> [lunnova's article](https://lunnova.dev/articles/amdgpu-disabling-ecc/).

## Notes

- **Consumer gfx1030** (RX 6800 / 6800 XT / 6900 XT / 6950 XT): skip this page.
- **Secure Boot / signed kernels:** same caveats as [power tuning](./power.md) — unsigned module
  overrides will not load.
- This does **not** replace the [120 W power-cap patch](./power.md). ECC and the VBIOS power floor are
  independent.

## Related

- [Supported Hardware](../setup/hardware.md)
- [Power Tuning](./power.md)
- [Getting Started](../setup/getting-started.md)
