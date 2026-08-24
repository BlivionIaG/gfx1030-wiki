# Power Tuning (120 W floor)

The **Radeon PRO V620** (gfx1030) ships with its power limit floor **locked to 250 W** in the VBIOS.
That's wasteful for inference, where the card spends most of its time memory-bound. This page summarizes
how to unlock a **120 W floor** and apply a **180 W boot cap**, following the
[`powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/powertuning) feature of the
[`v620_toolbox`](https://github.com/blivioniag/v620_toolbox) repo.

> Validated on **Fedora 43**, kernels **6.17.6** (pre-7) and **7.1.7** (post-7). This involves patching
> and rebuilding a kernel module — do it at your own risk.

## Why it's needed

The V620's VBIOS declares 250 W as its *minimum* power limit, and the `amdgpu` driver trusts that number,
so any lower cap is rejected:

```bash
echo 180000000 | sudo tee /sys/class/hwmon/hwmonX/power1_cap
# -> Invalid argument
```

The SMU firmware actually accepts values down to 120 W — only the **kernel** is in the way.

## The fix

A tiny patch to `sienna_cichlid_get_power_limit()` in
`drivers/gpu/drm/amd/pm/swsmu/smu11/sienna_cichlid_ppt.c` clamps the reported minimum to 120 W — but only
when the GPU matches the **V620 reference board** by PCI identity:

- PCI device `1002:73a1`, subsystem `1002:0e34`

It does **not** touch the VBIOS, `pp_table`, or `pp_features`, and works on any kernel ≥ 5.15 with any
number of V620s in the host. The canonical patch is
[`patches/v620-powercap-min-120W.patch`](https://github.com/blivioniag/v620_toolbox/blob/master/powertuning/patches/v620-powercap-min-120W.patch)
(portable across pre-7 and post-7 kernels via `noinline`).

```diff
+		if (smu->adev->pdev->vendor == 0x1002 &&
+		    smu->adev->pdev->device == 0x73a1 &&
+		    smu->adev->pdev->subsystem_vendor == 0x1002 &&
+		    smu->adev->pdev->subsystem_device == 0x0e34)
+			sienna_cichlid_v620_min_powercap_fix(smu, min_power_limit);
```

> **Other gfx1030 boards** (RX 6900 XT / 6800 use device `0x73bf`) have different PCI IDs. To power-tune
> those, change the identity match in the patch accordingly.

## Two ways to apply it

The toolbox provides scripts under
[`powertuning/scripts/`](https://github.com/blivioniag/v620_toolbox/tree/master/powertuning/scripts):

1. **Bake a kernel RPM** — `v620-kernel-bake.sh` builds a Fedora dist-git kernel RPM with the patch (and
   the P2P kernel-config delta) baked in. Survives cleanly across reboots.
2. **Out-of-tree module override** — `v620-module-install.sh` builds a patched `amdgpu.ko` against your
   *running* kernel and installs it as an override. Re-run after every kernel update.

Then apply the runtime cap at boot:

- `scripts/v620-cap-apply.sh` — writes the 180 W cap (matches the V620 by PCI ID).
- `systemd/v620-powercap.service` — a oneshot unit that runs `v620-cap-apply.sh 180` at boot.

## Verify

```bash
# The kernel logs the fix on match:
sudo dmesg | grep -i 'V620 powerfix'

# The reported minimum is now 120 W (120000000 µW):
cat /sys/class/hwmon/hwmon*/power1_cap_min

# The toolbox's own check:
sudo ./powertuning/scripts/v620-verify.sh
```

`v620-verify.sh` confirms both the `V620 powerfix` dmesg marker and `power1_cap_min=120000000`.

See the full recipe, prerequisites, and the deep-dive docs
([`docs/POWERCAP.md`](https://github.com/blivioniag/v620_toolbox/blob/master/powertuning/docs/POWERCAP.md))
in the repo.
