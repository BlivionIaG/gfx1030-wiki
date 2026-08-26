# Power Tuning (120 W floor)

The **Radeon PRO V620** (gfx1030) ships with its power limit floor **locked to 250 W** in the VBIOS.
That's wasteful for inference, where the card spends most of its time memory-bound. This page summarizes
how to unlock a **120 W floor** and apply a **180 W boot cap**, following the
[`powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/powertuning) feature of the
[`v620_toolbox`](https://github.com/blivioniag/v620_toolbox) repo.

> Validated on **Fedora 43** (kernels **6.17.6** pre-7 / **7.1.7** post-7) and **Ubuntu 26.04 LTS**
> (kernel `7.0.0-30-generic`). Community confirmation: **Fedora Server 44** + kernel **6.19**, 4× V620,
> powerfix + 180 W cap (`cap_min=120 W`). This involves patching and rebuilding a kernel module — do it
> at your own risk.

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

## Platform paths

| Platform | Toolbox path | How the patch lands |
|---|---|---|
| **Fedora 43** | [`powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/powertuning) | Kernel RPM bake or out-of-tree `amdgpu.ko` override |
| **Ubuntu 26.04** | [`ubuntu_powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/ubuntu_powertuning) | `v620-rebuild-amdgpu` — patches Ubuntu `amdgpu` source and installs to `/lib/modules/.../updates/` |

Both platforms share the same **120 W floor** patch logic, the same
[`v620-cap-apply.sh`](https://github.com/blivioniag/v620_toolbox/blob/master/powertuning/scripts/v620-cap-apply.sh)
runtime script, and the same
[`v620-powercap.service`](https://github.com/blivioniag/v620_toolbox/blob/master/powertuning/systemd/v620-powercap.service)
boot cap. Follow the README in the folder for your distro.

### Ubuntu quick path

```bash
git clone https://github.com/blivioniag/v620_toolbox.git
cd v620_toolbox/ubuntu_powertuning

# Install deps (see ubuntu_powertuning/README.md), then:
sudo cp v620-rebuild-amdgpu /usr/local/sbin/
sudo cp ../powertuning/scripts/v620-cap-apply.sh /usr/local/sbin/
sudo chmod 755 /usr/local/sbin/v620-rebuild-amdgpu /usr/local/sbin/v620-cap-apply.sh

sudo /usr/local/sbin/v620-rebuild-amdgpu "$(uname -r)"
sudo reboot
```

After reboot: `sudo dmesg | grep -i 'V620 powerfix'` and
`../powertuning/scripts/v620-verify.sh`. Install the systemd unit and optional
`kernel/postinst.d/v620-amdgpu` hook so future kernel updates rebuild the module —
see [`ubuntu_powertuning/README.md`](https://github.com/blivioniag/v620_toolbox/blob/master/ubuntu_powertuning/README.md).

> **Secure Boot:** disable it or sign the rebuilt module — unsigned overrides won't load with SB on.

## Two ways to apply it (Fedora)

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
After the override, idle around **~7 W per card** has been reported (Fedora 44, 4× V620).

## Slot power and PSU transients

Unlike many gaming cards, a **V620 reaches TDP from the PCIe slot**, not from extra 8-pin cables.
That makes the **+12 V rail** and slot power delivery more sensitive than a 3080-class swap-in:

- Prefill / tensor-split can trip old or miner PSUs even when average watts look fine — see
  [llama.cpp PSU troubleshooting](../troubleshooting/llama-cpp.md#psu-dies-the-moment-tensor-split-prefill-starts).
- If a new card hard-reboots the host on load, try a gentler SMU ramp before blaming the kernel:

  ```bash
  sudo rocm-smi --setperflevel standard   # community: ~80 W min / ~150 W max, softer ramp
  ```

  (`STANDARD` / `standard` — check `rocm-smi --help` on your ROCm; the enum name varies.)
- Cap at **160 W** or **140 W** if 180 W still trips protection.

See the full recipe, prerequisites, and the deep-dive docs
([`docs/POWERCAP.md`](https://github.com/blivioniag/v620_toolbox/blob/master/powertuning/docs/POWERCAP.md))
in the repo.
