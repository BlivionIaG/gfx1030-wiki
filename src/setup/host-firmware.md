# Host firmware (MMIO High / large BAR / SR-IOV)

> **WIP / community:** Workstation BIOS work for 4× V620 + PLX. Reports are from `#motherboard` and
> `#general` (Sep 2026). Treat every offset as **your dump**, not a copy-paste from a PDF. See
> [Verification status](../reference/verification.md#setuphost-firmwaremd).

Large-BAR cards (V620 is **32 GB** GDDR6, plus a huge prefetchable MMIO window) can fail to **POST**
on older workstation firmware even when the same cards boot on a modern desktop board. This page is
about getting the host to enumerate the GPUs — not about [PCIe P2P](../tuning/p2p.md) after Linux is
up.

## When this applies

Community symptoms:

- Host **POSTs fine** with no GPUs (or one GPU) and **hangs** with 4× V620 behind a **PLX / switch**
  + bifurcation card.
- The same 4× V620 + PLX kit **works** on a recent consumer board (community: Z690-class) but not on
  a **Dell Precision** tower you want for more RAM / more slots.
- `lspci` never appears because firmware never hands off.

If Linux already sees all cards, skip this page and go to [P2P](../tuning/p2p.md) / [Getting
started](../getting-started.md).

## Dell Precision T5820 / T7820 / T7920 (community)

`#motherboard` + `#general` (Sep 12–14 2026): these Xeon-W / dual-socket Precision towers often need
a **hidden MMIO High / large-BAR** change. A circulating public write-up is the
[Precision 7920 MMIO GPU Unlock Guide](https://github.com/user-attachments/files/28905671/Precision_7920_MMIO_GPU_Unlock_Guide.pdf)
(GitHub attachment). Community: **T7820 behaves like T5820** for this class of tweak.

### Operational rules (do these)

1. **Dump your own UEFI variables.** Do not copy a hex blob from the PDF or from someone else's
   BIOS version. Community first-fail: writing another host's dump back.
2. **Confirm the MMIO change without the PLX** (cards in CPU slots, or fewer cards). Then add the
   switch.
3. Prefer **EFI `.nsh` scripts** over typing hex by hand. Community used `setup_var.efi` and
   [UEFITool](https://github.com/LongSoft/UEFITool) against the **matching** BIOS update package
   (example: T7820 **2.52.1**) to find hidden variables.
4. Community variable name (T5820): `SocketCommonRcConfig`, sub-field **`mmiohsize`** set to
   **`0x03` or `0x04`**. Write **base + size together**; keep the existing base. Both sizes have
   been reported as “the one that worked” — A/B on your dump.
5. **Do not factory-reset, clear CMOS, or pull the CMOS battery** after a successful write — that
   clears the change.
6. After Linux boots, continue with [P2P readiness](../tuning/p2p.md#readiness--verification).

### Aperture size (community)

V620 VT-d / IOMMU windows reserve a **very large** prefetchable range per card (community order of
magnitude: hundreds of GB of address space per GPU, not “just 32 GB VRAM”). A **64 GB** MMIO High
window can be too small for **4×** V620. Community: there is often **no 128 GB** option; jump to
**256 GB** if the menu / variable allows it. Mixed large-BAR cards (e.g. adding 32 GB MI50s next to
V620s) made a too-small window worse until 256 GB was set.

### Dual-socket towers

T7820-class boards can route **some slots to CPU2 or the chipset**. Label the slots and keep all
inference GPUs on **one** CPU's root ports when you can. See [Host topology](../tuning/p2p.md#host-topology).

### What actually unblocked 4× V620 + PLX (one host)

Community (`#motherboard`, Sep 14 2026): **Dell Precision T7820**, **4× V620**, **LSI PEX880xx-class
switch** on a **bifurcation** card, guest **Proxmox**.

- MMIO High / `mmiohsize` work was **necessary but not sufficient** — POST still failed with the
  PLX populated.
- The change that then **posted and booted** was **disabling SR-IOV** via **hidden UEFI
  variables** located with UEFITool on that BIOS package.
- That SR-IOV fix is **board-specific**. Do not treat it as a recipe for ASUS / HP / consumer
  boards.

## Other platforms (short)

| Platform | Community note |
|---|---|
| **HP Z4 G4** | Firmware described as **32-bit MMIO** with V620 — BAR / crosstalk pain; dual-CPU Z6/Z8 G4 are a different (more complex) story. `#motherboard` (Sep 16), **Xeon W-2133** host: BIOS **PCIe MIMO Assignment = 32-bit** **and** **ReBAR on**. That 32-bit ceiling would fail to map **64 GB** of VRAM; Linux then **reallocates** with GRUB `pci=realloc=off pci=nocrs iommu=pt amdgpu.aspm=0 amdgpu.gpu_recovery=1`. **Needs verify** on other Z4 BIOS revs. |
| **ASUS WS-Z270 + ReBAR** | `#motherboard` WIP on **MI50** (not V620). **Above 4G on**, **CSM off**, [ReBARUEFI](https://github.com/xCuri0/ReBARUEFI). One card at 32 GB ReBAR, two cards collapsing to 16 GB / POST code **96** is **not** a gfx1030 recipe. Sep 15: `ReBarState` **> 14** with two 32 GB cards threw code **96**; UEFIPatch hit **PciBus** only, not **PciRootBridge**. Suggested A/B: [GpuMMIOFix](https://github.com/Radi0Glitch/GpuMMIOFix) (OS remap, typically **every boot**) — **Needs verify**. |
| **Modern desktop (Z690-class, etc.)** | Often POSTs 4× V620 + PLX **without** hidden MMIO patches (community contrast to Precision). |

BIOS editors and unsigned flashes **void warranty** and can brick the board. Use the vendor
flashback / recovery path if you have one. This wiki does not publish byte patches.

## Tools (public)

| Tool | Use |
|---|---|
| [UEFITool](https://github.com/LongSoft/UEFITool) | Decode the **same** BIOS update you are running; find hidden setup variables. |
| [ReBARUEFI](https://github.com/xCuri0/ReBARUEFI) | Expose / patch ReBAR + related MMIO options when the menu hides them. |
| [GpuMMIOFix](https://github.com/Radi0Glitch/GpuMMIOFix) | OS-side BAR remap above 4 GB (`#motherboard` Sep 15–16). Community: load **each boot**. **Needs verify** — not a substitute for a correct MMIO High / SR-IOV firmware fix on Precision towers. |
| Vendor BIOS **.exe** / capsule for **your** revision | Source of truth for variable layouts. |

## Related

- [Supported hardware](../hardware.md) — card identity
- [Multi-GPU P2P](../tuning/p2p.md) — after the host POSTs
- [Getting started](../getting-started.md)
- [Useful resources](../meta/resources.md)
