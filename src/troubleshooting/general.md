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

See [HSA_OVERRIDE](../setup/hsa-override.md). If building from source, use `PYTORCH_ROCM_ARCH` /
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
[Environment variables](../reference/env-vars.md). `#general` (Sep 28): **ComfyUI** image/video
graphs that default to bf16 are a poor V620 fit for the same reason — prefer fp16 / int8
workflows ([hardware](../setup/hardware.md)).

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
(e.g. Polaris / WX4100) and ignores the V620s — or `rocminfo` crashes with topology errors about an
unsupported KFD node.

`#general` / `#llamacpp`: some ROCm code paths **punt the entire stack** when they see an unsupported
AMD device, instead of skipping that one card. `ROCR_VISIBLE_DEVICES` alone is **unreliable** for this
on mixed hosts.

Workarounds (pick one):

1. **Unbind** the display/legacy card from `amdgpu` after boot (keep it for POST / recovery, but out of
   the compute topology).
2. Use a **non-amdgpu** display GPU (old NVIDIA / Intel / Terascale HD that never binds `amdgpu`) or
   run the host **headless**.
3. Hide devices carefully with `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` and re-check `rocminfo`
   still enumerates every V620.

Docker device passthrough of only the V620 does **not** always isolate an unsupported host GPU from
HSA topology init — VM passthrough or unbind is the reliable fix when mixed AMD cards fight ROCm.

## Overheating / thermal paste on V620 {#v620-thermals}

`#general`: V620s are passive server cards — airflow matters more than most desktop GPUs.

| Report | Notes |
|---|---|
| Stock pads | Some hosts call factory **graphene / graphite** pads high quality; a bad repaste can **worsen** temps |
| Batch heating | Community reports of cards that overheat or fail to init under ROCm after long load — test new cards under load for a long stretch; OS may still list an AMD GPU while `rocm-smi` does not |
| Temps | Rough community ballpark: ~**86 °C** at full load can be “OK”; **94–96 °C** is high (often stress / curve tuning) — improve airflow / power cap first |
| Cooling | Cap at **180 W** (or lower); single-fan / weak blower setups struggle at stock **250 W** |
| Unducted case 120 mm | `#general` (Sep 13 2026): loud **5000 rpm** 120 mm fans and **no ducts** still hit ~**95 °C** at a **120 W** cap. A **blower / ducted** shroud on the card immediately dropped temps enough to hold **180 W**. |
| Ducted 120 mm (push) | Community dual-V620: Arctic **P12 Pro** through a duct, ~**62–70 °C** at **175 W** depending on ambient. Needs **static pressure**; air bypasses the heatsink if you leave a gap. `#general` (Sep 27): public [120 mm → fanless GPU ducts](https://makerworld.com/en/models/3184181-custom-fan-ducts-120mm-fans-to-fanless-gpus). |
| 80 mm high-static (4-card) | `#other` (Sep 28): four **80 mm ~7k** fans at **50–60%** PWM held ~**60 °C** on **4× 180 W** vLLM (community). Two fans may be enough on that chassis. |
| 40 mm axial on the card | Works but is a **high-pitched screamer**. Prefer a **blower-style** mount (example: [Thingiverse 7153218](https://www.thingiverse.com/thing:7153218)) for compactness vs noise. |
| Unused on-card fan pads | `#general` (Oct 7): the V620 PCB has an unused **4-pad** fan footprint on the back. Community: **do not** wire it as a PWM header — current firmware would run the fan at **100%** with no curve, and one teardown could not use it as PWM. Prefer an **external** blower / hub (rows above). |
| 9733 blower (4U last slot) | `#general` (Oct 4–6 2026): [Thingiverse 7296707](https://www.thingiverse.com/thing:7296707) mounts a **9733** blower on V620 with intake toward the CPU so the last 4U slot does not suck the case wall. Fan rating cited **~2.94 A / 12 V**. **Molex-only** power runs the fan at **100%** (no curve). Community: **loud at 100%** when a cheap PWM box **pulses full/off** instead of true PWM. One report that **one Molex** can feed **four** — **Needs verify**. |
| EFH-08E12W + PWM hub | `#general` (Oct 4): quieter idle path than a 9733-at-100% is **EFH-08E12W-JP01** on [Thingiverse 7153218](https://www.thingiverse.com/thing:7153218) plus a real PWM hub (example: Arctic **ACFAN00351A**) and a software curve ([CoolerControl](https://docs.coolercontrol.org/)). |
| SUNON 9733 + SATA PWM hub | `#general` (Oct 6): community quieter swap from the stock-loud **2.94 A** 9733 is **SUNON PF97331B1-B00U-S99** (12 V, **~19.2 W**, 4-wire 97 mm). Power / PWM through a SATA-fed hub such as [SilverStone CPF04](https://www.silverstonetek.com/en/product/info/accessories/cpf04/) (one header → up to eight fans) plus a **BIOS or software fan curve**. Molex on the old 9733 stays 100%. |

PCB photos for cooler compatibility: community link
[Linus Tech Tips V620 thread](https://linustechtips.com/topic/1588645-radeon-pro-v620-but-tencent-edition-meeting-issues-with-everything/).
W6800 waterblocks are **not** a guaranteed fit.

## IOMMU faults then a PSU group of cards disappears {#iommu-psu-group-drop}

`#general` (31 Oct): on an **8× V620** host, a **TP8** request or sometimes
**model load** logged **IOMMU page faults**, then **fatal PCIe** errors, then
**six** cards vanished from the OS. Those six shared **auxiliary power from a
1500 W server PSU through a breakout board**. The two cards powered entirely
from a **workstation PSU** stayed. The first GPU to report the IOMMU fault was
**not** always the same card, and was **not** necessarily the problem card.

Same thread later (31 Oct): after **adding GPUs** into a **passthrough VM**,
**ACS Request Redirect** and **Relaxed Ordering** flipped **on**. Restoring
those controls stopped the IOMMU faults and the disappearing-card group.
**PP2/TP4** then ran; **TP8** still failed after the ACS fix. Mixed PSU /
breakout remains a risk, not a wiki default. **Needs verify** on a second host.

`#general` (2 Nov): same class of **8× V620** host, **vLLM TP8** still logged
**IOMMU faults** and the cards vanished — during **load**, mid-**inference**,
or after decode had already started. **llama.cpp TP8** on that host did **not**
drop cards and ran at **~99%** of the power cap. **PP2/TP4** in vLLM still
worked. That A/B makes a simple PSU trip less likely for the remaining TP8
fail (llama.cpp was already at full draw). Community guess: a vLLM / ROCm
kernel writing **outside its DMA buffer**. Unresolved — **Needs verify.** Prefer
**PP2/TP4** (or llama.cpp TP8) until a second host reproduces.

A/B before blaming the fork or ripping every cable:

1. Re-read ACS **inside the guest** (and on the host) after you add or
   passthrough cards — [ACS bits](../tuning/p2p.md#acs-bits). Request Redirect
   (bit 2) is the principal P2P obstacle; community also saw Relaxed Ordering
   flip (Device Control, not an ACS bit).
2. Confirm **dual 8-pin** on every card in the disappearing group — [power](../tuning/power.md#slot-power-and-psu-transients).
3. Read `dmesg` for IOMMU / AER / fatal PCIe around the drop.
4. Cap TDP and retry TP8 vs a smaller world size (PP2/TP4 was the working split here).
5. A/B **llama.cpp TP8** vs **vLLM TP8** on the same host. If only vLLM drops
   cards at full draw, treat it as a **vLLM / DMA** lead, not “buy a bigger PSU.”
6. Do **not** treat a mixed PSU / breakout layout as a wiki default.
7. On chassis **breakout boards**, balance cards across **strips** (one
   overloaded rail can look like a PSU trip). `#general` (10–11 Nov): an
   Octominer-class load-share still wobbled on **750 W** units; **1200 W HP
   server PSUs** then held even at **150 W** — [power](../tuning/power.md#slot-power-and-psu-transients).

`#general` (4–5 Nov): a **second** 8× class of host (dual **PLX88096**, community
**Tyan S8030**, **~128 GB** DRAM, multi server-PSU / breakout) still could not
keep **vLLM TP8** or **TP=4,PP=2** up while hunting the same drop:

- **MTP=2** vs **MTP=0:** MTP=2 triggered an **instant reset** in the **521–933 W**
  GPU-power band while MTP=0 **sustained ~951 W** across several ladders —
  treat as **PSU OCP / di/dt transient**, not “the average wattage is fine.”
  Caps do not always catch spikes — [power](../tuning/power.md#slot-power-and-psu-transients).
- **120 V** multi-PSU: community used **add2PSU** / a chassis **breakout** and
  preferred **same ground / same wall circuit** where possible. Reseat the
  breakout if TP8 only fails after you move cables.
- **Cross-switch hop:** `NCCL_P2P_LEVEL=pix` is **intra-switch** P2P. Dual-PLX
  **TP8** or **PP across the two switches** still forced host-staged traffic
  (community: KFD **userptr restore** storm). Prefer TP **inside** one switch —
  [PLX](../tuning/p2p.md#plx--pcie-switches).
- **Vision on** crashed TP8; **`--language-model-only`** then stayed up —
  [vision + TP8](./vllm-flash-next.md#flash-next-vision-tp8).
- **`PLE=int4` in RAM** on ~128 GB blocked TP8 / PP2; **`PLE=int4-disk`** was
  the fit — [PLE disk](./vllm-flash-next.md#flash-next-8x-ple-disk).

`#vllm-rdna` (6–7 Nov) on the same dual-PLX class:

- **Per-switch PSU:** two **Seasonic PX1600** — motherboard on one; each
  switch + its GPUs on its own PSU; **175 W** cap (250 W had been OK with
  one PSU and **no** switch) — [power](../tuning/power.md#slot-power-and-psu-transients).
- **BMC often silent:** a hang can look like an **OS freeze** with **no**
  PCIe SERR / crash log on the BMC. Use host `dmesg` / AER first.
- **TP=8 vs TP=4,PP=2:** community still expected **TP8 decode** to win and
  saw **better prefill** on **TP=4,PP=2**. Neither is a recipe; both still
  crash some hosts — [Flash-Next 8×](./vllm-flash-next.md#flash-next-8x-tp-vs-pp).
- **Inter-switch counters:** one host reading **PEX88096** registers did
  **not** see the uplink saturate (single-digit GB/s in steady inference;
  bursts mainly at startup with LMCache). Community guess is **latency**,
  not bandwidth — [PLX](../tuning/p2p.md#plx--pcie-switches).

`#vllm-rdna` / `#general` (7–8 Nov) on that class of host:

- **Kernel 7.3.1 → 7.3.2:** one **7.x-class** host (distro not named; wiki
  already has Ubuntu 26 / Fedora **7.x** kernels) said the bump stopped
  **load / prefill** crashes; rerunning the old kernel crashed again. Remaining
  **decode** crashes were **prompt-length dependent**. **Needs verify** — do
  not treat “install 7.3.2” as a recipe without knowing your distro package.
- **120 W / ~1900 MHz:** two hosts then kept **TP=8** and **TP=4,PP=2** up at
  that cap. One had last crash logs at **~950–970 W** wall; lowering clocks
  barely moved tokens. Treat leftover 8× drops as **PSU transients** after ACS
  / PLE-disk / vision-off — [power](../tuning/power.md#slot-power-and-psu-transients).
- One host then ran **vision-on** after the kernel bump (still under extra
  logging). Keep A/B vision **last** — [vision + TP8](./vllm-flash-next.md#flash-next-vision-tp8).

Still **Needs verify**. Not an 8× Flash-Next recipe.
