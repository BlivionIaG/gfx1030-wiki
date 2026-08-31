# Multi-GPU PCIe P2P

For multi-card V620 rigs, enabling **GPU↔GPU PCIe Peer-to-Peer (P2P)** lets the GPUs DMA directly to each
other's VRAM instead of bouncing through system memory — important for tensor/pipeline parallel serving.
This page summarizes the
[`pcie_p2p/`](https://github.com/blivioniag/v620_toolbox/tree/master/pcie_p2p) feature of the
[`v620_toolbox`](https://github.com/blivioniag/v620_toolbox) repo.

> Validated end-to-end on **Fedora 43 + AMD CPU** (EPYC 7452), **4× Radeon PRO V620**. Result:
> `rocminfo` enumerates 5 HSA agents and `amd-smi topology` shows **12/12 GPU↔GPU P2P ENABLED**.
>
> Power-cap tuning on **Ubuntu 26.04** is supported separately via
> [`ubuntu_powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/ubuntu_powertuning) — P2P
> recipes here remain Fedora-validated for now. Community Intel notes (Ice Lake) are in
> [When P2P helps — and when it does not](#when-p2p-helps--and-when-it-does-not).

## Two validated kernel paths

| Path | Kernel | Validated on | Notes |
|---|---|---|---|
| **Pre-7** | ≤ 6.19.x | 6.17.6 | |
| **Post-7** | ≥ 7.1 | 7.1.7 | |

Both kernels can be installed on the same host simultaneously; toggle the default with
`grubby --set-default /boot/vmlinuz-<evr>`. Both boot with the **180 W** power cap from
[Power Tuning](./power.md).

## Key prerequisites

- **AMD CPU** — the KFD P2P path is **wiki-validated** on AMD (EPYC 7452). Intel Ice Lake can
  enumerate P2P (community) but often does **not** speed up inference — see below.
- **≥ 2 Radeon PRO V620** for a meaningful P2P topology (bench validated on 4).
- **No pre-gfx1030 AMD GPU installed alongside the V620.** ROCm 7.x dropped support for older ASIC
  families (gfx8xx Polaris, gfx900 Vega, …). If any pre-gfx1030 AMD GPU is present at boot, KFD
  registration fails and `rocminfo` bails with `Failed to map remapped mmio page on gpu_mem 0`. This
  only matters if your host actually has older AMD cards — remove them first.

Identify your V620 reference boards:

```bash
lspci -nn | grep '1002:73a1'    # one line per V620, subsystem 1002:0e34
```

The full 4-tuple `1002:73a1:1002:0e34` uniquely identifies the V620 reference board and is what the
toolbox scripts match on. Other gfx1030 boards (RX 6900 XT / 6800) use device `0x73bf` and different
subsystem IDs — they'd need the identity match adjusted.

## Readiness & verification

The toolbox ships diagnostics that gate on kernel config, hardware, ACS/IOMMU, and runtime state:

```bash
# Four-gate readiness check (kernel + hardware + ACS/IOMMU + runtime)
sudo ./powertuning/scripts/v620-p2p-readiness.sh

# End-to-end P2P verification
sudo ./pcie_p2p/scripts/verify-p2p.sh
```

A healthy system shows:

```bash
rocminfo | grep -c '^  Name:.*gfx1030'     # one per V620
amd-smi topology                            # all GPU<->GPU pairs: P2P ENABLED
```

For the full recipe (kernel config deltas, BIOS/IOMMU settings, what can go wrong), read
[`pcie_p2p/README.md`](https://github.com/blivioniag/v620_toolbox/blob/master/pcie_p2p/README.md) and the
knowledge base
[`powertuning/docs/AMD_P2P.md`](https://github.com/blivioniag/v620_toolbox/blob/master/powertuning/docs/AMD_P2P.md).

## When P2P helps — and when it does not

`amd-smi topology` saying **P2P ENABLED** is not the same as faster tokens. Always A/B with
`NCCL_P2P_DISABLE=1`. Community reports (`#general`, Ice Lake 4× V620 host):

| Host | What people report |
|---|---|
| **EPYC (multi-CCD / multi-die)** | P2P is the case that usually **wins** — cards skip Infinity Fabric / cross-die hops. This is the wiki-validated toolbox path. |
| **Intel Ice Lake (monolithic die)** | P2P can enumerate (`io=1 p2p=3` on all four V620s) and still do **nothing** — or **regress ~4%** — on both llama.cpp (Qwen3.8-27B Q8) and vLLM. Cards already share one CPU PCIe root; GPU↔GPU DMA is not cheaper than going through the CPU. |
| **Desktop Ryzen (e.g. 3950X)** | P2P **tanked** when cards trained at gen3 x4. Turn it back off. |
| **Chipset / southbridge slot** | P2P is **worse** if one card is on the chipset rather than CPU root ports. |
| **vLLM on RDNA2** | Kernel P2P alone is **not enough**. Fork author: vLLM still needs an RDNA-side patch (these cards are not CDNA). Treat "P2P works in `amd-smi`" as a prerequisite, not a finished vLLM speedup. |

If bandwidth tests pass but inference regresses:

```bash
export NCCL_P2P_DISABLE=1          # llama.cpp / RCCL tensor parallel
```

On the RDNA2 fork, the README also documents a flag to disable P2P all-reduce fusion
(`GGML_HIP_GFX1030_P2P_ALLREDUCE=off`). Verified ~25 GB/s P2P between V620 pairs does not guarantee
faster token generation if links are narrow or the host is a single monolithic PCIe root.

### Intel: do not disable IOMMU to "help" P2P

On Ice Lake, **disabling VT-d / IOMMU broke P2P** even though some generic docs say IOMMU-off is more
permissive. Leave IOMMU **on**, then run the toolbox readiness script and ACS checks. Fedora Server
**44** + kernel **6.19** + the powerfix has been reported working on 4× V620 (community; not the
Fedora 43 AMD validation).

### PLX / PCIe switches

- Intra-switch P2P can stay full-width (e.g. 4× Gen4 x16 behind one PLX 88096). The **host↔switch
  uplink** (typically one x16) is the bottleneck.
- **Without tensor parallel**, a PLX box is usually **slower** than native CPU lanes: higher
  latency and less aggregate host bandwidth.
- **Do not tensor-split across a daisy-chained pair of switches** — that single inter-switch link
  is a TP bottleneck. Prefer TP **inside** one switch and pipeline-parallel **between** switches
  (vLLM can do TP+PP that way; llama.cpp generally cannot).
- Community: ACS often needs extra kernel cmdline fiddling; **`pcie surprise link down`** crashes
  were fixed by putting a small fan on the PLX heatsink (these boards often ship with no airflow
  notes).

## Host topology

Community reports, not wiki-benched:

| Topology | What people report |
|---|---|
| **PCIe 4.0 x16** per card (CPU root ports) | Best case for TP4. Community known-good llama.cpp TP4 board: Gigabyte **MC62-G40**. |
| **PCIe 4.0 x8** per card | Practical floor for 8× V620 without a switch; expected to still scale. |
| **PCIe 3.0 x4** | Throughput often **stops scaling at 3 cards** and can regress at 4. |
| **PLX / PCIe switch** | See [PLX / PCIe switches](#plx--pcie-switches). |
| **Dual-socket (NUMA)** | TP across sockets can **halve prefill**. Bind workers to the NUMA node of their GPUs. P2P is typically **per socket**. vLLM with NUMA-aware TP workers is less painful than llama.cpp crossing UPI/Infinity Fabric. |
| **Odd GPU counts (TP3)** | llama.cpp tensor-split on **3** cards has caused driver crashes; prefer 2 or 4. |

## Cabling / risers (community)

Not wiki-benched — common `#general` notes:

- **SlimSAS / SFF-8654** cables: **PCIe gen3** is usually fine at ~70 cm; **gen4** needs testing per
  cable/insulation. Prefer known-good gen4 kits over the cheapest Amazon/eBay passive ribbon risers.
- **Passive PCIe risers:** several community reports of timeouts / inability to hold **gen4 x16** with
  cheap passive risers under multi-GPU load, while the same slots work with cards seated directly.
  Brand-name gen4 risers (e.g. ADT-class) are often the next step — A/B one card first.
- **V620 + blower shroud length:** community measure ~**37 cm** with a common EFH-08E12W-style fan
  shroud installed — plan chassis / PLX slot spacing accordingly.
