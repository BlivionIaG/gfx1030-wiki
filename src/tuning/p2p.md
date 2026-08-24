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
> recipes here remain Fedora-validated for now.

## Two validated kernel paths

| Path | Kernel | Validated on | Notes |
|---|---|---|---|
| **Pre-7** | ≤ 6.19.x | 6.17.6 | |
| **Post-7** | ≥ 7.1 | 7.1.7 | |

Both kernels can be installed on the same host simultaneously; toggle the default with
`grubby --set-default /boot/vmlinuz-<evr>`. Both boot with the **180 W** power cap from
[Power Tuning](./power.md).

## Key prerequisites

- **AMD CPU** — the KFD P2P path is validated on AMD (EPYC 7452).
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

## When P2P is enabled but inference is slower

P2P can be **slower than non-P2P** on bad topologies — gen3 x4 links, southbridge-routed slots, or
ACS-blocked CPU root ports. `#llamacpp` reports include cards training at PCIe gen3 and tensor split
running faster with P2P disabled.

If bandwidth tests pass but inference regresses:

```bash
export NCCL_P2P_DISABLE=1          # llama.cpp / RCCL tensor parallel
```

On the RDNA2 fork, the README also documents a flag to disable P2P all-reduce fusion
(`GGML_HIP_GFX1030_P2P_ALLREDUCE=off`). Always A/B test — verified ~25 GB/s P2P between V620 pairs does
not guarantee faster token generation if links are narrow.

## Host topology

Community reports, not wiki-benched:

| Topology | What people report |
|---|---|
| **PCIe 4.0 x16** per card (CPU root ports) | Best case for TP4. Known-good llama.cpp TP4: Gigabyte **MC62-G40**. |
| **PCIe 4.0 x8** per card | Practical floor for 8× V620 without a switch; expected to still scale. |
| **PCIe 3.0 x4** | Throughput often **stops scaling at 3 cards** and can regress at 4. |
| **PLX / PCIe switch** | Intra-switch P2P can stay full-width (e.g. 4× Gen4 x16 per switch). Host↔switch link is the bottleneck. |
| **Dual-socket (NUMA)** | TP across sockets can **halve prefill**. Bind workers to the NUMA node of their GPUs. P2P is typically **per socket**. vLLM with NUMA-aware TP workers is less painful than llama.cpp crossing UPI/Infinity Fabric. |
| **Odd GPU counts (TP3)** | llama.cpp tensor-split on **3** cards has caused driver crashes; prefer 2 or 4. |
