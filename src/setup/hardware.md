# Supported Hardware

`gfx1030` is the LLVM target name for AMD's **Navi 21** GPU (RDNA2 generation, codename
*Sienna Cichlid*). Several retail and workstation cards use this die and therefore report themselves
as `gfx1030`.

## Cards that natively report as `gfx1030`

| Card | Architecture | VRAM | Compute Units | Notes |
| --- | --- | --- | --- | --- |
| Radeon RX 6950 XT | RDNA2 / Navi 21 | 16 GB GDDR6 | 80 | Highest-clocked Navi 21 |
| Radeon RX 6900 XT | RDNA2 / Navi 21 | 16 GB GDDR6 | 80 | |
| Radeon RX 6800 XT | RDNA2 / Navi 21 | 16 GB GDDR6 | 72 | |
| Radeon RX 6800 | RDNA2 / Navi 21 | 16 GB GDDR6 | 60 | Best price/VRAM for LLMs |
| Radeon PRO W6800 | RDNA2 / Navi 21 | 32 GB GDDR6 | 60 | Workstation, 32 GB is great for larger models |
| Radeon PRO V620 | RDNA2 / Navi 21 | 32 GB GDDR6 | 72 | Data-center / cloud card |

All of these are on the **officially supported** list for recent ROCm releases on Linux.

## RDNA2 relatives that can run gfx1030 code

The rest of the RDNA2 line uses a different LLVM target but shares the same ISA family. They are **not**
on the official support matrix, but in practice they run gfx1030 kernels once you set
`HSA_OVERRIDE_GFX_VERSION=10.3.0` (see [HSA_OVERRIDE for RDNA2 Cousins](./hsa-override.md)).

| Card | LLVM target | Die |
| --- | --- | --- |
| RX 6750 XT / 6700 XT / 6700 | `gfx1031` | Navi 22 |
| RX 6650 XT / 6600 XT / 6600 | `gfx1032` | Navi 23 |
| RX 6500 XT / 6400 | `gfx1034` | Navi 24 |
| Ryzen 6000/7000 iGPU (RDNA2) | `gfx1035` / `gfx1036` | Rembrandt / Phoenix |

## Architecture highlights (Navi 21)

- **RDNA2** compute units with a native **wavefront size of 32** (wave32), unlike GCN's wave64.
- **Infinity Cache** (128 MB on Navi 21) that dramatically raises effective memory bandwidth.
- **No dedicated matrix/tensor cores** — RDNA2 predates the WMMA/matrix instructions added in RDNA3
  (`gfx11xx`). Matrix math runs on the regular vector ALUs, so expect lower peak throughput than
  RDNA3 or CDNA cards, but very good price/performance for inference.
- **FP16** is well supported; **BF16** has limited/emulated support and is best avoided for hot paths
  (prefer `float16`).

## How to check your GPU target

```sh
rocminfo | grep -i 'gfx\|Name'
# or, more directly:
rocminfo | grep -m1 -o 'gfx[0-9]*'
```

If the output shows `gfx1030`, everything in this wiki applies directly. If it shows `gfx1031`,
`gfx1032`, etc., head to [HSA_OVERRIDE for RDNA2 Cousins](./hsa-override.md) first.
