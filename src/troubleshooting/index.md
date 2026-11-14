# Troubleshooting

Common errors on gfx1030 / RDNA2 and how to fix them. Pick the section that matches your stack:

| Stack | Page |
|---|---|
| ROCm install, hipBLASLt, BF16, iGPU, Secure Boot, CPU governor, V620 cooling, multi-PSU / IOMMU card drop (vLLM TP8 vs llama.cpp TP8) | [General](./general.md) |
| Host never POSTs with 4× V620 / PLX (Dell Precision, H12D, Z4 vs Z6 G4, R730 vs SYS-2028GR) | [Host firmware](../setup/host-firmware.md) |
| vLLM Docker, CUDA graphs (incl. 0.33 nightly PLE NaN / `!!!!`), TunableOp / rocBLAS mismatch, Inductor cache, GLM vision compile, AMDSMI, RCCL, rocprofiler idle CPU, `rdna_ar` / non-mono hangs, KV-offload loops | [vLLM](./vllm.md) |
| Flash-Next stalls, PP3 / FULL graphs, hybrid KV, PLE (8× RAM vs `int4-disk`), QSA vs FA-RDNA2, AutoRound `INCConfig`, vision OOM / 8× TP8 crash, 8× TP8 vs TP=4,PP=2, wvSplitK, dynamic MTP / KV-restore loops | [vLLM Flash-Next](./vllm-flash-next.md) |
| llama.cpp RCCL, KV checkpoints, FA occupancy abort, DAX mmap, tensor split, PSU, Vulkan ICD, Windows Unsloth / WSL | [llama.cpp](./llama-cpp.md) |

> **WIP:** Fixes involving latest `-extras` images assume a current image pull — see
> [Verification status](../reference/verification.md).

## Still stuck?

- [ROCm troubleshooting docs](https://rocm.docs.amd.com/)
- [Community resources](../meta/resources.md)
- [Contributing](../meta/contributing.md) — add your fix so others don't rediscover it
