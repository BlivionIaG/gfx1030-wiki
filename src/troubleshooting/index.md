# Troubleshooting

Common errors on gfx1030 / RDNA2 and how to fix them. Pick the section that matches your stack:

| Stack | Page |
|---|---|
| ROCm install, hipBLASLt, BF16, iGPU, Secure Boot, CPU governor | [General](./general.md) |
| vLLM Docker, CUDA graphs, kernel dispatch, AMDSMI, RCCL 7.2.1+, MTP stalls | [vLLM](./vllm.md) |
| llama.cpp RCCL, KV checkpoints, FA occupancy abort, DAX mmap, tensor split, PSU, Vulkan ICD | [llama.cpp](./llama-cpp.md) |

> **WIP:** Fixes involving latest `-extras` images assume a current image pull — see
> [Verification status](../reference/verification.md).

## Still stuck?

- [ROCm troubleshooting docs](https://rocm.docs.amd.com/)
- [Community resources](../meta/resources.md)
- [Contributing](../meta/contributing.md) — add your fix so others don't rediscover it
