# llama.cpp on gfx1030 — Overview

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a fast, low-dependency way to run GGUF LLMs on
gfx1030. This section covers stock builds and the community RDNA2-optimized fork.

## Where to start

| Goal | Page |
|---|---|
| Build stock llama.cpp (ROCm or Vulkan) | [Building & running](../building.md) |
| Multi-GPU tensor parallel, MMQ tuning, DFlash2 | [RDNA2 fork overview](../rdna2-overview.md) |
| Fork benchmarks and PR status | [RDNA2 benchmarks](../rdna2-benchmarks.md) |
| DFlash2, MTP, ngram speculative decoding | [RDNA2 speculative decoding](../rdna2-speculative.md) |
| Launch commands, Docker, limits | [RDNA2 serving](../rdna2-serving.md) |
| Something broke | [llama.cpp troubleshooting](../../troubleshooting/llama-cpp.md) |

## Which path?

- **Stock llama.cpp** — good starting point, single-GPU. Vulkan without ROCm is possible but
  `#llamacpp` prefers HIP for `--split-mode tensor` (RADV has crashed V620 hosts).
- **`edwinbrowwn/llama.cpp-rdna2`** — multi-GPU V620 rigs, tensor parallel, RCCL all-reduce, DFlash2.
  Most `#llamacpp` performance work happens here. A matched A/B on Qwen3.8-27B Q6_K + MTP reported
  **+57%** vs stock with identical outputs — see [Benchmarks](../rdna2-benchmarks.md#stock-llamacpp-vs-this-fork-community-ab).

Multi-GPU tensor parallel benefits greatly from [PCIe P2P](../../tuning/p2p.md).
