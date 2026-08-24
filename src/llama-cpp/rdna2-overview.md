# RDNA2 fork — Overview

> **WIP:** See [Verification status](../../reference/verification.md#llama-cpp-rdna2).

[`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2) is a community fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) with RDNA2-specific optimization work, developed on
**ROCm 7.14 / Ubuntu Server 26** and validated primarily on **four Radeon PRO V620 (`gfx1030`)** GPUs.
The focus is **tensor parallel (TP)** and **MMQ/MMVQ** (quantized matmul) kernels.

> Actively evolving and experimental. Treat throughput claims as needing your own matched before/after
> runs. See [Benchmarks](../rdna2-benchmarks.md).

## What it does

A **native RDNA2 profile activates automatically at runtime** — unsupported models, shapes, quants, and
topologies fall back to stock llama.cpp. Headline areas:

- **MMQ / MMVQ:** RDNA2 expert-width MMQ, Q4_0 DOT8 MMVQ, MXFP4/NVFP4 native arithmetic, MTP/DFlash paths.
- **FlashAttention:** native tiled RDNA2 arithmetic/reductions.
- **Tensor parallel:** RCCL tuner + P2P all-reduce schedules, embedding-sharded LM head, TP4 P2P fusion.
- **Graph fusion:** ADD/RMSNorm, Q8_1 reuse, SwiGLU→Q8_1, GDN sibling projection (Qwen3.5/3.6 MoE).

### Optimization highlights (author-reported)

- **RCCL tensor-parallel all-reduce** — `GGML_CUDA_ALLREDUCE=nccl`; reported **+10% tgen / +20% prefill**
  on Qwen 122B.
- **DFlash2** speculative decoding with ngram helpers — see [Speculative decoding](../rdna2-speculative.md).
- **RCCL autotuner**, parallel multi-GPU weight uploads, AMD checkpoint backports for Qwen.

See the fork's `README.md` and `docs/gfx1030-*` / `docs/rdna2-*` for the authoritative list.

## Requirements

- Linux, CMake, ROCm with HIP clang and **RCCL**.
- Validated path: four V620 / `gfx1030` with tensor splitting.
- Compatible main GGUF; optional DFlash/MTP draft GGUF.

## Build

```bash
git clone https://github.com/edwinbrowwn/llama.cpp-rdna2.git
cd llama.cpp-rdna2
./scripts/build-rdna2-portable.sh
```

Override discovery when needed:

```bash
ROCM_PATH=/path/to/rocm TARGET_ARCH=gfx1030 BUILD_DIR=build ./scripts/build-rdna2-portable.sh
```

Maintainer helper for gfx1030 / ROCm 7.14: `scripts/build-rdna2-rocm.sh` (defaults: ROCm
`/opt/rocm/core-7.14`, target `gfx1030`).

## Next steps

- [Benchmarks](../rdna2-benchmarks.md) — author-reported numbers and PR #10 status.
- [Serving](../rdna2-serving.md) — launch commands and Docker.
- [Speculative decoding](../rdna2-speculative.md) — DFlash2, MTP, ngram configs.
