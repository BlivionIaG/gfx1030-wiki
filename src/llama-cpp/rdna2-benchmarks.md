# RDNA2 fork — Benchmarks

> **WIP:** All numbers below are **author- or Discord-reported**. Run matched before/after on your
> hardware. See [Verification status](../../reference/verification.md#llama-cpp-rdna2-benchmarksmd).

Point-in-time numbers using [`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2),
ROCm/RCCL, Flash Attention on, **F16 KV**, batch/ubatch **2048/256**, with the env prefix
`GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0
HSA_NO_SCRATCH_RECLAIM=1` (ROCm 7.14 libs on `LD_LIBRARY_PATH`).

## 4× V620 (24 CPU threads)

| Model | Quant | Split | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|
| Qwen3.6-27B | F16 | tensor | 850.54 | 18.52 |
| Qwen3.6-27B (DavidAU/Fable) | Q4_K_M | tensor | 1,017.25 | 27.76 |
| Qwen3.5-122B-A10B | Q4_K_M | tensor | 1,064.40 | 39.12 |
| Qwen3.6-35B-A3B | Q4_K_M | layer | 2,293.36 | 72.17 |
| Qwen3.6-35B-A3B | Q4_K_M | tensor | 1,939.80 | 58.30 |

## 2-GPU tensor split (TP2)

| Model | Quant | Split | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|
| Qwen3.6-27B (DavidAU/Fable) | Q4_K_M | tensor | 707.14 | 25.63 |
| Qwen3.6-27B | F16 | tensor | 448.02 | 13.15 |

## Latest fork update (Aug 2026, `#llamacpp`)

MTP-4 TP4 on a pelican prompt:

| Quant | Average t/s | Peak t/s |
|---|---:|---:|
| Q4_0 + MTP-4 | 85 | 115 |
| Q8_0 + MTP-4@Q4_0 | 75 | 105 |

DFlash2 on TP2: ~52 t/s coding at 0–31k context, ~40 t/s at 64k. Checkpoint fix in
[PR #12](https://github.com/edwinbrowwn/llama.cpp-rdna2/pull/12) (in review).

> `pp512` = prefill (512-token prompt); `tg128` = generation (128 tokens). For A3B MoE, **layer split**
> beats tensor split; dense models use tensor split.

## Upcoming optimizations (PR #10)

> **Status:** in review — [PR #10](https://github.com/edwinbrowwn/llama.cpp-rdna2/pull/10). Env vars
> below will be consolidated post-merge.

Reported improvements: 15.9–61.2% lower MMVQ latency, up to 17.7% faster GDN prefill, +4.39% tgen from
DeltaNet sibling fusion. MTP works up to 6 with minimal hit on low acceptance.

### Peak throughput (4× V620 @ 140 W)

| Model | Quant | tg (t/s) | pp (t/s) |
|---|---|---:|---:|
| Qwen3.5-122B-A10B | Q4 | 100+ | ~1300 |
| Qwen3.6-35B-A3B | Q4 | 180+ | ~3800 |
| Qwen3.6-27B | Q8 | 75 | ~880 |

See [Power Tuning](../../tuning/power.md) for the 140 W cap. Long-context PP HIP crash fix tracks
[ROCm/rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817).

### Environment used for PR #10 results

```bash
GGML_HIP_SAFE_STATE_IO=1 GGML_HIP_GFX1030_Q8_CACHE=1 GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1 \
GGML_HIP_GFX1030_Q8_1_FUSION=1 GGML_HIP_GFX1030_NATIVE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PXB \
GGML_TP_SHARDED_OUTPUT=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 GGML_CUDA_P2P=1 GGML_HIP_GRAPHS=1
```
