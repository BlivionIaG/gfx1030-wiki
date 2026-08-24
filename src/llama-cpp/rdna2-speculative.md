# RDNA2 fork — Speculative decoding

> **WIP:** Community-tested configs. See [Verification status](../../reference/verification.md#llama-cpp-rdna2).

## DFlash / MTP basics

```bash
  --spec-type draft-dflash \
  --spec-draft-model /path/to/dflash.gguf \
  --spec-draft-n-max 6
```

`--spec-draft-n-max` is a workload knob; start at the draft model's block size and tune.

## DFlash2 + ngram (community-tested)

DFlash2 shines on **real workloads** (coding, agents, long context) more than short synthetic benches —
community members saw DFlash2 alone score lower than MTP on a 20k bench (~30 vs ~40 t/s) but win on actual
agent/coding sessions (~47 t/s with 700+ t/s prefill).

**Draft model quant:** use **Q4_K_M** for the DFlash2 draft GGUF — do **not** use Q8_0 for the drafter
(same acceptance, slower).

**Recommended draft GGUFs** (Aug 2026):

- [`incoai/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
- [`webhie/Qwen3.8-27B-Q4-AutoRound-Code-GGUF`](https://huggingface.co/webhie/Qwen3.8-27B-Q4-AutoRound-Code-GGUF)

**Coding / ngram-map-k4v**:

```bash
--spec-type draft-dflash,ngram-map-k4v \
  --spec-draft-n-max 5 \
  --spec-ngram-map-k4v-size-n 12 \
  --spec-ngram-map-k4v-size-m 48
```

**General / ngram-mod**:

```bash
--spec-type draft-dflash,ngram-mod \
  --spec-draft-n-max 5 \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64
```

Use **tg1024+** (not tg32) for realistic decode benchmarks. DFlash2 is more consistent at long context;
MTP acceptance tends to fall off.

### Full DFlash2 serve example (TP4, Qwen3.8-27B)

```bash
HSA_NO_SCRATCH_RECLAIM=1 GGML_HIP_RDNA2_AUTO=1 GGML_HIP_SAFE_STATE_IO=1 \
GGML_TP_SHARDED_OUTPUT=1 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
./build/bin/llama-server \
  -m ./models/qwen38-27b-q4s8/autoround/Qwen3.8-27B-Q4_0.gguf \
  -ngl all --split-mode tensor --tensor-split 1,1,1,1 \
  --device ROCm0,ROCm1,ROCm2,ROCm3 --flash-attn on \
  --ctx-size 262144 --batch-size 8192 --ubatch-size 4096 \
  --host 0.0.0.0 --port 8080 --metrics \
  --reasoning-effort xhigh --reasoning-preserve \
  --spec-type draft-dflash,ngram-map-k4v \
  --spec-ngram-map-k4v-size-n 12 --spec-ngram-map-k4v-size-m 48 \
  --spec-draft-n-max 5 \
  -md ./models/qwen38-27b-q4s8/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --device-draft ROCm0 --parallel 1 --spec-draft-ubatch-size 4096 \
  --cache-ram 65535
```

## MTP on tensor-split setups

Pin the draft to one GPU while the main model stays tensor-split:

```bash
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-ngl 999 --spec-draft-device ROCm0
```

**MXFP4** quants (e.g. `quark75/Qwen3.8-27B-MXFP4-GGUF`) pair well with MTP on TP2 — see
[Serving](../rdna2-serving.md#docker-example).
