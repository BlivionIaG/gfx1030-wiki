# RDNA2-optimized fork (llama.cpp-rdna2)

> **WIP:** Benchmarks and DFlash2 recipes are **author- or Discord-reported**. The page already warns
> to run matched before/after tests — see also [Verification status](./verification.md#llama_cpp_rdna2_forkmd).

[`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2) is a community fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) with weeks of RDNA2-specific optimization work,
developed on **ROCm 7.14 / Ubuntu Server 26** and validated primarily on **four Radeon PRO V620
(`gfx1030`)** GPUs. The focus is **tensor parallel (TP)** and **MMQ/MMVQ** (quantized matmul) kernels.

> This is an actively evolving, experimental fork whose author is seeking feedback. It's aimed at
> multi-GPU V620/gfx1030 rigs; on other systems it keeps conservative stock fallbacks. Treat throughput
> claims as needing your own matched before/after runs.

## What it does

A **native RDNA2 profile activates automatically at runtime** — you don't need to copy a long list of
feature flags. Unsupported models, shapes, quantizations, and topologies transparently fall back to
normal llama.cpp behavior. Headline areas:

- **MMQ / MMVQ:** RDNA2 expert-width MMQ selection, Q4_0 DOT8 MMVQ, conservative Q4_K/Q6_K MMVQ dispatch,
  MXFP4/NVFP4 native arithmetic, and validated MTP/DFlash "rows2" paths.
- **FlashAttention:** native tiled RDNA2 arithmetic/reductions.
- **Tensor parallel:** RCCL tuner + guarded host-snapshot/P2P all-reduce schedules, optional
  embedding-sharded LM head, and an optional "auto-expanded" TP4 P2P all-reduce fusion.
- **Graph fusion:** ADD/RMSNorm fusion, Q8_1 activation reuse, routed SwiGLU→Q8_1 staging, GDN sibling
  projection fusion (Qwen3.5/3.6 MoE).

### Optimization highlights (author-reported)

- **RCCL tensor-parallel all-reduce** — run with `GGML_CUDA_ALLREDUCE=nccl`; reported **+10% token
  generation and +20% prefill** on Qwen 122B.
- **Tensor-split HIP GPU sampling** — sampling runs on-GPU, split across cards.
- **rocPRIM partial TOP_K.**
- **Embedding-sharded Qwen output head** (122B and 35B).
- **RDNA2 Top-10 MoE routing** + compact routed MMQ tiles.
- **Laguna S.2 support** with DFlash.
- **DFlash2** speculative decoding with ngram-map and ngram-mod combos (see below).
- **RCCL autotuner** with custom all-reduce (recent `#llamacpp` update).
- **RDNA2-specific K-quant / MMQ LDS-tile optimizations** for **Q4 / Q5**.
- **Parallel multi-GPU weight uploads** with configurable buffer size (faster server startup).
- **AMD checkpoint handling** — backports several unmerged upstream PRs that fix AMD checkpoint issues
  with Qwen.

See the fork's `README.md` and its `docs/gfx1030-*` / `docs/rdna2-*` notes for the full, authoritative
list.

## Benchmarks (author-reported)

Point-in-time numbers using [`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2),
ROCm/RCCL, Flash Attention on, **F16 KV**, batch/ubatch **2048/256**, with the env prefix
`GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0
HSA_NO_SCRATCH_RECLAIM=1` (ROCm 7.14 libs on `LD_LIBRARY_PATH`).

### 4× V620 (24 CPU threads)

| Model | Quant | Split | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|
| Qwen3.6-27B | F16 | tensor | 850.54 | 18.52 |
| Qwen3.6-27B (DavidAU/Fable) | Q4_K_M | tensor | 1,017.25 | 27.76 |
| Qwen3.5-122B-A10B | Q4_K_M | tensor | 1,064.40 | 39.12 |
| Qwen3.6-35B-A3B | Q4_K_M | layer | 2,293.36 | 72.17 |
| Qwen3.6-35B-A3B | Q4_K_M | tensor | 1,939.80 | 58.30 |

### 2-GPU tensor split (TP2)

Reported by another community member on a 2-GPU tensor-split setup with the same fork:

| Model | Quant | Split | pp512 (t/s) | tg128 (t/s) |
|---|---|---|---:|---:|
| Qwen3.6-27B (DavidAU/Fable) | Q4_K_M | tensor | 707.14 | 25.63 |
| Qwen3.6-27B | F16 | tensor | 448.02 | 13.15 |

### Latest fork update (Aug 2026, `#llamacpp`)

Recent push to the fork added DFlash2, RCCL autotuner, and spec-decoding optimizations. Author-reported
MTP-4 TP4 numbers on a pelican prompt:

| Quant | Average t/s | Peak t/s |
|---|---:|---:|
| Q4_0 + MTP-4 | 85 | 115 |
| Q8_0 + MTP-4@Q4_0 | 75 | 105 |

DFlash2 on TP2 held ~52 t/s coding at 0–31k context and ~40 t/s at 64k. Checkpoint fix + MTP
optimization in [PR #12](https://github.com/edwinbrowwn/llama.cpp-rdna2/pull/12) (in review).

> `pp512` = prompt/prefill throughput (512-token prompt); `tg128` = token-generation throughput
> (128 tokens). For the A3B MoE model, **layer split** beats tensor split here; for the dense/large
> models, tensor split is used. These are snapshots on individual hosts — run matched before/after tests
> on your own hardware and commit before drawing conclusions.

## Upcoming optimizations (PR #10)

> **Status:** in review in [PR #10](https://github.com/edwinbrowwn/llama.cpp-rdna2/pull/10) — another
> round of optimizations plus an upstream llama.cpp merge, expected to land shortly. Details may change
> on merge, and the environment variables below will be **consolidated post-merge** (some become opt-in /
> debug-only, e.g. `NCCL_P2P_LEVEL`, `GGML_TP_SHARDED_OUTPUT`).

Reported improvements:

- **15.9–61.2% lower MMVQ operation latency**
- up to **17.7% faster Gated DeltaNet prefill**
- **+4.39% text-generation** speedup from DeltaNet sibling fusion

What's in it:

- Bounded six-row Q4_K/Q6_K MMVQ dispatch
- Chunked Gated DeltaNet prefill loads
- Opt-in tiled FlashAttention arithmetic
- Fused SwiGLU and DeltaNet sibling projections (Qwen3.5 / 3.6)
- Native Q4 and Q8 optimizations
- **MTP now works up to 6** with minimal performance hit on low acceptance

### Peak throughput (4× V620 @ 140 W power limit)

| Model | Quant | tg (t/s) | pp (t/s) |
|---|---|---:|---:|
| Qwen3.5-122B-A10B | Q4 | 100+ | ~1300 |
| Qwen3.6-35B-A3B | Q4 | 180+ | ~3800 |
| Qwen3.6-27B | Q8 | 75 | ~880 |

Highest throughput was achieved on **coding workflows**. Note the **140 W** cap — see
[Power Tuning](./tuning_power.md) for lowering the V620 power limit.

### Long-context PP HIP crash fix

A HIP crash that triggered mostly during long-context prefill is fixed (workaround) in this PR. Upstream
issue: [ROCm/rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817).

### Environment used for these results

To be consolidated post-merge (kept here for reproducibility while PR #10 is in review):

```bash
GGML_HIP_SAFE_STATE_IO=1 GGML_HIP_GFX1030_Q8_CACHE=1 GGML_HIP_GFX1030_GDN_SIBLING_FUSION=1 \
GGML_HIP_GFX1030_Q8_1_FUSION=1 GGML_HIP_GFX1030_NATIVE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PXB \
GGML_TP_SHARDED_OUTPUT=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 GGML_CUDA_P2P=1 GGML_HIP_GRAPHS=1
```

## Requirements

- Linux, CMake, and a ROCm install with HIP clang and **RCCL**.
- For the validated path: four V620 / `gfx1030` GPUs with tensor splitting.
- A compatible main GGUF; an optional DFlash/MTP draft GGUF.

## Build

The portable helper discovers ROCm, clang, RCCL, and the GPU arch where possible:

```bash
git clone https://github.com/edwinbrowwn/llama.cpp-rdna2.git
cd llama.cpp-rdna2
./scripts/build-rdna2-portable.sh
```

If discovery is ambiguous, pass values without editing the script:

```bash
ROCM_PATH=/path/to/rocm \
TARGET_ARCH=gfx1030 \
BUILD_DIR=build \
./scripts/build-rdna2-portable.sh
```

The helper enables HIP, RCCL, HIP graphs, the RDNA2 no-VMM path, shared/dynamic backends, the server,
examples, and tools, and builds Release.

### Maintainer helper (gfx1030 / ROCm 7.14)

`scripts/build-rdna2-rocm.sh` is the maintainer-oriented helper. The portable script above is the
recommended user entry point, but this one shows the exact gfx1030 configuration.

Current defaults:

```text
ROCm:   /opt/rocm/core-7.14
Target: gfx1030
Build:  ./build
```

Override when necessary:

```bash
ROCM_PATH=/opt/rocm \
TARGET_ARCH=gfx1030 \
./scripts/build-rdna2-rocm.sh
```

It configures:

```text
GGML_HIP=ON
GGML_HIP_RCCL=ON
GGML_HIP_GRAPHS=ON
GGML_HIP_NO_VMM=ON
GGML_NATIVE=ON
AMDGPU_TARGETS=gfx1030
CMAKE_HIP_ARCHITECTURES=gfx1030
LLAMA_BUILD_SERVER=ON
LLAMA_BUILD_TESTS=OFF
CMAKE_BUILD_TYPE=Release
```

## Launch

`HSA_OVERRIDE_GFX_VERSION=10.3.0` is **required** to activate the tested V620/`gfx1030` native profile —
don't force it on a different GPU architecture.

**TP2 and higher** (example uses four GPUs):

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_RDNA2_AUTO=1 \
GGML_HIP_SAFE_STATE_IO=1 \
GGML_TP_SHARDED_OUTPUT=1 \
./build/bin/llama-server \
  -m /path/to/main.gguf \
  -ngl all \
  --split-mode tensor \
  --tensor-split 1,1,1,1 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080
```

Use one `--tensor-split` value per device (TP2 → `--tensor-split 1,1`). For **TP1**, omit tensor
splitting and `GGML_TP_SHARDED_OUTPUT`:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_SAFE_STATE_IO=1 \
./build/bin/llama-server -m /path/to/main.gguf -ngl all --flash-attn on --host 0.0.0.0 --port 8080
```

- RCCL is auto-selected after an RCCL build; setting `GGML_CUDA_ALLREDUCE=nccl` explicitly is what the
  author reports gives **+10% tgen / +20% prefill** on Qwen 122B (see highlights above). Add
  `--device ROCm0,ROCm1,ROCm2,ROCm3` only to pin a device list.
- Optional validated four-V620 TP4 mode: `GGML_HIP_GFX1030_P2P_ALLREDUCE=auto-expanded` (shape/topology
  gated, safe fallback; leave unset elsewhere).

### Full validated example (4× V620, Qwen3.5-122B-A10B-MTP)

The author's full TP4 serving command for Qwen3.5-122B (adjust paths, context, and speculative-decoding
settings for your setup):

```bash
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
~/llama-cpp-mirrored/build/bin/llama-server \
  -m ~/models/Qwen3.5-122B-A10B-MTP-UD-Q4_K_M/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf \
  -ngl all --split-mode tensor --tensor-split 1,1,1,1 --main-gpu 0 \
  --ctx-size 262144 --cache-type-k f16 --cache-type-v f16 --kv-unified \
  --batch-size 2048 --ubatch-size 256 --parallel 4 \
  --host 0.0.0.0 --port 8080 --temp 0.6 --metrics --jinja --numa distribute \
  --reasoning on --reasoning-format deepseek --chat-template-kwargs '{"preserve_thinking": true}' \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-draft-p-min 0.0 --spec-draft-p-split 0.10 --no-spec-draft-backend-sampling --cache-ram 0
```

Note this profile pairs `GGML_CUDA_ALLREDUCE=nccl` with `GGML_CUDA_DISABLE_GRAPHS=1` and points
`LD_LIBRARY_PATH` at the ROCm 7.14 core libs.

### DFlash / MTP speculative decoding

```bash
  --spec-type draft-dflash \
  --spec-draft-model /path/to/dflash.gguf \
  --spec-draft-n-max 6
```

`--spec-draft-n-max` is a workload knob; start at the draft model's block size and tune.

### DFlash2 + ngram speculative decoding (community-tested)

Recent `#llamacpp` work added **DFlash2** with optional ngram helpers. DFlash2 shines on **real workloads**
(coding, agents, long context) more than short synthetic benches — community members saw DFlash2 alone
score lower than MTP on a 20k bench (~30 vs ~40 t/s) but win on actual agent/coding sessions (~47 t/s
with 700+ t/s prefill).

**Draft model quant:** use **Q4_K_M** for the DFlash2 draft GGUF — do **not** use Q8_0 for the drafter
(same acceptance, slower). Main model can stay Q4_0 / AutoRound for code.

**Recommended draft GGUFs** (Aug 2026):

- [`incoai/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
- [`webhie/Qwen3.8-27B-Q4-AutoRound-Code-GGUF`](https://huggingface.co/webhie/Qwen3.8-27B-Q4-AutoRound-Code-GGUF) — code-calibrated Q4_0 main model

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

DFlash2 is **more consistent at long context**; MTP acceptance tends to fall off. Combining DFlash2 +
ngram + MTP is possible but may not be worth the complexity. Use **tg1024+** (not tg32) for realistic
decode benchmarks — ngram pattern matching needs enough output tokens to show its value.

### Full DFlash2 serve example (TP4, Qwen3.8-27B)

Production command from the fork author (Aug 2026):

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

Recent fork builds report DFlash2 ~8% faster than earlier DFlash paths. Pull and rebuild regularly.

Combine with the standard RDNA2 env prefix. **MXFP4** quants (e.g. `quark75/Qwen3.8-27B-MXFP4-GGUF`)
are supported on the fork and pair well with MTP on TP2 — see the Docker serving example below.

### MTP on tensor-split setups

On TP2+, you can keep the main model tensor-split but pin the MTP draft to one GPU:

```bash
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-ngl 999 --spec-draft-device ROCm0
```

This runs the draft model on `ROCm0` while the main weights stay split across all cards.

## Docker serving example (community-validated)

A working TP2 + MTP4 setup from `#llamacpp` (ROCm 7.14, MXFP4 Qwen3.8-27B):

```yaml
environment:
  HSA_NO_SCRATCH_RECLAIM: "1"
  GGML_HIP_RDNA2_AUTO: "1"
  GGML_HIP_SAFE_STATE_IO: "1"
  GGML_TP_SHARDED_OUTPUT: "1"
  HSA_OVERRIDE_GFX_VERSION: "10.3.0"
command: >
  llama-server
  -hf quark75/Qwen3.8-27B-MXFP4-GGUF:MXFP4
  -ngl 999 -fa on -sm tensor -ts 1,1 -fit off
  --flash-attn on --ctx-size 32768
  --cache-type-k f16 --cache-type-v f16
  --batch-size 4096 --ubatch-size 4096 --parallel 1 --cont-batching
  --host 0.0.0.0 --port 8091 --temp 0.7 --jinja --metrics -kvu
  --reasoning-preserve
  --chat-template-kwargs '{"preserve_thinking": true}'
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-ngl 999
  --ctx-checkpoints 0
```

Build the image with `./scripts/build-rdna2-portable.sh` from the fork repo. RCCL needs working
[PCIe P2P](./tuning_p2p.md) for tensor split — use the [`v620_toolbox`](https://github.com/blivioniag/v620_toolbox)
P2P scripts to enable and verify it. If P2P is enabled but **slower** than without (common on gen3 x4
links), set `NCCL_P2P_DISABLE=1` or use the fork's P2P disable flag in the README. If all-reduce fails,
try `GGML_HIP_GFX1030_P2P_ALLREDUCE=off` or `GGML_CUDA_ALLREDUCE=none` (slower, but stable on broken
topologies). Always benchmark on your hardware — verified P2P bandwidth (~25 GB/s between V620 pairs) does
not guarantee faster inference if links are narrow or ACS-blocked.

## Notable limits

- Validated primarily on **4× V620 gfx1030, ROCm 7.14** and a specific PCIe topology; other systems keep
  conservative fallbacks.
- **Multi-socket** hosts (e.g. dual Xeon) can hurt TP performance — SP3/Epyc single-socket boards are
  preferred for multi-GPU tensor parallel in `#llamacpp` testing.
- **KV checkpoints** can crash on tensor-split setups (upstream and fork). Use `--ctx-checkpoints 0` to
  disable them if you hit `ggml-backend-meta.cpp` fatals during warmup.
- `GGML_TP_SHARDED_OUTPUT` and `GGML_TP_VOCAB_SHARDED_OUTPUT` are different, incompatible output-head
  modes; an external draft model can force a shared head to stay mirrored.
- **Tensor-split prefill can spike power** — if your PSU trips on tensor mode but layer split is fine, try
  lowering the power cap to 160 W (see [Power Tuning](./tuning_power.md); ~2–4% perf loss vs 180 W).
- Most single-flag `GGML_HIP_GFX1030_*` variables are redundant when the `HSA_OVERRIDE_GFX_VERSION=10.3.0`
  umbrella is active — leave them unset except for A/B testing (`GGML_HIP_RDNA2_AUTO=0` disables the
  profile). If RCCL all-reduce misbehaves on your topology, try `GGML_HIP_GFX1030_P2P_ALLREDUCE=off`.

For a plain, non-fork build see [Building & Running llama.cpp](./llama_cpp.md). Multi-GPU TP benefits
greatly from [PCIe P2P](./tuning_p2p.md).
