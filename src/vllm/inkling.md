# Inkling-2-Mini (vLLM HEAD)

> **WIP / HEAD-only.** Native gfx1030 serving landed on
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras`
> ([`#195`](https://github.com/opengfx1030/vllm-rdna/pull/195) 22 Nov 2026,
> [`#196`](https://github.com/opengfx1030/vllm-rdna/pull/196) 23 Nov).
> **Not** in Hub **`v0.31.0-extras`**. See
> [Verification status](../reference/verification.md#inklingmd).

[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
is a looping MoE (24 layers, 128 experts, NVFP4 + MXFP8 experts, vision tower).
On gfx1030 the fork dequantizes those packs **inside the GEMM** and converts
dense attention / `lm_head` **bf16 → fp16 at load**. Serving compute is HIP
(no rocBLAS, no Triton on the recipe path).

`#vllm-rdna` / `#general` (21 Nov) discussed a V620 port while the PRs were
still landing. `#general` (21–22 Nov) also had a **llama.cpp GGUF** path —
[community llama.cpp](#llamacpp-gguf). Prefer the in-tree vLLM recipes on
**HEAD** (clone + [host venv](host-venv.md), or org GHCR
[nightly](running.md#ghcr-nightly) after `#196`).

Public `rdna_extras` README on 23 Nov still names upstream **vLLM v0.33.0**.
`#vllm-rdna` / `#general` (21 Nov) said “0.34.0 based now” — **do not** treat
HEAD or Hub as 0.34 until the README / a release says so. `#vllm-rdna` (24 Nov):
the host that reported an **~18 h** crash was still on **0.33**, serving a
**Qwen3.8-27B** tune — **Needs verify**, not a recipe retraction.
[Troubleshooting](../troubleshooting/vllm.md#033-long-run-crash).

## Recipes

Files: [`tools/rdna/recipes/inkling2-mini-*.env`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/tools/rdna/recipes).
Guides: [`inkling2-serving.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/inkling2-serving.md),
[`inkling2-dspark.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/inkling2-dspark.md),
[`inkling2-benchmarks.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/inkling2-benchmarks.md).

```sh
MODEL=/path/to/inkling2-mini VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1
```

| Recipe | Cards | Notes |
|---|---|---|
| `inkling2-mini-tp1` | **1×** | `FULL_AND_PIECEWISE`, prefix cache, one image/prompt. `MAXLEN=180000` (weights ~19 GiB; KV ~32 KiB/token). Model limit is **1,048,576** — use TP=2/4 for that. |
| `inkling2-mini-tp2` | **2×** | 1M context. TP=1 prefill can still win at short prompts (all-reduce). |
| `inkling2-mini-tp4` | **4×** | Same 1M cap; faster long-context prefill in the `#195` table. |
| `inkling2-mini-tp1-dspark` | **1×** | Opt-in DSpark draft. Cuts KV (`MAXLEN=125000` in the file). |
| `inkling2-mini-tp1-dspark-adaptive` | **1×** | Adaptive K + 32k-id draft vocab. **TP=1 and greedy only.** `MAXLEN=105000`. Build the ids file once with `tools/rdna/inkling2/draft_vocab.py`. |
| `inkling2-mini-tp2-dspark` | **2×** | Static K=2. Draft vocab is **refused at TP>1**. |

Pick the GPU with `HIP_VISIBLE_DEVICES=<index>` — do not copy another host’s
device list out of the recipe comments. TunableOp is **off** on these recipes
(no rocBLAS GEMM on the serving path).

Default speed switches (leave on unless A/B’ing):
`VLLM_INKLING2_PREFILL_BM=1`, `VLLM_INKLING2_FUSE_GLUE=1`,
`VLLM_INKLING2_DECODE_FUSED=1`, `VLLM_INKLING2_FUSE_ATTN_CHAIN=1`,
`VLLM_INKLING2_PREFILL_ATTN_TILE` on (`0` is the shipped tile).
`VLLM_INKLING2_PREFILL_INT8=1` is experimental and **changes quality**.

## Author-host numbers (Fork-source)

One **V620**, native weights, HTTP c=1 unless noted. After `#196` (pipelined
prefill attention tile). **Not** wiki-tested.

| Setup | Prefill | Decode |
|---|---|---|
| TP=1, 2k prompt | **~1977** tok/s | short agent **~56** tok/s (~18 ms/tok) |
| TP=1, 16k prompt | **~1798** tok/s | **~52** tok/s after 16k |
| TP=1, 65k prompt | **~1443** tok/s | — |
| TP=1 DSpark adaptive + 32k vocab | prefill unchanged | **~1.4×** vs no-draft (held-out mix ~12.5 ms/tok) |
| TP=2, 65k prompt | **~1385** tok/s | short agent **~80** tok/s |
| TP=2 / TP=4, ~1.05M prompt (`#195`) | ~3000 s / ~1517 s | **~54 / ~32** ms/tok |

TP=1 conversation grown to **~144k** with prefix cache: new-token prefill falls
from ~1800 to ~730 tok/s; decode from ~54 to ~36 tok/s (`#196`). Prefix cache
**off** at that depth is minutes of TTFT — leave it on.

Known limits from the PRs: serving tables are mostly **c=1**; adaptive K at
TP=2 is **not** a recipe; DSpark spends KV on a replicated draft; a rare
engine-exit abort (~2/50) has no root cause yet.

## vs Flash-Next

`#general` (23 Nov): on **4× V620** hosts, community still prefers
**Flash-Next** for day-to-day. Inkling-2-Mini is the **1×** looping-MoE path
(HEAD also has TP=2/4) — **not** a QFN replacement. Same thread: more mileage
from **TP2 Flash-Next INT4 MoE offload** than from switching a 4-card QFN box
to Inkling. Independent **BetterBench 0.6.1** weighted decode (same day,
**Community**): **~56** tok/s TP=1 / **~79** tok/s TP=2 — same ballpark as the
author-host short-agent numbers above. Not wiki-tested.

`#strata` (23–24 Nov): **Inkling-2-Mini on Strata** only used **2 GPUs** on a
4-card host and was slow. Use these vLLM HEAD recipes, not Strata, for Inkling.

## llama.cpp GGUF {#llamacpp-gguf}

`#general` (21–22 Nov): community **1× V620** llama.cpp (not the vLLM recipes)
on Unsloth **Q4_K_XL** (~17–20 GB class): about **90** tok/s decode at 4k
context, **~50** at 230k; later **~2k** tok/s prefill after raising
`--batch-size` / `--ubatch-size` (one host used **16384 / 1024**; another
OOM’d above **512** ubatch). Native FP4 looping model — GGUF is a requant,
not the HIP NVFP4 path. **Community / Needs verify.** No wiki command line.

## Next

- [Serve scripts](serve-scripts.md) — `RECIPE=` table
- [Recipes](recipes.md) — 1× card-count pick
- [Fork progress](fork-progress.md)
- [Choose a stack](../choose-a-stack.md)
