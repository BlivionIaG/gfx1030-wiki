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

Public `rdna_extras` README (25 Nov) names upstream **vLLM v0.34.0**. That
retraces the 21–24 Nov Discord “0.34” talk that the wiki had parked as
unconfirmed while the README still said 0.33. **Hub bake is still 0.31.**
GHCR **`:v0.34.0`** is an older snapshot (19 Nov, **no Inkling**) — pull
`:nightly` / `:nightly-0.34.0-rocm7.14.0` or clone HEAD.
[Running](running.md#ghcr-nightly).

`#vllm-rdna` (24 Nov): the host that reported an **~18 h** crash was still on
**0.33**, serving a **Qwen3.8-27B** tune — **Needs verify**, not an Inkling
retraction. [Troubleshooting](../troubleshooting/vllm.md#033-long-run-crash).

Prefer the in-tree vLLM recipes on **HEAD** (clone + [host venv](host-venv.md),
or org GHCR [nightly](running.md#ghcr-nightly) after `#211`). `#general`
(21–22 Nov) also had a **llama.cpp GGUF** path — [community llama.cpp](#llamacpp-gguf).

## Recipes

Files: [`tools/rdna/recipes/inkling2-mini-*.env`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/tools/rdna/recipes).
Guides: [`inkling2-serving.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/inkling2-serving.md),
[`inkling2-dspark.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/inkling2-dspark.md).

[`#208`](https://github.com/opengfx1030/vllm-rdna/pull/208) (24 Nov) folded the
old `*-dspark-adaptive` / `tp2-dspark` / `tp4-dspark` files into four recipes.
[`#211`](https://github.com/opengfx1030/vllm-rdna/pull/211) (25 Nov) added a
fifth (`inkling2-mini-tp1-offload`) for 16 GiB cards.

```sh
MODEL=/path/to/inkling2-mini VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=inkling2-mini-tp1
```

| Recipe | Cards | Notes |
|---|---|---|
| `inkling2-mini-tp1` | **1× 32 GB** | Plain decode. `MAXLEN=194000` after [`#209`](https://github.com/opengfx1030/vllm-rdna/pull/209) (was 180k). `SEQS=16`, `FULL_AND_PIECEWISE`, prefix cache, one image/prompt. |
| `inkling2-mini-tp1-dspark` | **1× 32 GB** | DSpark on. `MAXLEN=135000` (`ADAPTIVE=1` → `125000`). Pass `DRAFT_VOCAB=` for the 32k-id file from `tools/rdna/inkling2/draft_vocab.py`. |
| `inkling2-mini-tp1-offload` | **1× 16 or 32 GB** | [`#211`](https://github.com/opengfx1030/vllm-rdna/pull/211). TP=1 expert spill (`EXPERT_SLOTS=48`). `MAXLEN=65536`. Simulated 16 GiB peak **12.94 GiB**. Optional `EXPERT_CACHE_COPY=native`. **TP>1 refused.** |
| `inkling2-mini-tp2` | **2×** | DSpark **on by default** (`MAXLEN=720000`). `DSPARK=0` for the 1M cap (`MAXBAT=320`). `ADAPTIVE=1` / `DRAFT_VOCAB=` work at TP=2 after `#208`. |
| `inkling2-mini-tp4` | **4×** | DSpark **on by default**; full **1,048,576** still fits. `DSPARK=0` for plain decode. |

Pick the GPU with `HIP_VISIBLE_DEVICES=<index>` — do not copy another host’s
device list out of the recipe comments. TunableOp is **off** on these recipes
(no rocBLAS GEMM on the serving path).

Default speed switches (leave on unless A/B’ing):
`VLLM_INKLING2_PREFILL_BM=1`, `VLLM_INKLING2_FUSE_GLUE=1`,
`VLLM_INKLING2_DECODE_FUSED=1`, `VLLM_INKLING2_FUSE_ATTN_CHAIN=1`,
`VLLM_INKLING2_PREFILL_ATTN_TILE` on (`0` is the shipped tile).
`VLLM_INKLING2_PREFILL_INT8=1` is experimental and **changes quality**.

### LoopCD (opt-in) {#loopcd}

[`#210`](https://github.com/opengfx1030/vllm-rdna/pull/210) (25 Nov): opt-in
LoopCD (`arXiv:2610.02185`) for Inkling-2-Mini. Default **`LOOPCD_ALPHA=0`**
(off; bit-identical). `#general` (24 Nov) discussed the paper the same day the
PR landed. Not a day-to-day recipe — A/B with
[`eval_loopcd.py`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/tools/rdna/inkling2/eval_loopcd.py)
and the explore write-up
[`docs/explore/loopcd-2610.02185/README.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/explore/loopcd-2610.02185/README.md).

Launcher knobs: `LOOPCD_ALPHA`, `LOOPCD_MODE=hidden|logits`, `LOOPCD_ADAPTIVE`,
`LOOPCD_BETA`, `LOOPCD_BLEND_BLOCKS`, `LOOPCD_WEAK_PASS` (also `VLLM_INKLING2_LOOPCD_*`).
Author-host: **hidden** `alpha=0.35` is near-free; **logits** `alpha=0.25`
adaptive lifted a 20-problem Math+Code set **75% → 90%** and costs ~10% decode.
**Fork-source**, not wiki-tested.

## Author-host numbers (Fork-source)

One **V620** unless noted. **Not** wiki-tested. Prefill TP=2/4 cells below are
after [`#205`](https://github.com/opengfx1030/vllm-rdna/pull/205) (24 Nov):
NVFP4 prefill used to fall back to a less-optimized kernel at TP=2/4 — Discord
`#general` the same morning.

| Setup | Prefill | Decode |
|---|---|---|
| TP=1 plain, 1k (`#209`) | — | **~55.5** tok/s (~18 ms) |
| TP=1 DSpark K=2, 1k (`#209`) | — | **~104** tok/s |
| TP=1 DSpark adaptive + 32k vocab, 1k RAG (`#209`) | — | **~172** tok/s (~5.8 ms) |
| TP=1 DSpark adaptive, 16k chat (`#209`) | — | **~161** tok/s |
| TP=1 DSpark, 1k c=1 (`#207`) | **~2085** tok/s | **~94** tok/s/req (real-text ~70) |
| TP=1 DSpark, c=8 (`#207`) | — | **~185** tok/s agg |
| TP=2 1k after `#205` | **~2334** tok/s (was ~1698) | — |
| TP=2 16k after `#205` | **~2330** tok/s (was ~1635) | — |
| TP=2 DSpark, 1k c=1 (`#207`) | **~3044** tok/s | **~136** tok/s/req |
| TP=4 1k after `#205` | **~3884** tok/s (was ~3112) | — |
| TP=4 DSpark, 1k c=1 (`#207`) | **~3867** tok/s | **~164** tok/s/req |
| TP=1 offload `#211` (32 GB, native copy) | 1k **~512** / 16k **~645** | b=1 **~19** / b=8 **~65** tok/s |

TP=1 conversation grown to **~144k** with prefix cache (older `#196` table):
new-token prefill falls from ~1800 to ~730 tok/s; decode from ~54 to ~36 tok/s.
Prefix cache **off** at that depth is minutes of TTFT — leave it on.

`#211` offload is slower than resident TP=1. Use it when the card is **16 GB**
or you want extra KV on 32 GB (`MAXLEN=430000` verified on the author 32 GB
host). 16 GB simulated peak **12.94 GiB**, KV **~71k**.

## vs Flash-Next

`#general` (23 Nov): on **4× V620** hosts, community still prefers
**Flash-Next** for day-to-day. Inkling-2-Mini is the **1×** looping-MoE path
(HEAD also has TP=2/4) — **not** a QFN replacement. Same thread: more mileage
from **TP2 Flash-Next INT4 MoE offload** than from switching a 4-card QFN box
to Inkling.

`#general` (24 Nov): at **W4A16**, Inkling Mini is **not** competitive with
Qwen3.8 Flash-Next and is only barely in the Qwen3.8 **27B** quality band.
People who like it are usually on **1–2 cards**. Treat as **Opinion /
Community**. Independent **BetterBench 0.6.1** weighted decode (23 Nov,
**Community**): **~56** tok/s TP=1 / **~79** tok/s TP=2 — same ballpark as
the older short-agent numbers. Not wiki-tested.

`#strata` (23–25 Nov): **Inkling-2-Mini on Strata** only used **2 GPUs** on a
4-card host and was slow (~250 PP / ~40 TG in a later try). Use these vLLM
HEAD recipes, not Strata, for Inkling.

## llama.cpp GGUF {#llamacpp-gguf}

`#general` (21–22 Nov): community **1× V620** llama.cpp (not the vLLM recipes)
on Unsloth **Q4_K_XL** (~17–20 GB class): about **90** tok/s decode at 4k
context, **~50** at 230k; later **~2k** tok/s prefill after raising
`--batch-size` / `--ubatch-size` (one host used **16384 / 1024**; another
OOM’d above **512** ubatch). Native FP4 looping model — GGUF is a requant,
not the HIP NVFP4 path. **Community / Needs verify.** No wiki command line.

`#general` (24 Nov): a later 1× V620 llama.cpp first-pass (~824 PP / ~29 TG)
is thinner than that GGUF table — skip it as a recipe.

## Next

- [Serve scripts](serve-scripts.md) — `RECIPE=` table
- [Recipes](recipes.md) — 1× card-count pick
- [Fork progress](fork-progress.md)
- [Choose a stack](../choose-a-stack.md)
