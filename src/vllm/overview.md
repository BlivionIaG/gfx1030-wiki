# vLLM on RDNA — Overview

> **WIP:** This section is actively expanded from Discord and fork release notes. See
> [Verification status](../../reference/verification.md) before treating benchmarks as gospel.

The quickest way to serve LLMs on gfx1030 / RDNA is the prebuilt
[`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) images on a
[`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) ROCm + PyTorch base. A single
image targets seven RDNA architectures (gfx1030 through RDNA4).

## Where to start

| Goal | Page |
|---|---|
| Pull an image and run your first model | [Running (Docker)](../running.md) |
| Which stack / model / card count | [Recipes](../recipes.md) |
| Env vars, CUDA graphs, Docker Compose | [Configuration](../configuration.md) |
| GPTQ vs AWQ, KV cache, MTP, INT4 | [Quantization](../quantization.md) |
| Custom RDNA2 HIP kernels (`-extras`) | [vLLM forks](../fork.md) (`rdna_extras`) |
| Which fork / Flash-Next / official org | [Fork landscape](../fork.md#fork-landscape) |
| Rebuild or extend Docker images | [Building images](../images.md) |
| Something broke | [vLLM troubleshooting](../../troubleshooting/vllm.md) |

## Image variants

| Variant | When to use |
|---|---|
| `v0.27.1` / `v0.27.1-rocm7.14.0` | Stock upstream vLLM — baseline or comparison. |
| `v0.27.1-extras` / `v0.27.1-extras-rocm7.14.0` | Official extras kernels — [`rdna_extras`](../fork.md#the-rdna_extras-fork) lineage; **recommended** day-to-day on gfx1030. |
| `v0.28.0-extras` | `#vllm-rdna` Sep 18 **test** bake of the 0.28 extras line. **Needs verify** — Hub CI is not fully tracked; A/B against `v0.27.1-extras` before treating as default. |

Image tags are **refreshed in place** when fixes land — always `docker pull` before debugging. Confirm your
`-extras` image includes the latest extras commits (AWQ dispatch, GDN HIP, TP graph fix).
`#vllm-rdna` (Aug 31 2026): `blivioniag/vllm-rdna:v0.27.1-extras` was refreshed again — re-pull even if
you already had that tag.

> **Official source moved.** Kernel work lives in
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) (`rdna_extras`). Day-to-day serving
> is still Hub **`blivioniag/vllm-rdna:*-extras`** (`v0.27.1-extras*` day-to-day; `v0.28.0-extras` is a
> Sep 18 **test** tag). Published bake still clones the historical
> personal fork. `#vllm-rdna` Sep 14–15: Flash-Next commits are **in `rdna_extras`**; the standalone
> [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen) branch is no longer
> the moving target. Published Flash-Next **containers** may still be the older leapdragon image.
> Details: [Fork landscape](../fork.md#fork-landscape).
>
> **Upstream vLLM 0.28.x:** Official GPU docs still omit Navi 21 / gfx1030. Keep community forks until
> upstream documents it.

### Qwen3.8 Flash-Next on vLLM

llama.cpp still struggles with Flash-Next on gfx1030 (upstream gaps). On **4× V620**, community
production is the Flash-Next vLLM stack — **not** Hub `-extras`. `#vllm-rdna` Sep 14–15: that work
now lives on [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras`;
[`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next)
is the older published container / docs tree and is **not being updated**. Prefer latest `rdna_extras`
(or the last known-good leapdragon image if you need a container today) over llama.cpp until the
RDNA2 llama.cpp fork catches up.

**2× V620 / RAM offload:** vLLM is a poor fit when weights or MoE experts must spill to host RAM.
`#vllm-rdna` (Sep 14): use [llama.cpp](../../llama-cpp/rdna2-speculative.md#flash-next-2x-iq4)
(`--n-cpu-moe`, `-ot …=CPU`) instead. Community 2-card llama.cpp Flash-Next is still **~25 t/s**
decode / **~150–200** PP — far below the 4-card vLLM class.

**3× V620:** vLLM wants **PP=3** (even TP). `#vllm-rdna` (Sep 15–16): a leapdragon image ran Flash-Next
PP3 without MTP; `rdna_extras` HEAD **still corrupted** decode on a fresh pull —
[troubleshooting](../../troubleshooting/vllm.md#flash-next-pp3-output-corruption).
Fitting MTP on 3 cards is [community / Needs verify](../recipes.md#flash-next-3x-pp3-mtp).
Sep 17–18: a public overlay that ports org **MoE HIP** onto leapdragon reports **~1078–2493** PP /
**~57–63** TG with MTP k=2 — [overlay](../recipes.md#flash-next-3x-moe-hip-overlay).

**4× V620 + `rdna_extras` graphs (`#vllm-rdna` Sep 16–24):** the **measured `#17` path** is
**`FULL_DECODE_ONLY`** — [Sep 23 serve notes](../recipes.md#flash-next-4x-pr17). Older hosts that
saw **FULL** decode **corrupt** stayed on **`PIECEWISE`** —
[FULL-graph corruption](../../troubleshooting/vllm.md#flash-next-full-graph-corruption) and the
[Sep 17 serve line](../recipes.md#flash-next-4x-piecewise). Community snapshot on that older recipe:
**~3331 tok/s** PP / **~73 tok/s** TG @ 16k/1k, c=8 (**Needs verify**). There is **no Q3** path on
vLLM.

`#vllm-rdna` (Sep 19): **pipeline parallel 4** on Flash-Next was reported to hit **~1950 tok/s**
prefill at 32k while **decode collapsed**. The same host dropped PP4 and continued on **TP=4**
(~**1550** tok/s prefill at long context; decode **~30–35 t/s** with MTP-2). Treat PP4 as a
prefill experiment, not a 4-card default. **Needs verify.**

`#vllm-rdna` (Sep 20): [`opengfx1030/vllm-rdna#15`](https://github.com/opengfx1030/vllm-rdna/pull/15)
**merged** into `rdna_extras`. AMD **QSA prefill scoring is bounded to the live prompt context**
(decode / missing metadata still use the capacity-wide path). The same PR folds V620 startup
prerequisites from superseded PR `#12`. Combined-stack benches on **TP=4 / PP=1 / EP=4 / MTP-2**,
Intel AutoRound INT4 experts, FP16 dense, original BF16 PLE in CPU RAM — **Community / Needs verify**
(full host stack, not an isolated QSA A/B):

| Suite | Context | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Regular | 16k | 2,011 | 63.5 |
| Coding | 16k | 2,036 | 68.5 |
| Regular | 32k | 2,024 | 55.2 |
| Regular | 64k | 1,974 | 59.4 |
| Coding | 64k | 1,986 | 70.6 |
| Regular | 128k | 1,859 | 53.9 |
| Coding | 128k | 1,832 | 72.5 |

The PR records a **31–33%** 16k/32k prefill lift versus an immediately collected **~1529–1543 tok/s**
baseline on that same combined stack. Community take: **near-PP4 prefill with TP4 decode**. TP2+PP2
was not tried. Fused QSA multi-step draft decode with **MTP-3** did not help. **INT8 decode shadows
are not in this merge.** Pull latest `rdna_extras` — Hub `-extras` tags still lag.

`#vllm-rdna` (Sep 21): **do not treat the ~1.8–2.0k PP table as the public-merge floor.** Two other
4× hosts on `rdna_extras` after `#15` landed **~1.07–1.45k tok/s** prefill. Decode often **did**
reach the PR band (**~50–77 t/s**) once CUDA graphs were on (one host: **~24 t/s** eager vs
**~66–76 t/s** with graphs). One of those hosts had octachannel DDR4-3200, PCIe gen4 x16 on all
four cards, and NVMe ~6.5 GB/s — so the gap is **not** just “slow RAM / gen3”. A second host
(DDR4-2144 ECC, PCIe gen3, SATA SSD, ~96 GB RAM, int4 PLE in RAM) sat around **~1.1–1.2k** PP /
**~45–50 t/s** with MTP-1. The PR author later said the 1350 → 1550 → 2000 prefill steps needed
**two local changes that were not on the git checkout** others pulled, and planned to re-apply
those from current `rdna_extras` through git. `#vllm-rdna` (Sep 22): the author later **recovered
~1950 tok/s PP** on a local tree and said something was **still missing** before those changes
could land on the public branch. Treat **~1.1–1.5k PP** as the **`#15`-only** public-merge class.

`#vllm-rdna` (Sep 23): that missing stack **merged** as
[`opengfx1030/vllm-rdna#17`](https://github.com/opengfx1030/vllm-rdna/pull/17) into `rdna_extras`.
PR-head 16k (same `llm-context-bench` harness): **1,957.8 / 1,982.8 tok/s** PP, decode **~69 / ~69
t/s**. A second 4× host **confirmed** those cells (**1,983.8 / 1,985.3 tok/s** PP) after rebuilding
TunableOp for its rocBLAS — [mismatch](../../troubleshooting/vllm.md#tunableop-rocblas-mismatch).
The `#15` author-host **~2k** table is no longer “unpublished”; Hub `-extras` still lags. **Community
/ Needs verify** on your topology.

`#vllm-rdna` (Sep 2026) ballpark on **4× V620** (host-dependent; fork author + community):

| Metric | Earlier recipe | After Sep 4–6 prefill/decode work |
|---|---|---|
| Sustained **prefill** | ~580–700 tok/s | **~1000–1400 tok/s** (fork author RESULTS.md ~1080–1180; `#vllm-rdna` Sep 13 community ~1100–1390 @ 32k–128k after latest Flash-Next pull) |
| **Decode** (single-stream) | ~50–64 tok/s | **~60–100+ tok/s** class depending on MTP acceptance / prompt (fork RESULTS.md; community Sep 13 ~53–78 t/s on long suites, MTP=2) |
| vs llama.cpp Flash-Next | — | Community: container **~40 t/s** vs llama.cpp ROCm **~18–19 t/s** on the same host |

Docs live under
[`docs/rdna2/`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next/docs/rdna2)
(`README.md`, `RESULTS.md`, `TROUBLESHOOTING.md`, `ROCR-CPU-FIX.md`). Pull latest before re-benching —
tags and container `latest` move with the prefill campaign.

**Long-prompt stalls / timeouts:** if large agentic prompts (tens of k tokens) hang or take many minutes
while short prompts are fine — or GPUs sit at **100% util / ~40 W** with TTFT swinging from seconds
to minutes — try `VLLM_USE_V2_MODEL_RUNNER=0`. `#vllm-rdna` Sep 15: that switch unblocked a TP4
leapdragon container. Community: **MoE on the 0.28 Flash-Next line wants the V1 runner** until
upstream **0.29** (V2 becomes the default). See
[vLLM troubleshooting](../../troubleshooting/vllm.md#flash-next-long-prompt-stalls).

**KV / concurrency (4× 32 GB, `#vllm-rdna` Sep 14):** Flash-Next KV is expensive. Community ballpark
**~24 GiB ≈ 300k tokens** once the PLE sidecar is loaded — a 4× V620 box can still be **tight**
(~280k tokens left on one host). Offloading KV to **128 GB** of system RAM was **not** enough;
budget **>128 GB** host RAM if you try that path. Concurrent streams split decode (community: total
TG stayed near single-stream ~70–80 t/s while **PP fell to ~500 t/s**) and re-prefill on every
tool-call session. Prefer **one stream** plus prefix / radix cache; do not expect a linear
multi-agent multiplier.

`#vllm-rdna` (Sep 17): the **logged** hybrid KV token count can be **~2.5× too high** versus a
measured peak — size `--max-model-len` from a real request, not the banner
([overstated pool](../../troubleshooting/vllm.md#flash-next-hybrid-kv-overstated)). Prefix cache on
`rdna_extras` needed
[`e45dd5cb`](https://github.com/opengfx1030/vllm-rdna/commit/e45dd5cb2de8218defe19878fd75e39528f0acbc)
before repeat prompts actually hit
([zero hits](../../troubleshooting/vllm.md#flash-next-prefix-cache-zero)).

### Intel AutoRound Flash-Next (draft, `#vllm-rdna` Sep 10–11 2026) {#intel-autoround-flash-next}

A second Flash-Next track is the Intel **W4A16 AutoRound** checkpoint
([`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound);
community also cites the RTN sibling
[`Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound)).
Weights are ~**75 GB**; plan on **4× V620**. Community quality notes: Intel AutoRound beats some
other INT4 Flash-Next packs on published tool-calling scores (example cited: **80.5** vs **76.0**
for a different INT4). Treat those as **publisher / community numbers**, not wiki benches.

Draft integration: [`opengfx1030/vllm-rdna#5`](https://github.com/opengfx1030/vllm-rdna/pull/5)
(open, **not** in Hub `-extras`). PR validation used **FP16 activations**, **CPU PLE / n-gram
offload**, and **`--max-num-batched-tokens 4096`**. Published short-run figures on **4× V620**
(no P2P in the Discord report):

| Metric | Community / PR snapshot |
|---|---|
| Uncached 1024-token prefill | **~962 tok/s** (vs ~535 BF16 on that host) |
| Short decode + MTP | **~41 tok/s**, ~61% MTP accept |
| Prose / code 16–64k | **~950–980 tok/s** PP, **~48–56 tok/s** decode |
| 128k after `4096` batched tokens | PP stays **~950 tok/s** class (was **~375 tok/s** at 2048 scheduled tokens) |
| Community long-suite (Sep 16, 4× V620) | Regular **32–128k**: **~1296–1393 tok/s** PP / **~56–60 tok/s** TG; coding suites **~68–73 tok/s** TG. Pin cited: `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` @ `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`. Further push hit **int8 decode-shadow** quality loss. |
| QSA live-context (`rdna_extras` `#15`, Sep 20) | **Author-host** combined stack **~1830–2040 tok/s** PP / **~54–73 tok/s** TG. `#vllm-rdna` Sep 21: other 4× hosts on `#15` only reproduced **~1.1–1.45k** PP. The unpublished recovery **merged as `#17`** (Sep 23) — see [QSA merge](#qwen38-flash-next-on-vllm). |
| Startup (warm-ish) | ~**4 min** vs earlier **8–10 min** on the same host |
| fp16 KV fit | ~**291k** tokens on 4 cards (one report) |

**Gotchas** (do not treat as a drop-in Hub image):

- Known-good PLE is the **embedded BF16** n-gram table (~**95 GiB** tensor data) with
  `--engram-config '{"cpu_offload":true}'`. **Quantized CPU PLE** (group-16 INT4 sidecar) is **not**
  end-to-end validated; `#vllm-rdna` saw **incoherent** generation on that path.
- `--max-num-batched-tokens 2048` + chunked prefill was the suspected cause of the **128k PP cliff**.
  Raise to **4096** (or see [troubleshooting](../../troubleshooting/vllm.md#flash-next-128k-prefill-cliff)).
- Concurrent varlen GDN prefill still has a fork kernel addressing bug
  (`gdn_prefill_o_rdna2.cu` reuses a global chunk index). Single-stream benches can look fine.
- The PR is huge and **dirty** against `rdna_extras` — wait for a cleaned cherry-pick / image
  before calling it production. Day-to-day Flash-Next remains
  [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen).

Also watch [TheRock ROCR idle-CPU spin](../../troubleshooting/vllm.md#rocr-idle-cpu-spin-therock-714)
on ROCm 7.14 hosts.

## llama.cpp vs vLLM

For a short comparison table (battle-tested GGUF vs agentic / Flash-Next), see
[llama.cpp overview](../../llama-cpp/overview.md#llamacpp-vs-vllm-on-v620-llamacpp--vllm-rdna).

## What fits well on V620 {#what-fits-well-on-v620}

`#general` (Sep 2026) consensus — RDNA2 has **no matrix / tensor cores**, so **prefill** on dense
models is the weak spot (agentic “read a pile of files” workloads frustrate people even when decode
looks fine):

| Workload | Community take |
|---|---|
| **MoE** (e.g. Qwen3.6 **35B-A3B**, Ornith-class) | Sweet spot on 1–4× V620 |
| Dense **27B** | Usable; expect mediocre PP vs newer silicon |
| **Flash-Next** | Promising on **4×** V620 via the [vLLM recipe](#qwen38-flash-next-on-vllm); not a 1-card path |
| Dense agentic on 1–2 cards | Often disappointing TTFT / PP — prefer MoE or more cards |

## Related

- [Multi-GPU PCIe P2P](../../tuning/p2p.md) — important for tensor parallel.
- [Environment variables](../../reference/env-vars.md) — cheat-sheet.
- [Recipes (wiki)](../recipes.md) — pick Hub `-extras` vs recipe container vs Flash-Next by card count.
- [Community recipe book](https://github.com/leapdragon/vllm-rdna2-recipe) (mirror
  [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe)) — presets,
  patches, concurrent MTP PRs.
