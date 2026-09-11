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

Image tags are **refreshed in place** when fixes land — always `docker pull` before debugging. Confirm your
`-extras` image includes the latest extras commits (AWQ dispatch, GDN HIP, TP graph fix).
`#vllm-rdna` (Aug 31 2026): `blivioniag/vllm-rdna:v0.27.1-extras` was refreshed again — re-pull even if
you already had that tag.

> **Official source moved.** Kernel work lives in
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) (`rdna_extras`). Day-to-day serving
> is still Hub **`blivioniag/vllm-rdna:*-extras`** (v0.27.1; bake still clones the historical personal
> fork). Flash-Next = **`leapdragon/vllm-rdna2-qwen`** until that line merges into the org. Details:
> [Fork landscape](../fork.md#fork-landscape).
>
> **Upstream vLLM 0.28.x:** Official GPU docs still omit Navi 21 / gfx1030. Keep community forks until
> upstream documents it.

### Qwen3.8 Flash-Next on vLLM

llama.cpp still struggles with Flash-Next on gfx1030 (upstream gaps). Community production path is the
[`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next)
fork — not the Hub `-extras` image. Prefer that stack over llama.cpp for Flash-Next until the RDNA2
llama.cpp fork catches up. Flash-Next work is expected to land in
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) after the 0.28 rebase / merge.

`#vllm-rdna` (Sep 2026) ballpark on **4× V620** (host-dependent; fork author + community):

| Metric | Earlier recipe | After Sep 4–6 prefill/decode work |
|---|---|---|
| Sustained **prefill** | ~580–700 tok/s | **~1000–1200 tok/s** (fork author; RESULTS.md ~1080–1180 @ 3k–30k) |
| **Decode** (single-stream) | ~50–64 tok/s | **~60–100+ tok/s** class depending on MTP acceptance / prompt (fork RESULTS.md; community warm benches ~85 t/s) |
| vs llama.cpp Flash-Next | — | Community: container **~40 t/s** vs llama.cpp ROCm **~18–19 t/s** on the same host |

Docs live under
[`docs/rdna2/`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next/docs/rdna2)
(`README.md`, `RESULTS.md`, `TROUBLESHOOTING.md`, `ROCR-CPU-FIX.md`). Pull latest before re-benching —
tags and container `latest` move with the prefill campaign.

**Long-prompt stalls / timeouts:** if large agentic prompts (tens of k tokens) hang or take many minutes
while short prompts are fine, try `VLLM_USE_V2_MODEL_RUNNER=0` — community report of stable **~68 t/s**
with dense INT8 + custom all-reduce after that switch; fork docs now call it out. See
[vLLM troubleshooting](../../troubleshooting/vllm.md#flash-next-long-prompt-stalls).

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
- [Community recipes](https://github.com/leapdragon/vllm-rdna2-recipe) — external collection (includes
  concurrent MTP / prefill-vs-decode work — check open PRs).
