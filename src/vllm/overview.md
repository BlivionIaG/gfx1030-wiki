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
fork — not the Hub `-extras` image: `#vllm-rdna` reports cold **~580–700 tok/s prefill** and **~50–53 tok/s
decode** on 16k–32k-tier prompts without MTP (ballpark; host-dependent). Prefer that stack over llama.cpp
for Flash-Next until the RDNA2 llama.cpp fork catches up. Flash-Next work is expected to land in
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) after the 0.28 rebase / merge.

Also watch [TheRock ROCR idle-CPU spin](../../troubleshooting/vllm.md#rocr-idle-cpu-spin-therock-714)
on ROCm 7.14 hosts.

## llama.cpp vs vLLM

For a short comparison table (battle-tested GGUF vs agentic / Flash-Next), see
[llama.cpp overview](../../llama-cpp/overview.md#llamacpp-vs-vllm-on-v620-llamacpp--vllm-rdna).

## Related

- [Multi-GPU PCIe P2P](../../tuning/p2p.md) — important for tensor parallel.
- [Environment variables](../../reference/env-vars.md) — cheat-sheet.
- [Community recipes](https://github.com/leapdragon/vllm-rdna2-recipe) — external collection (includes
  concurrent MTP / prefill-vs-decode work — check open PRs).
