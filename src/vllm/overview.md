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
| Custom RDNA2 HIP kernels (`-extras`) | [rdna2_extras fork](../fork.md) |
| Rebuild or extend Docker images | [Building images](../images.md) |
| Something broke | [vLLM troubleshooting](../../troubleshooting/vllm.md) |

## Image variants

| Variant | When to use |
|---|---|
| `v0.27.1` / `v0.27.1-rocm7.14.0` | Stock upstream vLLM — baseline or comparison. |
| `v0.27.1-extras` / `v0.27.1-extras-rocm7.14.0` | [`rdna2_extras`](../fork.md) fork with native RDNA2 kernels — **recommended on gfx1030**. |

Image tags are **refreshed in place** when fixes land — always `docker pull` before debugging. Confirm your
`-extras` image includes the latest `rdna2_extras` commits (AWQ dispatch, GDN HIP, TP graph fix).
`#vllm-rdna` (Aug 31 2026): `blivioniag/vllm-rdna:v0.27.1-extras` was refreshed again — re-pull even if
you already had that tag.

> **Upstream vLLM 0.28.x:** Discord asked whether stock vLLM now lists gfx1030. Official 0.28 GPU docs
> still target CDNA / RDNA3+ families, not Navi 21. Keep using the community `-extras` images (or the
> [rdna2_extras fork](../fork.md)) until an upstream release explicitly documents gfx1030. `#vllm-rdna`
> (Sep 2026): community work is rebasing HIP kernels toward **0.28** and consolidating into
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) — published Docker tags may lag
> that org until CI images catch up.

### Qwen3.8 Flash-Next on vLLM

llama.cpp still struggles with Flash-Next on gfx1030 (upstream gaps). Community production path is the
[`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next)
branch / recipe: `#vllm-rdna` reports cold **~580–700 tok/s prefill** and **~50–53 tok/s decode** on
16k–32k-tier prompts without MTP (ballpark; host-dependent). Prefer that stack over llama.cpp for
Flash-Next until the RDNA2 llama.cpp fork catches up.

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
