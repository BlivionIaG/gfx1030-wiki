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

## Related

- [Multi-GPU PCIe P2P](../../tuning/p2p.md) — important for tensor parallel.
- [Environment variables](../../reference/env-vars.md) — cheat-sheet.
- [Community recipes](https://github.com/leapdragon/vllm-rdna2-recipe) — external collection.
