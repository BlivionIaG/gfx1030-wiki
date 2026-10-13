# vLLM on RDNA — Overview

> **WIP:** This section is actively expanded from Discord and fork release notes. See
> [Verification status](../reference/verification.md) before treating benchmarks as gospel.

The quickest way to serve LLMs on gfx1030 / RDNA is the prebuilt
[`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) images on a
[`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) ROCm + PyTorch base. A single
image targets seven RDNA architectures (gfx1030 through RDNA4).

Not sure whether to use vLLM or llama.cpp? Start with [llama.cpp or vLLM](../choose-a-stack.md).

## Where to start

| Goal | Page |
|---|---|
| Pull an image and run your first model | [Running (Docker)](./running.md) |
| Which image / model / card count (27B, presets) | [Recipes](./recipes.md) |
| GLM-5.3-Flash on 8× (local 0.30 fork) | [Recipes — GLM-5.3-Flash](./recipes.md#glm-53-flash) |
| Qwen3.8 Flash-Next (benchmarks and serve lines) | [Flash-Next](./flash-next.md) |
| Env vars, CUDA graphs, Docker Compose | [Configuration](./configuration.md) |
| GPTQ vs AWQ, KV cache, MTP, INT4 | [Quantization](./quantization.md) |
| Custom RDNA2 HIP kernels (`-extras`) | [vLLM forks](./fork.md) |
| Which fork is official vs historical | [Fork landscape](./fork.md#fork-landscape) |
| Host `uv` venv (7.2.0 or 7.14.0) | [Host venv](./host-venv.md) |
| One launcher for Flash-Next / 27B / EXL3 (`serve_rdna.sh`) | [Serve scripts](./serve-scripts.md) |
| Rebuild or extend Docker images | [Building images](./images.md) |
| Something broke | [vLLM troubleshooting](../troubleshooting/vllm.md) |

## Image variants

| Variant | When to use |
|---|---|
| `v0.31.0-extras` / `v0.31.0-extras-rocm7.14.0` | Official extras on **vLLM 0.31.0** — [`rdna_extras`](./fork.md#the-rdna_extras-fork). `#announcements` (11 Oct 2026): **recommended** day-to-day Docker path (ROCm 7.14). Flash-Next MTP-2 (BF16 + INT4-disk) served from this tag. Same digest under both names. |
| `v0.28.0-extras` / `v0.27.1-extras*` | Historical extras bakes. Source for 0.28 is [`rdna_extra/v0.28.0`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extra/v0.28.0). Keep for A/B only. |
| `v0.27.1` / `v0.27.1-rocm7.14.0` | Stock upstream vLLM — baseline or comparison. `#vllm-rdna` Sep 24: plain **`v0.27.1-rocm7.14.0`** is **stale / removable**; prefer `-extras` or [host venv](./host-venv.md). |

Image tags are **refreshed in place** when fixes land — always `docker pull` before debugging. Confirm your
`-extras` image includes the latest extras commits (AWQ dispatch, GDN HIP, TP graph fix).
`#vllm-rdna` (Aug 31 2026): `blivioniag/vllm-rdna:v0.27.1-extras` was refreshed again — re-pull even if
you already had that tag.

> **Official source moved.** Kernel work lives in
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) (`rdna_extras` = **v0.31.0** as of
> 11 Oct 2026). Day-to-day serving is Hub **`blivioniag/vllm-rdna:v0.31.0-extras`**. Older
> `v0.27.1-extras*` / `v0.28.0-extras*` tags stay on Hub for A/B. `#vllm-rdna` Sep 14–15: Flash-Next
> commits are **in `rdna_extras`**; the standalone
> [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen) branch is no longer
> the moving target. Prefer the 0.31 Hub tag or a [host venv](./host-venv.md) over the old leapdragon image.
> Details: [Fork landscape](./fork.md#fork-landscape).
>
> **Upstream vLLM 0.28.x:** Official GPU docs still omit Navi 21 / gfx1030. Keep community forks until
> upstream documents it.

## Related

- [Choose a stack](../choose-a-stack.md) — llama.cpp vs vLLM, and what fits on a V620.
- [Flash-Next](./flash-next.md) — card counts, AutoRound, and serve lines.
- [Multi-GPU PCIe P2P](../tuning/p2p.md) — important for tensor parallel.
- [Environment variables](../reference/env-vars.md) — cheat-sheet.
- [Community recipe book](https://github.com/leapdragon/vllm-rdna2-recipe) (mirror
  [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe)) — presets,
  patches, concurrent MTP PRs.
