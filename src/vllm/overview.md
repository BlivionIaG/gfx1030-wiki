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
| **First serve on a V620** (Docker or fork + launcher) | [Onboard on V620](./onboarding.md) |
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
| `v0.31.0-extras` / `v0.31.0-extras-rocm7.14.0` | **Current** day-to-day Docker — `#announcements` 11 Oct 2026 (ROCm 7.14, torch 2.12, Triton 3.7.1). Matches `rdna_extras` = upstream **vLLM 0.31.0**. Start at [Onboard](./onboarding.md). |
| `v0.27.1` / `v0.27.1-rocm7.14.0` | Stock upstream vLLM — baseline or comparison. `#vllm-rdna` Sep 24: plain **`v0.27.1-rocm7.14.0`** is **stale / removable**. |
| `v0.27.1-extras` / `v0.27.1-extras-rocm7.14.0` | Older extras bake — keep for A/B only. Prefer **`v0.31.0-extras`**. |
| `v0.28.0-extras` | Sep 18 **test** bake of the 0.28 extras line. Superseded by **0.31** for day-to-day. |

Image tags are **refreshed in place** when fixes land — always `docker pull` before debugging. Confirm your
`-extras` image includes the latest extras commits (AWQ dispatch, GDN HIP, TP graph fix).
`#vllm-rdna` (Aug 31 2026): `blivioniag/vllm-rdna:v0.27.1-extras` was refreshed again — re-pull even if
you already had that tag.

> **Official source:** [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
> (`rdna_extras` = upstream **vLLM 0.31.0** + RDNA port; older 0.28 line:
> `rdna_extra/v0.28.0`). Day-to-day Docker:
> Hub **`blivioniag/vllm-rdna:v0.31.0-extras`**. First-time path: [Onboard on V620](./onboarding.md).
> Details: [Fork landscape](./fork.md#fork-landscape).
>
> **Upstream vLLM:** Official GPU docs still omit Navi 21 / gfx1030. Keep the community fork until
> upstream documents it.

## Related

- [Choose a stack](../choose-a-stack.md) — llama.cpp vs vLLM, and what fits on a V620.
- [Flash-Next](./flash-next.md) — card counts, AutoRound, and serve lines.
- [Multi-GPU PCIe P2P](../tuning/p2p.md) — important for tensor parallel.
- [Environment variables](../reference/env-vars.md) — cheat-sheet.
- [Community recipe book](https://github.com/leapdragon/vllm-rdna2-recipe) (mirror
  [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe)) — presets,
  patches, concurrent MTP PRs.
