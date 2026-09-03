# Useful Resources

Curated links for gfx1030 / RDNA2 work. Grouped by type — not everything here is “official” or
wiki-maintained; see [Verification status](../reference/verification.md) for claim audits on guides
that reference these projects.

## Community Discord

Primary hangout for V620 / gfx1030 inference work. Wiki pages often cite channel names
(`#vllm-rdna`, `#llamacpp`, `#general`, `#benchmarks`) — that is where the claim came from.

| | |
|---|---|
| **Server** | **gfx1030 club** |
| **Invite** | [https://discord.gg/mESex2aBp](https://discord.gg/mESex2aBp) |

## Official documentation

- [ROCm documentation](https://rocm.docs.amd.com/)
- [ROCm install on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [System requirements & supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [GPU architecture specs (Navi 21 = gfx1030)](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [ROCm Device Support Wishlist](https://github.com/ROCm/ROCm/discussions/4276) — community-tracked
  support matrix on the official ROCm repo

## Docker images

Prebuilt images for RDNA (gfx1030 through RDNA4). See [Running vLLM](../vllm/running.md) for run
commands and [Building images](../vllm/images.md) for how they are produced.

| Image | Purpose |
|---|---|
| [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) | ROCm + PyTorch base for RDNA cards |
| [`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) | vLLM serving images (upstream and `-extras` / `rdna_extras` lineage) |

Tags are listed on Docker Hub and in [Running (Docker)](../vllm/running.md#image-matrix). Re-pull before
debugging — tags are refreshed in place when fixes land.

## Upstream projects

Stock projects this wiki builds on or tracks. Use these when you want upstream behavior or to compare
against forks.

- [vLLM](https://github.com/vllm-project/vllm) — upstream serving engine
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF inference (HIP/ROCm and Vulkan)
- [vLLM PR #52391](https://github.com/vllm-project/vllm/pull/52391) — RDNA gfx1030 platform detection
  (upstream CI enablement for consumer Radeon)
- [TheRock: RDNA2 (gfx103X) support](https://github.com/ROCm/TheRock/pull/1629) — ROCm/TheRock gfx1030
  enablement work

## Community projects

Forks, recipes, and tooling maintained outside (or alongside) upstream. Often where gfx1030-specific
performance work happens first.

| Project | What it is |
|---|---|
| [`blivioniag/v620_toolbox`](https://github.com/blivioniag/v620_toolbox) | V620 power cap + PCIe P2P — [`powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/powertuning) (Fedora), [`ubuntu_powertuning/`](https://github.com/blivioniag/v620_toolbox/tree/master/ubuntu_powertuning) (Ubuntu 26.04); see [Power tuning](../tuning/power.md) |
| [`blivioniag/vllm-rdna-docker`](https://github.com/blivioniag/vllm-rdna-docker) | Docker build system for `rocm-rdna` / `vllm-rdna` images |
| [`opengfx1030/vllm-rdna` @ `rdna_extras`](https://github.com/opengfx1030/vllm-rdna) | **Official vLLM extras fork** — RDNA HIP kernels; PRs/issues here — [fork landscape](../vllm/fork.md#fork-landscape) |
| [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next) | **Flash-Next / Qwen3.8** fork (still separate until merged into the org) — [ROCR idle-CPU fix](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/ROCR-CPU-FIX.md) |
| [`blivioniag/vllm` @ `rdna2_extras`](https://github.com/blivioniag/vllm/tree/rdna2_extras) | **Historical** — predecessor of `opengfx1030/vllm-rdna`; Hub `-extras` still clones this until docker bake is retargeted |
| [`leapdragon/vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe) | Community compose/env recipes (not a separate engine fork; watch open PRs for concurrent MTP) |
| [`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2) | RDNA2/V620 llama.cpp fork — see [overview](../llama-cpp/rdna2-overview.md) |
| [`GeorgeMA-Strong/llm-context-bench`](https://github.com/GeorgeMA-Strong/llm-context-bench) | Reproducible long-context PP/TG benches (real prompts) — used by `#benchmarks` |
| [`sebastianmechno-sys/vllm-rocm-windows-rdna2`](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2) | Unofficial Windows 11 + ROCm 7.x vLLM for RX 6000 — **not wiki-validated** |
| [`skyne98/wiki-gfx906`](https://github.com/skyne98/wiki-gfx906) | Sibling wiki for gfx906 (Vega 20 / MI50) |

## Tools & write-ups

- [Disabling ECC on Radeon Pro GPUs (lunnova.dev)](https://lunnova.dev/articles/amdgpu-disabling-ecc/) —
  `amdgpu.ras_enable=0` + two reboots; pinned in `#vllm-rdna`. See [Disabling ECC](../tuning/ecc.md).
- `rocminfo`, `rocm-smi`, `amd-smi`, `clinfo` — GPU visibility, topology, and P2P checks. See
  [Environment variables](../reference/env-vars.md#handy-commands).

> Know a gfx1030 resource that belongs here? [Contribute](./contributing.md) a link and say which section
> it fits (official / Docker / upstream / community).
