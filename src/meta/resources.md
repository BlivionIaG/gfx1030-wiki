# Useful Resources

Curated links for gfx1030 / RDNA2 work. Grouped by type — not everything here is “official” or
wiki-maintained; see [Verification status](../reference/verification.md) for claim audits on guides
that reference these projects.

## Community Discord

Primary hangout for V620 / gfx1030 inference work. Wiki pages often cite channel names
(`#vllm-rdna`, `#llamacpp`, `#general`, `#benchmarks`, `#lmcache`) — that is where the claim came from.

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
| [`Tamalero/amd-v620-soft-unlock`](https://github.com/Tamalero/amd-v620-soft-unlock) | V620 OverDrive soft unlock in **passthrough VMs** (QEMU `romfile=` pptable patch; no flash, keeps 72 CUs) — [Power tuning](../tuning/power.md#soft-unlock-passthrough-vms) |
| [`opengfx1030/vllm-rdna-docker`](https://github.com/opengfx1030/vllm-rdna-docker) | Docker bake for `rocm-rdna` / `vllm-rdna` — **moved here** `#vllm-rdna` Sep 14 2026; Hub tags still `blivioniag/*` — [Building images](../vllm/images.md) |
| [`BlivionIaG/vllm-rdna-docker`](https://github.com/BlivionIaG/vllm-rdna-docker) | **Historical** bake repo (pre-move) |
| [`opengfx1030/vllm-rdna` @ `rdna_extras`](https://github.com/opengfx1030/vllm-rdna) | **Official vLLM extras fork** — RDNA HIP kernels + Flash-Next cherry-picks (Sep 14–15); PRs/issues here — [fork landscape](../vllm/fork.md#fork-landscape) |
| [`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx) | HIP kernel op zoo (gfx1030 / gfx1100 / gfx900 tile contracts) — `#hippihx` Sep 16: intended to keep `rdna_extras` thin; **not** a published vLLM image yet |
| [`BlivionIaG/vllm-rdna-qa`](https://github.com/BlivionIaG/vllm-rdna-qa) | Maintainer QA / landing playbook for `opengfx1030/vllm-rdna` (`#vllm-rdna` Sep 17) — not a second kernel wiki |
| [`alanoo81/flashnext-v620-pp3`](https://github.com/alanoo81/flashnext-v620-pp3) | Temporary 3× V620 PP3 overlay (leapdragon image + org MoE HIP) — `#vllm-rdna` Sep 17–18; `#general` Sep 23 **unmaintained** (KV-pool dump loops at high concurrency) — [recipe](../vllm/recipes.md#flash-next-3x-moe-hip-overlay) |
| [`yiminyuan/vllm` @ `gfx1030/v0.28.0`](https://github.com/yiminyuan/vllm/tree/gfx1030/v0.28.0) | Community DeepSeek-V4 Flash gfx1030 tree — `#vllm-rdna` Sep 17; [weights](https://huggingface.co/yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16); **Needs verify** |
| [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next) | Older **Flash-Next** container / `docs/rdna2` — **not being updated** (`#vllm-rdna` Sep 15); keep for published images — [ROCR idle-CPU fix](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/ROCR-CPU-FIX.md) |
| [`blivioniag/vllm` @ `rdna2_extras`](https://github.com/blivioniag/vllm/tree/rdna2_extras) | **Historical** — predecessor of `opengfx1030/vllm-rdna`; Hub `-extras` still clones this until docker bake is retargeted |
| [`leapdragon/vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe) | Community recipe book + GHCR presets (27B/122B; parts pile; concurrent MTP PRs) — [wiki recipes](../vllm/recipes.md) |
| [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe) | Org mirror of the recipe book (`#vllm-rdna`) |
| [`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2) | RDNA2/V620 llama.cpp fork — see [overview](../llama-cpp/rdna2-overview.md) |
| [`GeorgeMA-Strong/llm-context-bench`](https://github.com/GeorgeMA-Strong/llm-context-bench) | Reproducible long-context PP/TG benches (real prompts) — used by `#benchmarks` |
| [`LMCache/LMCache`](https://github.com/LMCache/LMCache) | KV cache layer (RAM/SSD/remote) — `#lmcache` WIP; use [`lmcache/standalone`](https://hub.docker.com/r/lmcache/standalone) (CPU) + connector. Official kernels **CDNA-only**; hybrid Qwen needs HMA the connector lacks — [troubleshooting](../troubleshooting/vllm.md#upstream-kv-offload-tanks-decode) |
| [`2kiss/flash-attention-rdna2`](https://github.com/2kiss/flash-attention-rdna2) | Independent RDNA2 FlashAttention kernel — `#vllm-rdna` Sep 22–24: **stale**; org target remains in-tree `fa_rdna2` (currently disabled after `#17`) — [fork](../vllm/fork.md#attention) |
| [`intentee/paddler`](https://github.com/intentee/paddler) | LLM load balancer / multi-instance router — discussed for multi-agent llama.cpp; **not wiki-validated** |
| [`sebastianmechno-sys/vllm-rocm-windows-rdna2`](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2) | Unofficial Windows 11 + ROCm 7.x vLLM for RX 6000 — **not wiki-validated** |
| [`skyne98/wiki-gfx906`](https://github.com/skyne98/wiki-gfx906) | Sibling wiki for gfx906 (Vega 20 / MI50) |

## Tools & write-ups

- [Unsloth: Qwen3.8-Flash-Next / MTP llama.cpp](https://unsloth.ai/docs/models/qwen3.8-next) —
  public MTP build notes (`#llamacpp` / `#general` Sep 16–17).
- [Disabling ECC on Radeon Pro GPUs (lunnova.dev)](https://lunnova.dev/articles/amdgpu-disabling-ecc/) —
  `amdgpu.ras_enable=0` + two reboots; pinned in `#vllm-rdna`. See [Disabling ECC](../tuning/ecc.md).
- `rocminfo`, `rocm-smi`, `amd-smi`, `clinfo` — GPU visibility, topology, and P2P checks. See
  [Environment variables](../reference/env-vars.md#handy-commands).
- [UEFITool](https://github.com/LongSoft/UEFITool) / [ReBARUEFI](https://github.com/xCuri0/ReBARUEFI)
  — locate hidden MMIO High / ReBAR / SR-IOV setup vars. See [Host firmware](../setup/host-firmware.md).
- [Precision 7920 MMIO GPU Unlock Guide](https://github.com/user-attachments/files/28905671/Precision_7920_MMIO_GPU_Unlock_Guide.pdf)
  — circulating T5820/T7820/T7920 write-up; **dump your own UEFI**, do not copy another host's
  offsets.
- [GpuMMIOFix](https://github.com/Radi0Glitch/GpuMMIOFix) — OS-side PCI BAR remap above 4 GB
  (`#motherboard` Sep 15–16). Community: typically **reload each boot**; **Needs verify** on V620
  hosts — [Host firmware](../setup/host-firmware.md).

> Know a gfx1030 resource that belongs here? [Contribute](./contributing.md) a link and say which section
> it fits (official / Docker / upstream / community).
