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
| [`opengfx1030/vllm-rdna` @ `rdna_extras`](https://github.com/opengfx1030/vllm-rdna) | **Official vLLM extras fork** — RDNA HIP kernels + Flash-Next cherry-picks. HEAD **v0.33.0** as of 1–2 Nov 2026; Hub bake still **`v0.31.0-extras`**. Org GHCR `ghcr.io/opengfx1030/vllm-rdna:nightly` — [Running](../vllm/running.md#ghcr-nightly), [fork landscape](../vllm/fork.md#fork-landscape) |
| [`BlivionIaG/hippihx`](https://github.com/BlivionIaG/hippihx) | HIP kernel op zoo (gfx1030 / gfx1100 / gfx900 tile contracts) — `#hippihx` Sep 16: intended to keep `rdna_extras` thin; **not** a published vLLM image yet |
| [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) [`#389`](https://github.com/ashhart/TensorFold/pull/389) | Draft **ROCm** backend (`--backend rocm`) for MLX affine checkpoints (Qwen3.5 / 3.8 dense, Qwen3.6-35B-A3B). **RDNA2** is FP16 `v_dot2`, **not** WMMA. Earlier community work: [`BlivionIaG/TensorFold#1`](https://github.com/BlivionIaG/TensorFold/pull/1). `#tensorfold` (Oct 4): **not faster than vLLM** yet. `#tensorfold` (Oct 6): the **Python** engine was **frozen**; the ROCm path is being rewritten in **Zig**. No wiki serve recipe. Author PR has a V620 / W7800 matrix — **Needs verify**. |
| [`BlivionIaG/vllm-rdna-qa`](https://github.com/BlivionIaG/vllm-rdna-qa) | Maintainer QA / landing playbook for `opengfx1030/vllm-rdna` (`#vllm-rdna` Sep 17) — not a second kernel wiki |
| [`alanoo81/flashnext-v620-pp3`](https://github.com/alanoo81/flashnext-v620-pp3) | Temporary 3× V620 PP3 overlay (leapdragon image + org MoE HIP) — `#vllm-rdna` Sep 17–18; `#general` Sep 23 **unmaintained** (KV-pool dump loops at high concurrency) — [recipe](../vllm/flash-next-serve.md#flash-next-3x-moe-hip-overlay) |
| [`yiminyuan/vllm` @ `gfx1030/v0.28.0`](https://github.com/yiminyuan/vllm/tree/gfx1030/v0.28.0) | Community DeepSeek-V4 Flash gfx1030 tree — `#vllm-rdna` Sep 17; [weights](https://huggingface.co/yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16); **Needs verify** |
| [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next) | Older **Flash-Next** container / `docs/rdna2` — prefer org `rdna_extras` / Hub **`v0.31.0-extras`** for day-to-day; tree still receives occasional commits (last push 11 Oct 2026) — [ROCR idle-CPU fix](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/ROCR-CPU-FIX.md) |
| `blivioniag/vllm` @ `rdna2_extras` | **Historical** — predecessor of `opengfx1030/vllm-rdna`; public repo **404** as of Oct 2026. Current Hub extras bakes clone the org repo; only older `v0.26` / `v0.27.1-extras*` bake targets still name this clone |
| [`leapdragon/vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe) | Community recipe book + GHCR presets (27B/122B; parts pile; concurrent MTP PRs) — [wiki recipes](../vllm/recipes.md) |
| [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe) | Org mirror of the recipe book (`#vllm-rdna`) |
| [`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2) | RDNA2/V620 llama.cpp fork — see [overview](../llama-cpp/rdna2-overview.md) |
| [`Niko1221/Strata`](https://github.com/Niko1221/Strata) | Separate HIP engine (not llama.cpp / vLLM). gfx1030 notes in [`docs/AMD_HIP.md`](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md). Opt-in `STRATA_HIP_PROMPT_F16=1` (PR **#835**) runs 16-bit prompt GEMMs FP16 I/O on gfx103x. `#general` Oct 7–8: 2× V620 0.1.39 + **#835/#849** ~doubled PP vs 0.1.38; 0.1.40.2 similar with the env on. 3-card: prefill still behind vLLM; decode collapsed with concurrency. `#strata` (25 Oct): **0.1.49.2** signed DeepSeek-engine feed has **no gfx1030** HIP package (HIP build lists gfx1200/gfx1201; CUDA packages are a different arch). `#vllm-rdna` (28 Oct): easier onboarding than vLLM; more interesting below 4 cards / when experts spill. `#strata` (31 Oct–1 Nov): **`--calibrate`** is the host-tuning knob people A/B; one **4× V620** host used `--prefill-pipe 3072` + `STRATA_SH_STREAM=1`. Decode still behind vLLM. **Avalon** (llama.cpp fork, `#general` / `#strata` 31 Oct) is **RDNA3-only** — not a V620 path. `#strata` (3 Nov): **2× V620** extra streams gave little/no total TG (two streams **slower**); second host also a miss. `#strata` (5 Nov): `STRATA_HIP_PROMPT_F16=1` still first (one host **~500 → ~1000** PP); then **`--tune thorough`**. Same thread: **4× V620** **Q8** prefill beat **Q4** (~1900 PP on Q4) while decode was similar — **Needs verify**. `#strata` (7–9 Nov): Qwen3.8 Flash-Next **Q8** + **VS Code Copilot** dropped tool calls. **`tool_call_recovery`** is **on since 0.2.8** ([DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)); it recovered some malformed calls but did not stop the misses. One host had no failures with **pi.dev**. Community V620 patch stack: [`andrewprock/Strata` `v620-stack-0210-f2-mmq`](https://github.com/andrewprock/Strata/tree/v620-stack-0210-f2-mmq). `#strata` (10 Nov): stock Strata **DeepSeek** still not AMD — use Maya / [Cilla](https://github.com/crispyging/cilla-v620). **2× V620** Qwen Flash-Next community ballpark **~1200–1600** PP / **~40–70** TG on IQ3-class packs with **128 GB** DRAM (one Q3 host ~800/40). `#strata` (11 Nov): OS-level **2 MiB hugepages** (`vm.nr_hugepages`) was required for stability on one **2×** host (Strata **0.2.7** DeepSeek note; not a JSON flag; container untested). That **IQ3_S** host ~**1200–2000** PP / ~**50–70** TG. Prefer **QFN / q3.8fn** over Mix-128G-class DeepSeek packs. **Needs verify.** **No wiki serve recipe.** |
| [`crispyging/cilla-v620`](https://github.com/crispyging/cilla-v620) | Experimental **gfx1030** HIP fork of [Project Maya](https://github.com/mw00/project-maya) (Strata-descended). `#general` (9 Nov): public `release/cilla-v620`. README: DeepSeek-V4.1-Flash-1017, **ROCm 7.2.4**, Python 3.11, four-V620 opt-in profile in [`docs/LAUNCHING.md`](https://github.com/crispyging/cilla-v620/blob/release/cilla-v620/docs/LAUNCHING.md). Pinned Engram + copied CPU experts can take **hundreds of GB** host RAM. **No tested 8× preset.** `#strata` (10 Nov): one **2× V620** + **128 GB DDR4** Maya DeepSeek host ~**30** t/s decode / ~**200** PP vs ~**1.5k** Qwen Flash PP on the same cards. `#strata` (11 Nov): Mix-128G-class DeepSeek packs were a poor **2×** fit; if **Engram** hits **NVMe**, tokens can drop to **single digits** — keep the table in RAM. Follow the public README — not Discord private launchers. **No wiki serve recipe.** |
| [`BlivionIaG/hi2`](https://github.com/BlivionIaG/hi2) | Quantizer for `rdna_hint_flex` (variable INT2/INT4/INT8 rows). `#vllm-rdna` / `#hi2` (27 Oct): intended middle ground vs EXL3 (quality) and AWQ (speed). `#hi2` (1 Nov): example pack [`BlivionIaG/Qwen3.8-27B-HI2-3.0bpw`](https://huggingface.co/BlivionIaG/Qwen3.8-27B-HI2-3.0bpw) (~15 GiB). Still **no vLLM recipe / Hub bake**. |
| [`GeorgeMA-Strong/llm-context-bench`](https://github.com/GeorgeMA-Strong/llm-context-bench) | Reproducible long-context PP/TG benches (real prompts) — used by `#benchmarks` |
| [`vllm-project/vmon`](https://github.com/vllm-project/vmon) | Upstream vLLM author monitor for prefill/decode viz — `#vllm-rdna` Oct 13; **not wiki-validated** on gfx1030 |
| [`LMCache/LMCache`](https://github.com/LMCache/LMCache) | KV cache layer (RAM/SSD/remote) — `#lmcache` WIP; use [`lmcache/standalone`](https://hub.docker.com/r/lmcache/standalone) (CPU) + connector. Official kernels **CDNA-only**; hybrid Qwen needs HMA the connector lacks — [troubleshooting](../troubleshooting/vllm.md#upstream-kv-offload-tanks-decode) |
| [`2kiss/flash-attention-rdna2`](https://github.com/2kiss/flash-attention-rdna2) | Independent RDNA2 FlashAttention kernel — `#vllm-rdna` Sep 22–24: **stale**; org target remains in-tree `fa_rdna2` (currently disabled after `#17`) — [fork](../vllm/fork.md#attention) |
| [`intentee/paddler`](https://github.com/intentee/paddler) | LLM load balancer / multi-instance router — discussed for multi-agent llama.cpp; **not wiki-validated** |
| [`sebastianmechno-sys/vllm-rocm-windows-rdna2`](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2) | Unofficial Windows 11 + ROCm 7.x vLLM for RX 6000 — **not wiki-validated** |
| [`skyne98/wiki-gfx906`](https://github.com/skyne98/wiki-gfx906) | Sibling wiki for gfx906 (Vega 20 / MI50) |

## Tools & write-ups

- [Unsloth: Qwen3.8-Flash-Next / MTP llama.cpp](https://unsloth.ai/docs/models/qwen3.8-next) —
  public MTP build notes (`#llamacpp` / `#general` Sep 16–17).
- [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  — drop-in Jinja for official Qwen chat templates (`#vllm-rdna` Oct 8; also usable with llama.cpp
  `--chat-template-file`). Community: fewer empty-think / KV-invalidation stalls. `#general`
  (Oct 9): tool calls holding up (OG Qwen3.8 + Open WebUI). **Needs verify.**
- [Disabling ECC on Radeon Pro GPUs (lunnova.dev)](https://lunnova.dev/articles/amdgpu-disabling-ecc/) —
  `amdgpu.ras_enable=0` + two reboots; pinned in `#vllm-rdna`. See [Disabling ECC](../tuning/ecc.md).
- `rocminfo`, `rocm-smi`, `amd-smi`, `clinfo` — GPU visibility, topology, and P2P checks. See
  [Environment variables](../reference/env-vars.md#handy-commands). `rocm-smi --showvbios` and
  AMD `amdvbflash` identify / dump stock V620 images — [hardware](../setup/hardware.md#v620-vbios).
  Community D60306 dump: [`baldpope/uncategorized` `amd_v620/`](https://github.com/baldpope/uncategorized/tree/main/amd_v620).
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
