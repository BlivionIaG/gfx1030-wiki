# Introduction

> **Work in progress:** This wiki is being actively expanded from Discord research and fork release
> notes. Many tuning recipes and throughput numbers are **community-reported** and have not been
> independently reproduced here. See [Verification status](./reference/verification.md) for what is solid vs what
> still needs checking on your hardware.

Welcome to the **GFX1030 Wiki** — a focused, hands-on knowledge base for running LLM inference on AMD
**gfx1030** (RDNA2 / Navi 21) GPUs, with a strong bias toward the **Radeon PRO V620** and the
purpose-built tooling collected here.

`gfx1030` is the LLVM/ROCm target for the **Navi 21 "Sienna Cichlid"** die. It powers the consumer
Radeon **RX 6800 / 6800 XT / 6900 XT / 6950 XT** and the workstation/data-center **PRO W6800** and
**PRO V620**. These cards are officially supported by ROCm on Linux, which makes them a cost-effective
platform for modern LLMs — but getting the most out of them (power tuning, multi-GPU P2P, and RDNA-tuned
kernels) takes a bit of extra work. That's what this wiki documents.

## Read it in this order

1. [Getting started](./setup/getting-started.md) — hardware check, ROCm, optional V620 tuning, first container.
2. [llama.cpp or vLLM](./choose-a-stack.md) — which stack matches the model and card count.
3. The matching section: [vLLM](./vllm/overview.md) or [llama.cpp](./llama-cpp/overview.md).
4. [Troubleshooting](./troubleshooting/index.md) when a command fails. [Verification status](./reference/verification.md) says which numbers are solid.

Host firmware (the machine never POSTs) and Flash-Next serve lines are specialist pages. Skip them until you need them.

## What this wiki focuses on

- **[Tuning](./tuning/power.md)** — the [`v620_toolbox`](https://github.com/blivioniag/v620_toolbox)
  recipes: lowering the V620's VBIOS-locked 250 W floor to **120 W**, enabling **GPU↔GPU PCIe
  Peer-to-Peer** between multiple V620s (Fedora + AMD hosts; power tuning also on **Ubuntu 26.04**;
  Intel Ice Lake often sees no P2P win — [A/B it](./tuning/p2p.md#when-p2p-helps--and-when-it-does-not)),
  and optionally [disabling Pro-card ECC](./tuning/ecc.md) for ~2 GB extra VRAM.
- **[llama.cpp](./llama-cpp/overview.md)** — stock builds plus the RDNA2 fork (tensor parallel, MTP /
  DFlash). Usually the easiest first stack on V620.
- **[vLLM on RDNA](./vllm/overview.md)** — ready-to-run Docker images;
  [recipes](./vllm/recipes.md) for Hub vs recipe container, and a separate
  [Flash-Next](./vllm/flash-next.md) page
  ([`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) on a
  [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) PyTorch base), how they are
  built with [`opengfx1030/vllm-rdna-docker`](https://github.com/opengfx1030/vllm-rdna-docker), and
  the **official** vLLM extras fork [`opengfx1030/vllm-rdna`](./vllm/fork.md) (`rdna_extras`, now
  including Flash-Next cherry-picks). Published Hub `-extras` tags still lag HEAD.

New here? Start with [Getting Started](./setup/getting-started.md). Card identity is on [Supported Hardware](./setup/hardware.md).

## Community Discord

Much of this wiki is distilled from the **gfx1030 club** Discord — live tuning tips, fork releases,
and multi-GPU benches land there first.

- **Server:** gfx1030 club
- **Join:** [https://discord.gg/mESex2aBp](https://discord.gg/mESex2aBp)
- Useful channels: `#vllm-rdna`, `#llamacpp`, `#general`, `#benchmarks`, `#harnesses`, `#lmcache`,
  `#motherboard`

When turning Discord threads into docs, follow the privacy rules in [Contributing](./meta/contributing.md#privacy-discord--wiki).

> **Disclaimer:** This is a community wiki, not affiliated with or endorsed by AMD. Kernel patches and
> power-cap changes are done at your own risk. Always cross-check against the official
> [ROCm documentation](https://rocm.docs.amd.com/). Contributions welcome — see
> [Contributing](./meta/contributing.md).
