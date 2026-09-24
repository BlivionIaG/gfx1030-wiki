# llama.cpp or vLLM

Pick a stack before you follow a recipe. Both run on gfx1030; they are good at different jobs.
Numbers below are **community** snapshots — see [Verification status](./reference/verification.md).

## Which stack

| Workload | Prefer |
|---|---|
| Getting ROCm + multi-GPU working; GGUF; single-stream / low concurrency | **[llama.cpp](./llama-cpp/overview.md)** (RDNA2 fork) — more battle-tested on V620 |
| Multi-stream / agentic loads with prefix caching | **[vLLM `-extras`](./vllm/overview.md)** — caching + concurrency usually win |
| Qwen3.8 **Flash-Next** on 4× V620 (weights in VRAM) | **vLLM** Flash-Next / `rdna_extras` — see [Flash-Next](./vllm/flash-next.md). llama.cpp on the same job is typically much slower |
| Flash-Next on **2×** V620, or any host that **must RAM/CPU-offload** | **llama.cpp** (`--n-cpu-moe`, `-ot …=CPU`). Community 2-card IQ4_XS ~**25 t/s** / **150–200** PP — [recipe](./llama-cpp/rdna2-speculative.md#flash-next-2x-iq4) |
| MoE / lighter agentic (Qwen3.6 35B-A3B, Ornith-class) | **Either** — the sweet spot on these cards |

New to the cards? Start with [RDNA2 serving](./llama-cpp/rdna2-serving.md), then [vLLM](./vllm/overview.md) when you need concurrency. Multi-GPU tensor parallel benefits from [PCIe P2P](./tuning/p2p.md).

## What fits well on V620 {#what-fits-well-on-v620}

`#general` (Sep 2026) consensus — RDNA2 has **no matrix / tensor cores**, so **prefill** on dense
models is the weak spot (agentic "read a pile of files" workloads frustrate people even when decode
looks fine):

| Workload | Community take |
|---|---|
| **MoE** (e.g. Qwen3.6 **35B-A3B**, Ornith-class) | Sweet spot on 1–4× V620 |
| Dense **27B** | Usable; expect mediocre PP vs newer silicon |
| **Flash-Next** | Promising on **4×** V620 via [vLLM Flash-Next](./vllm/flash-next.md); not a 1-card path |
| Dense agentic on 1–2 cards | Often disappointing TTFT / PP — prefer MoE or more cards |

## Next

- [Getting started](./setup/getting-started.md)
- [vLLM recipes](./vllm/recipes.md) — Hub `-extras` vs recipe container
- [llama.cpp overview](./llama-cpp/overview.md)
