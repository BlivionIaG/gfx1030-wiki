# llama.cpp or vLLM

Pick a stack before you follow a recipe. Both run on gfx1030; they are good at different jobs.
Numbers below are **community** snapshots — see [Verification status](./reference/verification.md).

## Which stack

| Workload | Prefer |
|---|---|
| Getting ROCm + multi-GPU working; GGUF; single-stream / low concurrency | **[llama.cpp](./llama-cpp/overview.md)** (RDNA2 fork) — more battle-tested on V620 |
| Multi-stream / agentic loads with prefix caching | **[vLLM `-extras`](./vllm/overview.md)** — caching + concurrency usually win |
| Qwen3.8 **Flash-Next** on 4× V620 (weights in VRAM) | **vLLM** Flash-Next / `rdna_extras` — see [Flash-Next](./vllm/flash-next.md). llama.cpp on the same job is typically much slower |
| Flash-Next on **2×** V620, or any host that **must RAM/CPU-offload** | **llama.cpp** (`--n-cpu-moe`, `-ot …=CPU`) is the non-HEAD path — community 2-card IQ4_XS ~**25 t/s** / **150–200** PP. HEAD vLLM after [`#146`](https://github.com/opengfx1030/vllm-rdna/pull/146) (28 Oct) has `serve_offload.sh` — [2× offload](./vllm/flash-next-serve.md#flash-next-2x-moe-offload). **Needs verify** on a second host. |
| MoE / lighter agentic (Qwen3.6 35B-A3B, Ornith-class) | **Either** — the sweet spot on these cards |
| Curious about [Strata](https://github.com/Niko1221/Strata) (HIP, not llama.cpp) | **Experimental.** `#general` Oct 7–8: 2× V620 prompt jumped after gfx103x PRs **#835 / #849** + `STRATA_HIP_PROMPT_F16=1`; 3-card hosts still could not match vLLM prefill and decode sagged under concurrency. `#strata` (25 Oct): Strata **0.1.49.2** signed **DeepSeek engine** feed has **no gfx1030** HIP package (HIP build lists gfx1200/gfx1201). `#vllm-rdna` (28 Oct): easier onboarding / dashboard than vLLM; more interesting on **fewer than 4** cards or when experts must spill. `#strata` (31 Oct–1 Nov): try **`--calibrate`** before comparing forks; **Avalon** is **RDNA3-only**. `#strata` (3 Nov): **2× V620** extra streams gave little/no total TG (two streams were **slower**); a second host also called concurrency a miss. `#strata` (5 Nov): `STRATA_HIP_PROMPT_F16=1` still the first knob (`--tune thorough` after that); one **4×** host saw **Q8 prefill faster than Q4**. `#strata` (7–9 Nov): Qwen3.8 Flash-Next **Q8** in the **VS Code Copilot** harness dropped tool calls; Strata **`tool_call_recovery`** is **on since 0.2.8** (off in 0.2.7 and earlier). Recovery kept a bad turn going; it did not stop the misses. One host had no failures with **pi.dev**. Prefer vLLM when you need multi-stream. Not a wiki default — [resources](./meta/resources.md) |
| DeepSeek-V4.1-Flash on **4×** V620 (HIP, not vLLM) | **[Cilla](https://github.com/crispyging/cilla-v620)** — experimental Maya/Strata **gfx1030** source fork. `#general` (9 Nov): public `release/cilla-v620`. README focus is DeepSeek-V4.1-Flash-1017; GitHub description also lists GLM-5.3-Flash / MiniMax-M3.1-Flash (**Needs verify**). Tested build: **ROCm 7.2.4**, Python **3.11**, HIP `maya-ds4`. Four-card profile is **opt-in** ([LAUNCHING](https://github.com/crispyging/cilla-v620/blob/release/cilla-v620/docs/LAUNCHING.md)); pinned Engram + `--mmap-experts off` wants **hundreds of GB** host RAM. **No tested 8× preset.** Community first-tries mixed. Not a wiki default — [resources](./meta/resources.md) |

New to the cards? Start with [RDNA2 serving](./llama-cpp/rdna2-serving.md), then [vLLM onboard](./vllm/onboarding.md) when you need concurrency. Multi-GPU tensor parallel benefits from [PCIe P2P](./tuning/p2p.md).

## What fits well on V620 {#what-fits-well-on-v620}

`#general` (Sep 2026) consensus — RDNA2 has **no matrix / tensor cores**, so **prefill** on dense
models is the weak spot (agentic "read a pile of files" workloads frustrate people even when decode
looks fine):

| Workload | Community take |
|---|---|
| **MoE** (e.g. Qwen3.6 **35B-A3B**, Ornith-class) | Sweet spot on 1–4× V620 |
| Dense **27B** | Usable; expect mediocre PP vs newer silicon |
| **Flash-Next** | Promising on **4×** V620 via [vLLM Flash-Next](./vllm/flash-next.md); not a 1-card path. `#vllm-rdna` (5 Nov): **2×** with only **~16 GB** host RAM still wants **27B**, not Flash-Next |
| **GLM-5.3-Flash** / DeepSeek-class large MoE | **Not 1–2 cards.** Public [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) is **320B / 18B active**. `#general` (25 Oct): 1× V620 + host-RAM offload is a poor fit (community Maya snapshot ~200 PP / ~13 TG — **Needs verify**). Community vLLM path is **8×** — [recipes](./vllm/recipes.md#glm-53-flash). `#general` (9 Nov): **4×** experimental HIP path is [Cilla](https://github.com/crispyging/cilla-v620) (Maya fork) — not vLLM, and not a 1-card offload fix. |
| Dense agentic on 1–2 cards | Often disappointing TTFT / PP — prefer MoE or more cards |
| ComfyUI image / video | Poor fit — many workflows default to **bf16** (emulated on V620). See [hardware](./setup/hardware.md) |

## Next

- [Getting started](./setup/getting-started.md)
- [vLLM recipes](./vllm/recipes.md) — Hub `-extras` vs recipe container
- [llama.cpp overview](./llama-cpp/overview.md)
