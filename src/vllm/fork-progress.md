# vLLM fork progress (RDNA2 / gfx1030)

> **WIP snapshot:** Readable status of
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras`.
> Dated **31 Oct 2026** from public merges + `#vllm-rdna` / `#announcements`.
> Kernel file lists and older chronicle rows stay on [Fork landscape](fork.md).
> See [Verification status](../reference/verification.md#vllm-fork-progressmd).

## How to read this

| Layer | What you get |
|---|---|
| **Hub `v0.31.0-extras`** (11 Oct 2026) | Day-to-day Docker — [Running](running.md), [Onboard](onboarding.md) |
| **`rdna_extras` HEAD** | Everything after that bake — clone + [host venv](host-venv.md) + [`serve_rdna.sh`](serve-scripts.md) |
| **Open drafts** | Backlog / WIP — not a recipe until merged |

Published Hub tags **lag HEAD**. If a row says “Not Hub”, do not expect it on
`blivioniag/vllm-rdna:v0.31.0-extras` until a newer extras bake.

## Shipped in Hub `v0.31.0-extras` (11 Oct)

`#announcements` (11 Oct 2026): `rdna_extras` = upstream **vLLM 0.31.0** + RDNA port.
Bake: ROCm **7.14**, torch **2.12**, Triton **3.7.1**.

| Area | In the bake |
|---|---|
| Quants / MoE | W4A16 AWQ/GPTQ dense+MoE HIP; opt-in W4A8 prefill; MXFP4 MoE; EXL3 (27B recipe) |
| Attention / GDN | FA-RDNA2 (split decode), GDN HIP decode, fused hyper-connection |
| Spec / graphs | MTP=2; FULL_AND_PIECEWISE; decode-stall cap; shipped TunableOp table |
| Flash-Next PLE | `PLE=bf16\|int4\|int4-disk` — all three tested on 4× V620 |
| Launcher | `serve_rdna.sh RECIPE=flashnext-mtp2\|flashnext-mtp0\|flashnext\|27b-awq\|27b-exl3\|dsv4-flash\|full` |

Author-host 4× V620 TP=4 (probes / PPL / prefix cache / concurrency, 0 GPU faults):
Flash-Next AWQ ~**2000** PP, **45** (MTP-0) / **69** (MTP-2) TG, ~**205** agg @ 8 users —
**Fork-source**. Details: [fork.md v0.31 row](fork.md#consolidation-status).

## On HEAD since the bake (13–28 Oct)

Prefer these over Hub when you need the feature. Serve lines:
[Flash-Next serve](flash-next-serve.md), [Serve scripts](serve-scripts.md).

### Flash-Next / serving

| Merged | What landed | Serve |
|---|---|---|
| [`#96`](https://github.com/opengfx1030/vllm-rdna/pull/96) / [`#95`](https://github.com/opengfx1030/vllm-rdna/pull/95) (13 Oct) | **PP3** recipes `flashnext-pp3` / `flashnext-pp3-mtp2`; row-chunk INT8 shadows | [PP3](flash-next-serve.md#flash-next-3x-pp3) |
| [`#117`](https://github.com/opengfx1030/vllm-rdna/pull/117) (17 Oct) | **EXL3 Flash-Next** — `RECIPE=flashnext-exl3`, `PLE=exl3`, turboderp pack | [Serve scripts](serve-scripts.md) |
| [`#94`](https://github.com/opengfx1030/vllm-rdna/pull/94) (16 Oct) | **MonoKernel** decode — `--monokernel` / `VLLM_ROCM_MONO_DECODE=1` (off by default) | [MonoKernel](flash-next-serve.md#flash-next-monokernel) |
| [`#126`](https://github.com/opengfx1030/vllm-rdna/pull/126) (18 Oct) | MonoKernel **QSA indexer in HIP** | same |
| [`#130`](https://github.com/opengfx1030/vllm-rdna/pull/130) (19 Oct) | **HIP set default-on** for Flash-Next recipes (closes `#99`) | [HIP defaults](flash-next-serve.md#flash-next-hip-defaults) |
| [`#132`](https://github.com/opengfx1030/vllm-rdna/pull/132) (21 Oct) | **DFlash2** recipe `flashnext-dflash2` + FA-RDNA2 `BLHNC` draft | [DFlash2](flash-next-serve.md#flash-next-dflash2) |
| [`#137`](https://github.com/opengfx1030/vllm-rdna/pull/137) (24 Oct) | MTP-2 **MonoKernel** fault containment / status reduce | opt-in `--monokernel` |

### Correctness / graphs / AR

| Merged | What landed |
|---|---|
| [`#127`](https://github.com/opengfx1030/vllm-rdna/pull/127) (18 Oct) | Dynamic MTP graph shapes — **opt-in**; leave schedule unset for day-to-day ([note](../troubleshooting/vllm-flash-next.md#flash-next-dynamic-mtp)) |
| [`#128`](https://github.com/opengfx1030/vllm-rdna/pull/128) (19 Oct) | `rdna_ar` sticky abort → WEDGED ([hang](../troubleshooting/vllm.md#rdna-ar-silent-hang)) |
| [`#131`](https://github.com/opengfx1030/vllm-rdna/pull/131) (19 Oct) | Non-mono CUDA-graph hang: no SharedExperts aux-stream fork on gfx10x ([hang](../troubleshooting/vllm.md#non-mono-graph-hang)) |
| [`#107`](https://github.com/opengfx1030/vllm-rdna/pull/107) / [`#108`](https://github.com/opengfx1030/vllm-rdna/pull/108) (16 Oct) | Flash-Next TP4 `GPU_MAX_HW_QUEUES=4`; `ROCPROFILER_REGISTER_ENABLED=0` |

### Memory / dense / KV

| Merged | What landed | Notes |
|---|---|---|
| [`#138`](https://github.com/opengfx1030/vllm-rdna/pull/138) (25 Oct) | **INT4 KV** dtype `int4_fp16_block` (FP16 per-32-channel block scales); native **QSA** read | Opt-in `--kv-cache-dtype int4_fp16_block`. Other backends refuse it. **Not** Hub. MonoKernel path still **Needs verify** for this dtype. |
| [`#140`](https://github.com/opengfx1030/vllm-rdna/pull/140) (26 Oct) | **fp16 SSM** state (smaller mamba pages); derived prefill budget; **`W4A8=1` default** on `27b-awq` | Author-host 27B: 16k c=1 PP **860 → 1075** tok/s; c=8 agg out **42.5 → 52.3** — **Fork-source**. Quality paired PPL ~unchanged. |
| [`#146`](https://github.com/opengfx1030/vllm-rdna/pull/146) (28 Oct) | **TP2 AWQ MoE expert offload** — pinned-RAM expert banks + bounded GPU slots (`serve_offload.sh`) | Author-host **2× V620** Flash-Next AWQ MTP-2 — [2× serve](flash-next-serve.md#flash-next-2x-moe-offload). **Not** Hub. Supersedes draft `#37`. |
| [`#98`](https://github.com/opengfx1030/vllm-rdna/pull/98) (14 Oct) | Asymmetric W4A16 MoE for compressed-tensors AWQ | [Quantization](quantization.md) |

### Tooling / cleanup (HEAD hygiene)

[`#97`](https://github.com/opengfx1030/vllm-rdna/pull/97) prefix-cache off flag;
[`#100`](https://github.com/opengfx1030/vllm-rdna/pull/100)–[`#106`](https://github.com/opengfx1030/vllm-rdna/pull/106) / [`#88`](https://github.com/opengfx1030/vllm-rdna/pull/88) dead-path cleanup;
[`#92`](https://github.com/opengfx1030/vllm-rdna/pull/92) TunableOp grid expand;
[`#141`](https://github.com/opengfx1030/vllm-rdna/pull/141) stale-doc / lint fixes.

## Open backlog (selected)

Do **not** treat drafts as recipes. Full list:
[open PRs on `rdna_extras`](https://github.com/opengfx1030/vllm-rdna/pulls?q=is%3Aopen+base%3Ardna_extras).

| Track | PR | Why it matters |
|---|---|---|
| INT8 KV + scales | Draft [`#71`](https://github.com/opengfx1030/vllm-rdna/pull/71) | `#vllm-rdna` (21 Oct): maintainer priority ahead of experimental INT4-KV ideas; restore native INT8 HIP path |
| Dynamic speculative drafting | Draft [`#125`](https://github.com/opengfx1030/vllm-rdna/pull/125) | Proper per-step K; builds on `#119`/`#127` |
| Vectorized `rdna_ar` | Open [`#129`](https://github.com/opengfx1030/vllm-rdna/pull/129) | Left out of `#130` |
| MonoKernel MTP drafter | Draft [`#109`](https://github.com/opengfx1030/vllm-rdna/pull/109) | Drafter still off-mono today |
| MonoKernel PP3 | Draft [`#116`](https://github.com/opengfx1030/vllm-rdna/pull/116) | Do not combine `--monokernel` with `flashnext-pp3*` yet |
| EXL3 Flash-Next MTP / vision / MoE cut | Drafts [`#122`](https://github.com/opengfx1030/vllm-rdna/pull/122) / [`#123`](https://github.com/opengfx1030/vllm-rdna/pull/123) / [`#124`](https://github.com/opengfx1030/vllm-rdna/pull/124) | `flashnext-exl3` is MTP=0 for now |
| DFlash2 BF16 GEMV opts | Draft [`#135`](https://github.com/opengfx1030/vllm-rdna/pull/135) | Follow-up to `#132` |
| W8A8 sdot4 dense/MoE | Draft [`#93`](https://github.com/opengfx1030/vllm-rdna/pull/93) | Explore — not day-to-day |
| Disagg prefill/decode | Open [`#23`](https://github.com/opengfx1030/vllm-rdna/pull/23) | Cross-PLX KV bridge — Needs verify |
| GLM-5.3 / Glm5Next | Drafts [`#51`](https://github.com/opengfx1030/vllm-rdna/pull/51) / [`#133`](https://github.com/opengfx1030/vllm-rdna/pull/133) | Correctness WIP |
| Tiered expert budgets + hybrid KV paging | Draft [`#148`](https://github.com/opengfx1030/vllm-rdna/pull/148) | `#vllm-rdna` (30 Oct): follow-up to merged `#146`. Live paging **stalls**; PR stays **draft** until mixed-workload gaps look acceptable. **Not** a recipe. |

`#vllm-rdna` (21 Oct): community DeepSeek-V4-Flash correctness write-up on a fork of
`rdna_extras` (bf16→fp16 cast, logits all-gather, EP routing). In-tree
`RECIPE=dsv4-flash` still **Needs verify** — [serve](flash-next-serve.md#deepseek-v4-flash).

## Practical defaults (31 Oct)

| Goal | Prefer |
|---|---|
| Docker / least friction | Hub **`v0.31.0-extras`** |
| 4× Flash-Next day-to-day on HEAD | `RECIPE=flashnext-mtp2` (+ HIP defaults from `#130`) |
| Faster Flash-Next decode experiment | `EXTRA_ARGS="--monokernel"` — A/B ITL; regenerate TunableOp if PP cliffs |
| 3× Flash-Next | `RECIPE=flashnext-pp3` or `flashnext-pp3-mtp2` (HEAD only) |
| Longer KV on QSA path | `--kv-cache-dtype int4_fp16_block` (HEAD; opt-in) |
| 2× Flash-Next (weights do not fit) | HEAD `tools/rdna/mono/serve_offload.sh` (`#146`) — **Needs verify** on a second host. llama.cpp RAM/MoE offload remains the non-HEAD path. |
| 27B AWQ prefill | `RECIPE=27b-awq` with HEAD `W4A8=1` (recipe default after `#140`) |
| Spec draft A/B | Native **MTP-2** first; `flashnext-dflash2` for short/single-user experiments |

## Related

- [Fork landscape + chronicle](fork.md) — repos, kernels, full consolidation table
- [Flash-Next](flash-next.md) / [Serve lines](flash-next-serve.md)
- [Serve scripts](serve-scripts.md)
- [Verification](../reference/verification.md)
