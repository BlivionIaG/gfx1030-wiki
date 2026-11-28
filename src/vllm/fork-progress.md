# vLLM fork progress (RDNA2 / gfx1030)

> **WIP snapshot:** Readable status of
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras`.
> Dated **28 Nov 2026** from public merges + `#vllm-rdna` / `#announcements` /
> `#benchmarks` / `#general` / `#strata` / `#bios`.
> Kernel file lists and older chronicle rows stay on [Fork landscape](fork.md).
> See [Verification status](../reference/verification.md#vllm-fork-progressmd).

## How to read this

| Layer | What you get |
|---|---|
| **Hub `v0.31.0-extras`** (11 Oct 2026) | Day-to-day Docker — [Running](running.md), [Onboard](onboarding.md) |
| **`rdna_extras` HEAD** | **v0.34.0** as of 28 Nov — clone + [host venv](host-venv.md) + [`serve_rdna.sh`](serve-scripts.md) |
| **Org GHCR `nightly`** | Source CI snapshot of HEAD — [Running](running.md#ghcr-nightly). Not a Hub bake. |
| **Open drafts** | Backlog / WIP — not a recipe until merged |

Published Hub tags **lag HEAD**. If a row says “Not Hub”, do not expect it on
`blivioniag/vllm-rdna:v0.31.0-extras` until a newer extras bake. Org GHCR
`nightly` tracks `rdna_extras` tip (anonymous pull **Needs verify**; this run still 401).

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

## On HEAD since 1–2 Nov (0.33 rebase)

`#vllm-rdna` (1 Nov): `rdna_extras` rebased **0.32.0** then **0.33.0** (Qwen4
readiness). No new Hub extras tag. Author: prefill looked better on 0.32 than
0.33 — **Community**.

| Landed | What | Notes |
|---|---|---|
| CI [`#151`](https://github.com/opengfx1030/vllm-rdna/pull/151) / [`#154`](https://github.com/opengfx1030/vllm-rdna/pull/154) | Source GHCR `ghcr.io/opengfx1030/vllm-rdna:nightly` (+ `:nightly-rocm7.14.0`, `:sha-<commit>`) | `--platform linux/amd64`. Day-to-day stays Hub **0.31**. |
| 2 Nov PLE graph fix | Breakable `FULL_AND_PIECEWISE` froze PLE layout → NaN / `!!!!` / `_ple_conv_kernel` | Fixed on HEAD: PLE breaks **PIECEWISE** only. Nightly from **before** that commit is unsafe. [Troubleshooting](../troubleshooting/vllm.md#033-nightly-ple-cuda-graphs) |
| `#vllm-rdna` 4–5 Nov (Community) | 4× Flash-Next HEAD A/B: one host 1k/512 c=1 decode **122 → ~150** t/s (`MAXBAT=2048`, INT4 PLE in RAM, MTP-2, W4A8 prefill). Author expected **~90–100**. Mixed **RDNA2+RDNA3** is **not** a wiki path. | **Needs verify**. Not a recipe. |
| [`#168`](https://github.com/opengfx1030/vllm-rdna/pull/168) (14 Nov) | **LeapQuant** INT8 GDN state — `RECIPE=flashnext-leapquant-mtp2` | Opt-in, lossy, gfx10x HIP. Not a day-to-day replacement for `flashnext-mtp2`. **Not** Hub. [LeapQuant](flash-next-serve.md#flash-next-leapquant) |
| [`#170`](https://github.com/opengfx1030/vllm-rdna/pull/170) (15 Nov) | LeapQuant **config flag** `--replayssm-state-dtype leapquant` (replaces `VLLM_RDNA_GDN_LEAPQUANT`; no alias) | Packing `VLLM_RDNA_GDN_LEAPQUANT_PACK=1` is a **no-op** at block 512. Same KV as `#168`. **Not** Hub. |
| [`#171`](https://github.com/opengfx1030/vllm-rdna/pull/171) (15 Nov) | **HI2** 27B TP=2 recipes `27b-hi2-tp2` / `27b-hi2-tp2-mtp2` | 2× V620 only. Switches off in code. **Not** Hub. [Quantization](quantization.md) |
| [`#172`](https://github.com/opengfx1030/vllm-rdna/pull/172) (16 Nov) | `gdn_attn` import after `#170`+`#171` | Without this, **no GDN model boots**. Pull HEAD / GHCR nightly **after** this merge. |
| [`#173`](https://github.com/opengfx1030/vllm-rdna/pull/173) (16 Nov) | TunableOp **rank files agree per key** | Data-only. Pre-fix, 531 keys disagreed across ranks (1023-row router GEMM). `rocm10-rocblas5.6` already agreed. [Rank agreement](../troubleshooting/vllm.md#tunableop-rank-agreement). |
| [`#175`](https://github.com/opengfx1030/vllm-rdna/pull/175) / [`#176`](https://github.com/opengfx1030/vllm-rdna/pull/176) (16 Nov) | Lean **EPLB** study + opt-in load counters | Decode EPLB / placement **no-go** at TP4/EP4 (ITL gain **0.02–0.19%**). Counters: `VLLM_RDNA_EPLB_STATS_DIR` (unset = off). Still skip upstream `--enable-eplb`. |
| [`#169`](https://github.com/opengfx1030/vllm-rdna/pull/169) / [`#174`](https://github.com/opengfx1030/vllm-rdna/pull/174) / [`#177`](https://github.com/opengfx1030/vllm-rdna/pull/177) / [`#189`](https://github.com/opengfx1030/vllm-rdna/pull/189) (16 Nov) | Flash-Next **GGUF** `RECIPE=flashnext-gguf-mtp0` | `--load-format qwen4exp_gguf`, UD-Q4_K_XL, MTP=0. Needs [`opengfx1030/vllm-gguf-plugin`](https://github.com/opengfx1030/vllm-gguf-plugin) (image after `#189`). **Not** Hub. [GGUF](flash-next-serve.md#flash-next-gguf). |
| [`#195`](https://github.com/opengfx1030/vllm-rdna/pull/195) (22 Nov) | **Inkling-2-Mini** native HIP — `RECIPE=inkling2-mini-tp{1,2,4}` | NVFP4/MXFP8 in-GEMM dequant, vision, prefix cache, 1M ctx at TP=2/4. **Not** Hub. [Inkling](inkling.md). |
| [`#196`](https://github.com/opengfx1030/vllm-rdna/pull/196) (23 Nov) | Inkling pipelined prefill attention tile + DSpark recipes | First DSpark files. Folded by `#208`. **Not** Hub. |

## On HEAD since 24–25 Nov (0.34 README + Inkling follow-ups)

Public `rdna_extras` README (25 Nov) names upstream **vLLM v0.34.0**. Hub bake
is still **0.31**. `#vllm-rdna` / `#general` (21 Nov) “0.34” talk is now
**Fork-source**, not unconfirmed.

| Landed | What | Notes |
|---|---|---|
| [`#198`](https://github.com/opengfx1030/vllm-rdna/pull/198) (24 Nov) | GHCR tag scheme | Prefer `:nightly` / `:nightly-0.34.0-rocm7.14.0`. Per-commit `:sha-*` is disposable. Git tag **`:v0.34.0`** is 19 Nov — **pre-Inkling**. [Running](running.md#ghcr-nightly) |
| [`#205`](https://github.com/opengfx1030/vllm-rdna/pull/205) (24 Nov) | TP=2/4 NVFP4 prefill K-tails | Author-host 1k TP2 **~1698 → ~2334** tok/s; TP4 **~3112 → ~3884**. Same-morning `#general`. **Not** Hub. |
| [`#207`](https://github.com/opengfx1030/vllm-rdna/pull/207) (24 Nov) | DSpark TP=1/2/4 | TP1 1k c=1 **~2085** PP / **~94** TG; TP2 **~3044** / **~136**; TP4 **~3867** / **~164**. **Not** Hub. |
| [`#208`](https://github.com/opengfx1030/vllm-rdna/pull/208) (24 Nov) | Recipe fold | Five files: `inkling2-mini-tp1`, `tp1-dspark`, `tp1-offload` (`#211`), `tp2`, `tp4`. Removed `*-dspark-adaptive` / `tp2-dspark` / `tp4-dspark`. DSpark **on by default** at TP=2/4 (`DSPARK=0` for plain). |
| [`#209`](https://github.com/opengfx1030/vllm-rdna/pull/209) (24 Nov) | TP=1 decode / KV | `MAXLEN=194000` (was 180k). DSpark K=2 **~104** tok/s; adaptive + 32k vocab **~172** tok/s. **Not** Hub. Do not copy PCI BDFs from the PR body. |
| [`#210`](https://github.com/opengfx1030/vllm-rdna/pull/210) (25 Nov) | Opt-in LoopCD (`arXiv:2610.02185`) | Default `LOOPCD_ALPHA=0` (off). Hidden `alpha=0.35` near-free; logits `0.25` adaptive **75% → 90%** on a 20-problem Math+Code set (~10% decode). **Fork-source**, not wiki-tested. |
| [`#211`](https://github.com/opengfx1030/vllm-rdna/pull/211) (25 Nov) | `inkling2-mini-tp1-offload` | TP=1 expert spill (`EXPERT_SLOTS=48`), `MAXLEN=65536`. Simulated 16 GiB peak **12.94 GiB**. **TP>1 refused.** Native copy slower than resident TP=1. |
| [`#213`](https://github.com/opengfx1030/vllm-rdna/pull/213) (25 Nov) | Inkling fused glue + DSpark K≤3 bucket + TP=2 KV | TP=2 DSpark `MAXLEN=760000` (was 720k); `ADAPTIVE=1` → `720000`. **Not** Hub. |
| [`#214`](https://github.com/opengfx1030/vllm-rdna/pull/214) (25 Nov) | Offload prefetch | `EXPERT_CACHE_COPY=native` and `EXPERT_CACHE_PREFETCH=1` are recipe **defaults**. **Not** Hub. |
| [`#215`](https://github.com/opengfx1030/vllm-rdna/pull/215) (25 Nov) | RDNA KV-margin floor | Engine default **1 GiB** on gfx10x (CDNA stays 3 GiB). Author-host TP=1 Inkling KV **7.08 → 9.08** GiB; decode **~55.3** unchanged. **Not** Hub. |
| [`#216`](https://github.com/opengfx1030/vllm-rdna/pull/216) (25 Nov) | LoopCD idle skip | `VLLM_INKLING2_LOOPCD_ENABLE` default **1**; set **0** to strip graph-baked LoopCD launches. **Not** Hub. |
| [`#217`](https://github.com/opengfx1030/vllm-rdna/pull/217) (25 Nov) | `27b-awq` RDNA2 switches + HI2 MoE | Recipe now sets unfused FA prefill, fused add-norm, in-place GDN, LM-head GEMV. Author-host **2×** 27B AWQ c=8 **147 → 190** tok/s. **Not** Hub. `RDNA_AR=1` with `VLLM_RDNA_AR_MAX_KB=256` and `VLLM_RDNA_AR_ONESHOT_KB=256`. |

## On HEAD since 26–28 Nov (kernels + PP3 + INT4 KV + decode)

`#vllm-rdna` (26 Nov, ~22:00 UTC) asked for lossy kernels to still report how much
token loss there is, including W8A8 prefill where token-exact is the wrong gate.
Merged [`#224`](https://github.com/opengfx1030/vllm-rdna/pull/224) (26 Nov 22:16 UTC)
puts that in the contributing contract. Git tip after
[`#227`](https://github.com/opengfx1030/vllm-rdna/pull/227) (27 Nov 15:59 UTC) is
`53f15741`. ROCm **7.14.0**, torch **2.12.0+rocm7.14.0**, Triton **3.7.1** remain
the wiki serve pin. gfx1030 stays off torch **2.15+** and Triton **3.9+**.
The `#227` author-host receipt used **ROCm 10.0** / torch **2.13** — do not treat
that as a new pin. `RECIPE=27b-awq` still sets `RDNA_AR=1` with both AR caps at
**256** KiB.

| Landed | What | Notes |
|---|---|---|
| [`#222`](https://github.com/opengfx1030/vllm-rdna/pull/222) (27 Nov 00:15 UTC) | Wave-7 BAR1 P2P `rdna_ar`, `moe_mxfp8` `BK=192`, opt-in Inkling MoE offload | Default numerics unchanged. Offload flags default **0**. Draft [`#223`](https://github.com/opengfx1030/vllm-rdna/pull/223) holds the monokernel split. **Not** Hub. |
| [`#221`](https://github.com/opengfx1030/vllm-rdna/pull/221) (27 Nov 05:41 UTC) | HIP W8A8 dense/HC prefill + QSA pipe KV fix + grouped MTP MoE prefill | Flags default **0**. QSA `>4.29 GiB` path is exact. Author-host 4× 16k PP **2552 → 3627** tok/s — **Fork-source**, taken before `#222` was in the same tree. **Not** Hub. |
| [`#228`](https://github.com/opengfx1030/vllm-rdna/pull/228) (27 Nov 06:53 UTC) | `RECIPE=flashnext-w8a8-mtp2` + long-context / passkey receipt cells | Sources `flashnext-mtp2` and turns the three W8A8 flags **on**. Default `flashnext-mtp2` stays off. Tools/recipes only. **Not** Hub. |
| [`#225`](https://github.com/opengfx1030/vllm-rdna/pull/225) (27 Nov 06:59 UTC) | Restore PP=3 Qwen4Exp MTP after v0.34 `#60234` | `flashnext-pp3-mtp2` on `#217` (`93c59de2`) does **not** boot. Pull HEAD **after this**. [PP3 0.34](../troubleshooting/vllm-flash-next.md#flash-next-pp3-mtp-034). |
| [`#229`](https://github.com/opengfx1030/vllm-rdna/pull/229) (27 Nov 07:45 UTC) | Enforce ≥64k cells + passkey needle on Ready runtime PRs | CI/docs only (`check_pr_contract.py`). |
| [`#226`](https://github.com/opengfx1030/vllm-rdna/pull/226) (27 Nov 09:48 UTC) | Accept QSA `BLNHC` / `slot_outer` layout for `int4_fp16_block` | Unlocks INT4 KV on Flash-Next QSA. Author-host 4× `flashnext-mtp0`: **297,756 → 864,517** tokens (**2.90×**), PPL **+0.396%**, passkey **3/3** at 32k/64k/128k. Concurrent `MAXBAT=2048` serializes (INT4 `block_size=2048`); use **`MAXBAT≥2560`**. Default FP16 unchanged. **Not** Hub. [INT4 KV](quantization.md#int4-kv-cache-int4_fp16_block-head). |
| [`#227`](https://github.com/opengfx1030/vllm-rdna/pull/227) (27 Nov 15:59 UTC) | HIP decode kernels: INT8 GEMV/SE/HC `M=9..32`, Wave32 QSA sort, fp32-scale MoE, GDN vec | Default **on** (`VLLM_RDNA_DECODE_I8_MAX=32`, `VLLM_RDNA_HC_I8_MAX=16`; set both to **8** to roll back). Author-host 4× Flash-Next MTP-2: c=4 decode **38.1 → 62.2** tok/s/req (**+63%**); c=8 **+31%**. Receipt is **Flash-Next W4A16 TP4**, not Inkling (`#vllm-rdna` 27 Nov). **Not** Hub. |

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
| W8A8 sdot4 dense/MoE | Draft [`#93`](https://github.com/opengfx1030/vllm-rdna/pull/93) | Explore — not day-to-day. Merged `#221` is the HIP W8A8 **prefill** path (opt-in recipe `flashnext-w8a8-mtp2`) |
| Inkling decode MoE monokernel | Draft [`#223`](https://github.com/opengfx1030/vllm-rdna/pull/223) | Split out of `#222`. **Not** a recipe. |
| Disagg prefill/decode | Open [`#23`](https://github.com/opengfx1030/vllm-rdna/pull/23) | Cross-PLX KV bridge — Needs verify |
| GLM-5.3 / Glm5Next | Drafts [`#51`](https://github.com/opengfx1030/vllm-rdna/pull/51) / [`#133`](https://github.com/opengfx1030/vllm-rdna/pull/133) | Correctness WIP |
| Tiered expert budgets + hybrid KV paging | Draft [`#148`](https://github.com/opengfx1030/vllm-rdna/pull/148) | `#vllm-rdna` (30 Oct): follow-up to merged `#146`. Live paging **stalls**; PR stays **draft** until mixed-workload gaps look acceptable. **Not** a recipe. |
| RidgeFill late-layer cache fill | Open [`#155`](https://github.com/opengfx1030/vllm-rdna/pull/155) | `#vllm-rdna` (4–6 Nov) + PR verdict **6 Nov**: opt-in `VLLM_RDNA_RIDGEFILL` **defaults off**. Fill path is exact; **no map set met the quality gate**. **Not** a recipe. |
| Native oQ6e kernels | Draft [`#178`](https://github.com/opengfx1030/vllm-rdna/pull/178) | `#vllm-rdna` (16 Nov): still **not** a serve path. Public HF packs exist; HIP recipe does not. [Quantization](quantization.md). |

`#vllm-rdna` (21 Oct): community DeepSeek-V4-Flash correctness write-up on a fork of
`rdna_extras` (bf16→fp16 cast, logits all-gather, EP routing). In-tree
`RECIPE=dsv4-flash` still **Needs verify** — [serve](flash-next-serve.md#deepseek-v4-flash).

## Practical defaults (28 Nov)

| Goal | Prefer |
|---|---|
| Docker / least friction | Hub **`v0.31.0-extras`** (still the published bake) |
| HEAD / 0.34 kernels in Docker | Org GHCR **`nightly`** / `:nightly-0.34.0-rocm7.14.0` — pull a digest **after `#227`** (27 Nov HIP decode + `#226` INT4 KV + `#225` PP3 MTP), or clone tip `53f15741`. Do **not** use `:v0.34.0` for Inkling (19 Nov). Anonymous GHCR pull **Needs verify**. |
| 4× Flash-Next day-to-day on HEAD | `RECIPE=flashnext-mtp2` (+ HIP defaults from `#130`; `#227` decode kernels are **on**). W8A8 prefill stays **off** unless you pick `flashnext-w8a8-mtp2`. `#vllm-rdna` (27 Nov): one 4× host **>3k** tok/s PP and **100–170** tok/s TG with MTP; two agents **~80** tok/s with [`#36`](https://github.com/opengfx1030/vllm-rdna/pull/36) `VLLM_RDNA_DYNAMIC_PREFILL=1`. **Community**. Leapdragon-era images still sit around **~1900** PP / **~68** TG. |
| Faster Flash-Next decode experiment | HEAD `#227` is already the default HIP decode path. `EXTRA_ARGS="--monokernel"` remains an ITL A/B. Set `VLLM_RDNA_DECODE_I8_MAX=8` and `VLLM_RDNA_HC_I8_MAX=8` to roll `#227` back. |
| 3× Flash-Next | `RECIPE=flashnext-pp3` or `flashnext-pp3-mtp2` — HEAD **after `#225`**. `#217` / `93c59de2` MTP2 does **not** boot. Prefer `nightly-rocm7.14.0` after that merge. |
| Opt-in W8A8 Flash-Next prefill | `RECIPE=flashnext-w8a8-mtp2` (`#228`, after `#221`). Lossy. `#vllm-rdna` (27 Nov): W8A8 vs W4A16 **felt similar** (no bench). **Needs verify**. **Not** Hub. |
| Flash-Next GGUF (HEAD) | `RECIPE=flashnext-gguf-mtp0` + `GGUF_DIR=` (plugin fork, not PyPI). **Not** Hub. |
| Longer KV on QSA path | HEAD **after `#226`**: `--kv-cache-dtype int4_fp16_block`. Raise `MAXBAT` to **≥2560** for concurrent INT4 KV. Default FP16 is still the day-to-day dtype. |
| 2× Flash-Next (weights do not fit) | HEAD `tools/rdna/mono/serve_offload.sh` (`#146`) — **Needs verify** on a second host. llama.cpp RAM/MoE offload remains the non-HEAD path. |
| 27B AWQ prefill | `RECIPE=27b-awq` with HEAD `W4A8=1` (recipe default after `#140`) plus `#217` RDNA2 switches (unfused FA prefill / in-place GDN). **Not** Hub. |
| Spec draft A/B | Native **MTP-2** first; `flashnext-dflash2` for short/single-user experiments |
| LeapQuant GDN experiment (HEAD) | `RECIPE=flashnext-leapquant-mtp2` (`#168`/`#170`, after `#172`) — opt-in, lossy; 1k c=1 is slower. Stay on `flashnext-mtp2` unless you are A/B'ing. Do not set `VLLM_RDNA_GDN_LEAPQUANT`. |
| 2× 27B HI2 (HEAD) | `RECIPE=27b-hi2-tp2` (or `-mtp2` for c=1). **Not** Hub. |
| Inkling-2-Mini (HEAD, 0.34) | `RECIPE=inkling2-mini-tp1` (1× 32 GB) / `-tp1-dspark` / `-tp1-offload` (16 GB) / `-tp2` / `-tp4`. TP=2/4 DSpark **on by default**. **Not** Hub. |

## Related

- [Fork landscape + chronicle](fork.md) — repos, kernels, full consolidation table
- [Flash-Next](flash-next.md) / [Serve lines](flash-next-serve.md)
- [Serve scripts](serve-scripts.md)
- [Verification](../reference/verification.md)
