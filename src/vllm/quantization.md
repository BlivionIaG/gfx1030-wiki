# vLLM Quantization on gfx1030

> **WIP:** Throughput numbers are **community-reported**. See
> [Verification status](../reference/verification.md#vllm-quantizationmd).

## GPTQ vs AWQ

| Format | `-extras` kernel path | Notes |
|---|---|---|
| **GPTQ** (e.g. `btbtyler09/Qwen3.8-27B-GPTQ-4bit`) | `RDNA2W4A16LinearKernel` — native gfx1030 HIP | Best `-extras` throughput. Force with `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel`. |
| **AWQ** (e.g. `Qwen3.8-27B-AWQ-INT4`) | `RDNA2W4A16LinearKernel` on gfx10x | As of Aug 2026 extras, AWQ dense routes through the same native W4A16 kernel as GPTQ (fork-author reported ~151 output t/s; **needs verify** on your image). |
| **compressed-tensors** (e.g. `cyankiwi/Qwen3.8-27B-AWQ-INT4`) | Mixed — use `--quantization compressed-tensors` | Custom int4 re-quants; benchmark against GPTQ/AWQ. |
| **AWQ-vd** (e.g. `ikantkode/Qwen3.8-27B-AWQ-vd`) | `RDNA2W4A16LinearKernel` when dense | Community-tuned AWQ variant; confirm kernel in logs. |

On **older images** (before the AWQ→RDNA2 dispatch fix), AWQ fell through to Triton/Exllama and could
stall at ~4–5 t/s on a 27B. Pull the latest `-extras` image and confirm
`Using RDNA2W4A16LinearKernel` in startup logs. Qwen3.8-27B AWQ also needs the fork's
**`head_size=256`** FlashAttention path — without it, FA falls back or never lists `RDNA_ATTN`.

Kernel dispatch details: [rdna_extras fork](fork.md#kernel-dispatch-on-gfx1030).

`#vllm-rdna` (14 Oct) + merged [`#98`](https://github.com/opengfx1030/vllm-rdna/pull/98): RDNA2 W4A16
**MoE** no longer synthesizes **symmetric** zero points. Asymmetric compressed-tensors checkpoints
(e.g. `cyankiwi/*-AWQ-INT4`, group 32, int8 zeros) used to miss the native experts kernel and fall
back to generic Triton WNA16 MoE. Asymmetric **8-bit** checkpoints still fall back. Confirm
`RDNA2W4A16MoEExperts` (or dense `RDNA2W4A16LinearKernel`) in logs. **Not** in Hub
`v0.31.0-extras` (11 Oct bake) — clone HEAD.

## KV-cache dtype

| Dtype | When people use it | `#vllm-rdna` notes |
|---|---|---|
| **`float16`** | Long-context / agents / tool calling | Default recommendation. `VLLM_USE_FA_RDNA2=1` currently needs fp16 KV. **Required** on current QSA / Flash-Next backends (see below). |
| **`int8_per_token_head`** | Throughput on older GPTQ / Triton FA | Reported **5–10 t/s above fp8** in TG (and higher PP) in limited testing. One report that it misbehaves with chunked prefill. **Not selected** on QSA Flash-Next — those backends do not list int8 KV. |
| **`fp8`** | VRAM savings (older dense 27B) | `#vllm-rdna` Sep 22: **blocked** on current QSA / Flash-Next. Two independent gates — [troubleshooting](../troubleshooting/vllm-flash-next.md#fp8-kv-rejected-on-qsa). Older dense-27B reports that fp8 was slower than `int8_per_token_head` and dropped quality on long sessions. |
| **KVarN** | Third-party KV compression | Raised concurrency on Qwen, **broke tool calling**, failed on Gemma 4. Community verdict: skip for agents. |

Prefer **`float16`**. Do not plan on `--kv-cache-dtype fp8` for QSA / Flash-Next until the extras fork
re-enables a gfx1030 path. `#vllm-rdna` (Sep 23–24): community is **testing INT8 KV** via upstream
[quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
(RDNA2 has no FP8) aiming at more concurrent long context (e.g. **4× 256k**). That is **not** the
same as INT8 **decode shadows**, and QSA backends still **do not list** int8 KV — treat as
**Needs verify**, not a recipe.

## MTP speculative decoding

MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) can boost throughput on GPTQ
models with CUDA graphs enabled. Acceptance rates dropped after a v0.27.1 speculator update (~0.25), but
base decode speed remains good — worth testing on your model. Example in
[Configuration](configuration.md#docker-compose-example-gptq--mtp--cuda-graphs).

MTP is **not free at high concurrency**. A `#vllm-rdna` TP4 matrix on **Qwen3.6-35B-A3B-FP16**
(4× V620, `--enforce-eager`, 16k/1k-style bench) reported MTP-2 **+17%** output tok/s at `c=1`, but
**−53%** at `c=8`. Use MTP for latency-critical single-stream; leave it off for batched throughput.

`#vllm-rdna` (Sep 15): `RDNA2W4A16MoEExperts` was reported to raise **usable concurrency** on MoE
W4A16 — confirm the kernel in logs. **Pipeline-parallel MTP** needs upstream
[`vllm#46994`](https://github.com/vllm-project/vllm/pull/46994) (merged; next vLLM release). A 3×
V620 Flash-Next + MTP squeeze is the in-tree `flashnext-pp3-mtp2` recipe on HEAD
([PP3](flash-next-serve.md#flash-next-3x-pp3)); the older leapdragon overlay is
[historical](flash-next-serve.md#flash-next-3x-pp3-mtp). Hub **`v0.31.0-extras`** lags.

`#vllm-rdna` (Sep 18): on Flash-Next, **MTP can block the int4 PLE fused decode path**. If logs show
Triton GDN / PLE fallback and decode is stuck in the teens of t/s, A/B **MTP off** before blaming
P2P. Separately, the MTP **draft head** may still be **bf16 MoE** even when the main experts are
W4A16 — quantize those draft experts offline if you stay on MTP
([3× overlay](flash-next-serve.md#flash-next-3x-moe-hip-overlay)). `#vllm-rdna` Sep 18: **no MTP on the
Hub `v0.28.0-extras` line**; that work is aimed at the `rdna_extra/v0.29.0` rebase.

`rdna_extras` HEAD
[`700753d9`](https://github.com/opengfx1030/vllm-rdna/commit/700753d9add5c4a5586126d561f23c6099e71609)
(25 Sep) **unbreaks MTP draft inference** on gfx1030: `amdsmi_shut_down()` exceptions no longer
mask a successful MoE config lookup (that used to kill the drafter on ranks 1–3 and deadlock
NCCL), and Qwen4Exp MTP uses **local-argmax** draft sampling when
`speculative_config.use_local_argmax_reduction=true`. Author-host 4× V620 W4A16 Flash-Next
**MTP-2** then **boots** (FA + `FULL_DECODE_ONLY` **~46 t/s**; Triton + `FULL_AND_PIECEWISE`
**~49 t/s**) but stays **below MTP-0 (~61 t/s)** — documented **~0.73–0.85×** penalty. `#vllm-rdna`
(Sep 25): community still reports the same “boots, not faster than MTP-0” class.

[`30632b2`](https://github.com/opengfx1030/vllm-rdna/commit/30632b2fa3239ea9057afd39595b84b540359d69)
(27 Sep) captures the **draft** decode graph at `max_num_reqs` (MTP-2 **c=8** was eager).
Author-host: **+18%** at 8×1k / **+5%** at 8×16k; still **~0.87× vs MTP-0** at 16k. One later
host with a matching TunableOp table reported MTP **above** MTP-0 — treat that as **Needs
verify**, not a retraction of the 0.73–0.87× class. Present in Hub **`v0.31.0-extras`**;
commits after the bake pin still want clone + [host venv](host-venv.md).

Merged [`#26`](https://github.com/opengfx1030/vllm-rdna/pull/26): concurrent MTP-2 (9–16 rows)
can stay on HIP MoE with `VLLM_ROCM_MOE_SKINNY_MAX_M=16`. Default remains **8**. See
[prefill vs decode](../troubleshooting/vllm.md#prefill-blocks-decode--mtp-stalls-under-concurrency).

`#vllm-rdna` (Oct 8): on current `rdna_extras` MTP-2, if **temperature > 0**, set
`"draft_sample_method":"probabilistic"` inside `--speculative-config`. The greedy / local-argmax
draft path (`700753d9`) is the default; non-zero temp without the probabilistic sampler is the
community-reported miss. Example:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}'
```

Same thread: day-to-day `rdna_extras` MTP-2 was **~60–90 t/s** on one host (community; not a
recipe). Hub **`v0.31.0-extras`** includes the MTP=2 path; A/B acceptance on your host.

## INT4 on gfx1030 (no native int4 ALUs)

RDNA2 has no hardware int4 matrix units. The `-extras` W4A16 kernels use **vdot2 on fp16 with on-the-fly
dequant** — int4 weights packed and processed via `dp4a`-style instructions. Both GPTQ and AWQ dense now
hit the same native HIP kernel on current `-extras` images.

Recent fork work on hybrid GDN models (Qwen3.8-27B-AWQ-INT4, TP4) **reported** **~93 output tok/s** with CUDA
graphs (1024/512), **~331 total tok/s** at 8 concurrent requests (16k/512), and prefill peaks of
**1450–1573 tok/s** — with the full HIP GDN prefill + decode chain replacing Triton JIT.

## Tips

- Lower `--gpu-memory-utilization` (e.g. `0.9` → `0.8`) if KV-cache allocation OOMs on 16 GB cards.
- For GPTQ or AWQ on `-extras`: set `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel` and
  watch logs for `RDNA2W4A16LinearKernel`.
- On older images, AWQ could fall through to Triton (~4–5 t/s on a 27B). Pull latest `-extras` and confirm
  the native kernel is active before blaming the quant format.
- Don't force `--attention-backend` or `--quantization` — let vLLM auto-select unless A/B testing.
- Mount Triton and torch-compile caches (see [Configuration](configuration.md#cache-volumes-first-boot-is-slow)).

## Experimental: EXL3 and Quark (`#vllm-rdna`, Sep–Oct 2026)

`#announcements` (11 Oct 2026): **EXL3** is listed among the **v0.31.0** extras features, and
the launcher ships `RECIPE=27b-exl3`. Older Hub tags (`v0.27.1-extras*`, `v0.28.0-extras*`) do
**not** have that bake. Treat EXL3 as **experimental** even on 0.31 — mul1-tested path; A/B
before production. **AMD Quark** is still not a drop-in on current `-extras`.

| Format | Status | Notes |
|---|---|---|
| **EXL3** (e.g. community 9B 3bpw Ornith builds) | **Experimental** | Single-card serve recipes with CUDA graphs (`FULL_AND_PIECEWISE`, capture sizes `1,2,4,8`) were shared in `#vllm-rdna`. Goal is fitting small models on **16 GB** consumer cards; Triton leftovers can still bloat VRAM. Sep 19 kernel was [3inst only](#exl3-3inst-only-vllm-rdna-sep-19); `#vllm-rdna` Sep 24: **`rdna_extra/v0.29.0` now has mul1**. `#vllm-rdna` Oct 2: **HEAD is mul1-tested only**; 3inst is untested. Not a drop-in on every Hub tag. |
| **AMD Quark** (e.g. [`amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)) | **Needs verify** | Marketed near MXFP4 quality; needs Quark-capable runtime (upstream PRs `#48606` / `#46110`). Community hit import issues — not a drop-in on current `-extras`. |

### EXL3: 3inst only (`#vllm-rdna`, Sep 19) {#exl3-3inst-only-vllm-rdna-sep-19}

`#general` (Sep 18) and `#vllm-rdna` (Sep 19): the gfx1030 EXL3 kernel on **`rdna_extras` HEAD** was
**3inst**, not **mul1**. 3inst is easier to execute at inference, **not** higher quality. Mixed
bit-widths are still not wired: MTP at 6 bpw, `lm_head` at 8 bpw, and vision in bf16 will not load
together. On **published Hub `-extras`**, keep **uniform 3 bpw** and the **head in bf16**.

`#vllm-rdna` (Sep 24): [`rdna_extra/v0.29.0`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extra/v0.29.0)
now implements **mul1**, so **existing EXL3 packs can load**. HIP EXL3 work is still in testing.
**Needs verify** — do not assume Hub tags have this.

`#vllm-rdna` (Sep 29–30) + `rdna_extras` commits (30 Sep–1 Oct): EXL3 is a **weight format** —
you do not need the **exllamav3** runtime. mul1 decode / K=1..8 from `rdna_extra/v0.30.0` is
now **on `rdna_extras` HEAD** and **GPU-run on gfx1030** (215/215 EXL3 tests; Qwen3.8-27B
3.00 bpw mul1 **TP=4** `FULL_AND_PIECEWISE` + `RDNA_ATTN` coherence pass). Author-host
[`74315f4`](https://github.com/opengfx1030/vllm-rdna/commit/74315f4cf7e6f20a31363dbbc6e4b6c3fae51f4d):
1k/512 **~18.9 t/s** (c=1) / **~81.8 t/s** agg (c=8); 16k/1k **~13.7 t/s** / **~830 tok/s**
PP (c=1). **Needs verify** on other hosts. Not in Hub `-extras`. Draft
[`#32`](https://github.com/opengfx1030/vllm-rdna/pull/32) is the earlier kernel-port PR; the
loader rewrite landed as follow-up commits on HEAD.

Stale inductor artifacts: `VLLM_DISABLE_COMPILE_CACHE` does **not** clear
`TORCHINDUCTOR_CACHE_DIR` — [troubleshooting](../troubleshooting/vllm.md#torchinductor-cache-survives-vllm-disable-compile-cache).

`#vllm-rdna` (Oct 2): current `rdna_extras` HEAD is **tested for mul1 only**. The maintainer has
**not** run 3inst. Prefer a **mul1** pack on HEAD. Public community mul1 27B:
[`quark75/Qwen3.8-27B-EXL3-3.0bpw`](https://huggingface.co/quark75/Qwen3.8-27B-EXL3-3.0bpw)
(use with the [27b-exl3 serve recipe](serve-scripts.md)). A calibrated 3inst Flash-Next upload
exists — [`BlivionIaG/Qwen3.8-Flash-Next-EXL3-3bpw-3inst`](https://huggingface.co/BlivionIaG/Qwen3.8-Flash-Next-EXL3-3bpw-3inst)
— but was **not GPU-tested** on that date. Do not treat it as a recipe.

A mixed 3+6 bpw pack around **50 GB** was called tight for **2× V620** and still needed
calibration. Experimental kernels had only been tried on an **Ornith 1.5 9B** quant — not
confirmed on 2× Flash-Next. `#general` (Sep 18) and `#vllm-rdna` (Oct 2): older **0.27 / 0.28**
images can **unpack** an EXL3 download (one ~12.5 GB pack ballooned to **~54 GB**) and OOM —
size host RAM, not just the download. Flash-Next EXL3 is still **not a 1× V620 path** without
heavy RAM offload.

An uncalibrated public 3inst upload was posted and immediately flagged for rework — do not treat
it as a stable checkpoint:
[`BlivionIaG/Qwen3.8-Flash-Next-EXL3-3bpw-3inst-uncalibrated`](https://huggingface.co/BlivionIaG/Qwen3.8-Flash-Next-EXL3-3bpw-3inst-uncalibrated).
The later **calibrated** 3inst repo above is still **Needs verify**.

`#vllm-rdna` (27 Oct): **EXL3 vs AWQ** on current gfx1030 kernels — EXL3 wins
**size and quality**; AWQ wins **speed and complexity**. Day-to-day testing still
uses **AWQ**. EXL3 kernel work is **paused** in favor of a new RDNA-oriented
weight format ([`BlivionIaG/hi2`](https://github.com/BlivionIaG/hi2)). Public
README: `rdna_hint_flex` stores each row as **INT2 or INT4** (INT8 on request)
plus block scales, an optional smoother, and an **order-16 Hadamard**; you set a
**quality target** and the tool reports the bit rate (`--target-bpw 3.0` is the
compact recipe). HIP kernels and the vLLM loader live in the **vLLM RDNA2 fork**,
not in the hi2 repo. `#vllm-rdna` (30 Oct): **3 bpw** was called a bit
aggressive; **3.5 bpw** may hold quality better but QAD is slow. `#hi2` (1 Nov):
public
[`BlivionIaG/Qwen3.8-27B-HI2-3.0bpw`](https://huggingface.co/BlivionIaG/Qwen3.8-27B-HI2-3.0bpw)
is a **~15 GiB** 27B pack (HF may mis-report the parameter count). Author card:
**66 % INT2 / 34 % INT4**, measured **3.012 bpw**; tool-call / first-tool match
is weaker than **AWQ INT4** at this rate (community: 3.2 bpw may be the next
try). There is still **no** vLLM `RECIPE=` / Hub bake. Do not treat hi2 as a
day-to-day serve path — kernels stay on the RDNA2 fork, not a second clone.

Prefer GPTQ/AWQ on published images until EXL3/Quark land in a tagged Docker build.

`#vllm-rdna` (Sep 29): **W4A8** started as an explore path (public
[`opengfx1030/vllm-rdna#9`](https://github.com/opengfx1030/vllm-rdna/pull/9), `w4a8-wiring` branch).
`#vllm-rdna` (Oct 1): W4A8 is **in `rdna_extras` HEAD** (`VLLM_RDNA2_W4A8_SDOT4` / serve-script
`W4A8=1`) but was **not much faster** than W4A16 dequant-on-the-fly — the stated win then was
**more context**, not decode. `#vllm-rdna` (Oct 9): a later kernel on the **v0.31.0** test line
was called **somewhat competitive vs W4A16**. Merged
[`#140`](https://github.com/opengfx1030/vllm-rdna/pull/140) (26 Oct): `27b-awq` recipe defaults
**`W4A8=1`** on HEAD (int8-activation prefill GEMMs for M≥33; decode stays W4A16). Author-host
4× TP4: 16k c=1 PP **860 → 1075** tok/s; c=8 aggregate out **42.5 → 52.3**; paired PPL ~unchanged.
**Fork-source** on that host; Hub **`v0.31.0-extras`** may still leave the knob off — check
`PRINT=1`. Set `W4A8=0` to force W4A16 prefill. EXL3 still **needs optimizations**. **W8A8**
was called theoretically the fastest int8 layout, not a current recipe.

## INT4 KV cache (`int4_fp16_block`, HEAD)

Merged [`#138`](https://github.com/opengfx1030/vllm-rdna/pull/138) (25 Oct): opt-in
`--kv-cache-dtype int4_fp16_block` — packed INT4 payload + FP16 per-32-channel block scales
(plus a 4-bit zero-point in the low nibble of each scale word). The RDNA2 QSA path
**reads/writes natively** (dequant in registers into the same FP16 LDS tile). Backends that
cannot pack the record **refuse** the dtype up front. Default FP16 KV is unchanged.

**Not** in Hub `v0.31.0-extras`. MonoKernel + this dtype still **Needs verify**.
`#vllm-rdna` (21 Oct): maintainer priority for longer context is still restoring the
**INT8** KV HIP path ([draft `#71`](https://github.com/opengfx1030/vllm-rdna/pull/71)), not
replacing day-to-day FP16 with experimental packs. Snapshot: [Fork progress](fork-progress.md).

## Intel AutoRound W4A16 (Flash-Next, `#vllm-rdna` Sep 2026)

Not in published `-extras` tags. Community + draft
[`opengfx1030/vllm-rdna#5`](https://github.com/opengfx1030/vllm-rdna/pull/5) are exercising Intel
**W4A16 AutoRound** Flash-Next:

| Checkpoint | Role |
|---|---|
| [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | PR-validation checkpoint (W4A16 weights, FP16 serve path) |
| [`Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound) | RTN sibling cited in `#vllm-rdna` (high publisher recovery %) |
| [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ) | `#vllm-rdna` Sep 25 — shorter-reasoning AWQ (`compressed-tensors`). [Swift notes](flash-next.md#swift-15-flash-next) |
| [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AutoRound) | Swift AutoRound sibling — **50-iter light** tune; not the Intel 200-iter pack. `#vllm-rdna` Sep 27: **looped badly** on one host — prefer the AWQ sibling first |

Community take: Intel AutoRound sits **between Unsloth Q5_XL and Q6_XL** on quality while the
weight pack is tens of GB smaller (~**75 GB**). That still wants **four 32 GB** cards plus a
large **CPU PLE / n-gram** table (embedded BF16 table is ~**95 GiB** — offload to RAM).

Do **not** assume the historical group-16 INT4 PLE **sidecar** matches published AutoRound benches
(those used BF16 PLE). Merged [`#49`](https://github.com/opengfx1030/vllm-rdna/pull/49) (11 Oct)
plus the v0.31.0 launcher (`PLE=int4|int4-disk`) is the current INT4 path. See
[overview](flash-next.md#intel-autoround-flash-next).

`#vllm-rdna` (12 Oct): latest `rdna_extras` can fail at construction on this Intel pack
(`NotImplementedError` / `INCConfig` for the unquantized PLE table). Upstream
[vLLM #59990](https://github.com/vllm-project/vllm/pull/59990) is merged; the gfx1030 cherry-pick
was still queued. [Troubleshooting](../troubleshooting/vllm-flash-next.md#autoround-ple-incconfig).

`#vllm-rdna` (Sep 20): merged [`opengfx1030/vllm-rdna#15`](https://github.com/opengfx1030/vllm-rdna/pull/15)
used this Intel AutoRound pack with the **unquantized / original BF16** PLE table (~**100 GB** class
in RAM), **not** `wtdcode` AWQ + `primitive-ai` PLE-quant. **INT8 decode shadows** remain
experimental — they are **excluded** from that merge (one host reported ~42 → ~55 t/s without MTP;
**Needs verify**, and quality loss was already seen on Sep 16).
