# Flash-Next troubleshooting

Symptoms that show up on the Qwen3.8 Flash-Next / `rdna_extras` path. Startup failures that also hit
Hub `-extras` (CUDA graphs, AMDSMI, RCCL version pin, first-boot compile) stay on
[vLLM troubleshooting](./vllm.md).

Serve lines: [Flash-Next](../vllm/flash-next.md).

## Flash-Next long-prompt stalls {#flash-next-long-prompt-stalls}

Symptom (Flash-Next fork / recipe containers, `#vllm-rdna` Sep 2026): short prompts decode fine, but
**large prompts** (tens of k tokens — agentic coding, session resume) take many minutes, timeout, or
appear wedged. Temps and power caps look healthy. Sep 15 community add-on: **intermittent** TTFT
from a few seconds to **minutes** (one host: up to ~300 s on a tiny prompt), with one or more GPUs
at **100% util but ~40 W** — not a PCIe drop; looks like an RCCL / runner stall.

**Community fix that unblocked 4× V620 hosts:**

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

One host then saw stable **~68 tok/s** with dense INT8 + custom all-reduce; another (leapdragon
container, TP4) reported the stall **gone** after the same switch. `#vllm-rdna` Sep 15: **MoE on
the 0.28 Flash-Next line wants V1** until upstream **0.29** (V2 becomes the default). The Flash-Next
docs already call `V2=0` out. Official-extras authors note separate Dense-on-V2 fixes on the org
rebase — A/B both values on your image.

Also rule out thermal / power first ([Power tuning](../tuning/power.md)), and measure expected
prefill time (~1k tok/s class ⇒ ~40 s for 40k tokens, not minutes). Prefill campaign numbers:
[vLLM overview](../vllm/flash-next.md#qwen38-flash-next-on-vllm).

## Flash-Next 128k prefill cliff {#flash-next-128k-prefill-cliff}

Symptom (`#vllm-rdna` Sep 2026, Intel AutoRound / draft
[`opengfx1030/vllm-rdna#5`](https://github.com/opengfx1030/vllm-rdna/pull/5)): prefill holds
**~950 tok/s** through 64k, then falls to **~375 tok/s** at **128k** while decode stays flat
(~48–55 tok/s). Looks like a scheduler / chunking misconfig more than a kernel cliff.

**Community fix on that host:** raise the batch cap:

```bash
--max-num-batched-tokens 4096
```

The suspected combination was `enable_chunked_prefill=True` with **2048** scheduled / batched
tokens. After 4096, 128k prose/code PP returned to the **~950 tok/s** class (same 4× V620, FP16
KV, CPU PLE offload). See [overview](../vllm/flash-next.md#intel-autoround-flash-next).

If output is **wrong** rather than just slow, stay on **2048** and switch graphs to **PIECEWISE**
first — [FULL-graph corruption](#flash-next-full-graph-corruption).

If 128k is still slow after 4096, A/B [V2 runner](#flash-next-long-prompt-stalls) and confirm you
are not on quantized PLE (known-good is the embedded BF16 n-gram table).

## Flash-Next PP3 output corruption on `rdna_extras` {#flash-next-pp3-output-corruption}

`#vllm-rdna` (Sep 15): **Qwen Next / Flash-Next** on **3× V620**, `PP=3` / `TP=1`, TheRock **10.0**,
`--enforce-eager`. A **leapdragon** tree (with the same local PP patches) decoded correctly. The same
prompt on **`rdna_extras` HEAD** (community pin `f663686`) **corrupted after ~3 chunks** — first
request of a fresh server, ~4k-token prompt, only a fraction of tokens correct.

Disabling `VLLM_USE_V2_MODEL_RUNNER`, `VLLM_USE_RDNA2_FA`, `VLLM_GDN_HIP_PREFILL`, prefix cache, and
forcing single-chunk prefill (`--max-num-batched-tokens 8192`) **did not** fix it. Suspected area:
HIP GDN decode / prefix-cache interaction — **not confirmed**.

`#vllm-rdna` (Sep 16): a **fresh `rdna_extras` pull still corrupted** on PP3. One host still decoded
on a **leapdragon** tree. Community suspected
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (`_forward_core_decode_non_spec` /
`gdn_decode_rdna2`) — **Needs verify**. `--max-num-seqs 4` (and later **6** with CUDA-graph capture
sizes `[1,2,4,8]`) reduced some TP corruption while debugging; it is **not** a PP3 fix.

**What to do:** if you need 3-card Flash-Next today, stay on a **known-good leapdragon image** unless
you are on the later pin below. Do not treat an arbitrary org HEAD as a drop-in for PP3.

## Flash-Next PP3 graph KeyError on small prefill (`b33f9b6`) {#flash-next-pp3-graph-keyerror}

`#vllm-rdna` (Sep 19): community test of
[`rdna_extras` @ `b33f9b6`](https://github.com/opengfx1030/vllm-rdna/commit/b33f9b66eb2b90d62b8febd78b53aab1f4fd30c2)
with a production config transposed to **PP=3 / TP=1** on **3× V620** (P2P distance **PHB**, V1
runner, `FULL_AND_PIECEWISE` + breakable graphs). Six PP3 patches for that pin are public in
[`alanoo81/flashnext-v620-pp3` `patches/rdna_extras-b33f9b6`](https://github.com/alanoo81/flashnext-v620-pp3/tree/master/patches/rdna_extras-b33f9b6)
(write-up: `docs/opengfx1030-report.md`). This is a **later** pin than the
[Sep 15–16 corruption](#flash-next-pp3-output-corruption) — do not collapse the two bugs.

On that pin:

- CUDA graphs **booted** under PP.
- Deterministic **NaN** with prefix caching + concurrent requests was **gone** on their stab
  (seed 1, 45 iterations including 4-stream bursts, 0 corrupted outputs). That does **not**
  retract older pins.

**Still blocking, PP only:** the engine dies on the first prefill batch small enough to hit a
**captured CUDA-graph size**. A **74-token** prompt failed, and so did the **240-token** tail of a
**4336-token** prompt; **2048** and **4096** passed. Symptom: `KeyError` on the request id inside
`scheduler.py` `update_from_output` (`model_runner_output.req_id_to_index`, path
`step_with_batch_queue`). No worker-side error. Same failure with `cudagraph_mode=PIECEWISE`,
prefix caching off, and TunableOp off. **Eager is fine.**

Workaround — pin capture sizes so prefill stays eager:

```bash
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
```

Community numbers **with** that workaround on the same host: prefill **1077 / 1611 tok/s** at
4K / 16K, decode **27.8 tok/s** versus **26.7** eager (graphs bought ~4% under PP). **Needs verify.**

Also on that run:

- Stage 0 **OOM** when `--kv-cache-memory-bytes` went above about **1.5e9** (`qsa_mqa_paged` on
  the first long prompt at **2.5e9**). Do not copy the 4× recipe's **7 GiB** cap onto this PP3 pin.
- Logged prefix-cache hit rate stayed **0.0%**. Maintainer: the cache fix is in, but **hit-rate
  reporting is unreliable** — judge by repeat-prompt TTFT, not the counter. See
  [prefix cache](#flash-next-prefix-cache-zero).

## Flash-Next FULL-graph decode corruption {#flash-next-full-graph-corruption}

Symptom (`#vllm-rdna` Sep 16–17, `rdna_extras` Flash-Next, **TP**): serve starts, then output
**corrupts** after a few chunks or after turning features back on. Prefix cache + graphs can look
healthy, then fail again. Community isolated **FULL** CUDA graphs (decode) as the bad path;
**`PIECEWISE` held**.

**Community workarounds that unblocked a 4× V620 host:**

```bash
--compilation-config '{"cudagraph_mode":"PIECEWISE","compile_ranges_endpoints":[]}'
--max-num-seqs 6
--max-num-batched-tokens 2048
```

Also used while hunting: `HSA_NO_SCRATCH_RECLAIM=1`, `--max-num-seqs 4` (stricter), and leaving
CUDA-graph capture sizes **unpinned** so vLLM chose `[1,2,4,8]`. A 16k/1k × 8 concurrency
snapshot was called a **stable base** before performance claw-back.

This does **not** replace [PP3 corruption](#flash-next-pp3-output-corruption). Hub `-extras` **27B**
can still use `FULL_AND_PIECEWISE` — see [Configuration](../vllm/configuration.md#cuda-graphs-preferred-over---enforce-eager).
Sanitized serve line: [Flash-Next PIECEWISE recipe](../vllm/flash-next-serve.md#flash-next-4x-piecewise).

The in-tree Flash-Next launcher (`scripts/serve_gfx1030_flashnext.sh`, comment 18 Sep 2026) uses
`FULL_AND_PIECEWISE` and states that mode **executes as PIECEWISE on ROCm**
(`rocm_full_executes_as_piecewise`). It also attributes an earlier “corrupts at c=8” report to
**probe artifacts** (reasoning-parser field / reasoning-budget), not the graphs.

`#vllm-rdna` (Sep 23–24): merged [`vllm-rdna#17`](https://github.com/opengfx1030/vllm-rdna/pull/17)
**measures `FULL_DECODE_ONLY`** (mode `0`, capture `[3,6,12]` with MTP-2). That is **not** the
same as **FULL-only** (which still corrupted earlier). **Merged**
[`#20`](https://github.com/opengfx1030/vllm-rdna/pull/20) (24 Sep) makes compiled `FULL_AND_PIECEWISE`
**boot and stay correct**, but decode dropped **~63–70 → ~26–34 t/s** (prefill held; community
~2.5× slower). Prefer `#17`’s `FULL_DECODE_ONLY` until a graph-launch follow-up lands. See
[4× `#17` recipe](../vllm/flash-next-serve.md#flash-next-4x-pr17).

`#vllm-rdna` (Sep 17): `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (in that recipe) **turns torch.compile off**.
Community A/B on a 4× TP Flash-Next tree: **~39 t/s** with breakable/eager vs **~55 t/s** after compile
stayed on (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`) plus a `.contiguous()` on a hyper-connection injection
view. Keep `1` until output is clean; then A/B `0` if you want the compile path back.

## wvSplitK GPU fault after profile {#wvsplitk-gpu-fault}

`#vllm-rdna` (Sep 17), `rdna_extras` pin `50120e13b`, TheRock **7.14.1**, **4× V620** TP4: right after
the profile run, skinny GEMM `wvSplitK_hf_sml_<half,…>` (`skinny_gemms`) raised
`HSA_STATUS_ERROR_EXCEPTION`. **`VLLM_RDNA_DENSE_GEMV=1`** avoided the fault on more than one host.
It does **not** fix [PP3 corruption](#flash-next-pp3-output-corruption).

The GPU core dumps written on that path are **several GB each** and land in the **process working
directory** — start the server from a scratch dir (the 4× recipe already `cd`s to `/tmp`).

## Flash-Next vision startup / runtime OOM {#flash-next-vision-oom}

Two different OOM modes (`#vllm-rdna` Sep 17–21 + the in-tree launcher):

| When | Cause | What to do |
|---|---|---|
| **Startup** | mm-profiling dummy image (~24.8M px) → vision SDPA math backend builds a **~64 GiB** L×L fp32 score matrix | Set `--mm-processor-kwargs '{"max_pixels":1605632}'`. `--limit-mm-per-prompt '{"image":1}'` alone is **not** enough (count was already 1). |
| **Runtime** | vLLM may **not reserve** vision memory; a **tight KV** plus an image OOMs after a clean start | Leave more free VRAM / lower `--max-model-len`; community **~15–20 s per image**. `#general` (Sep 21): vision + **PP3** is especially tight. |

Leave `--language-model-only --skip-mm-profiling` if you do not need images. See
[4× recipe vision notes](../vllm/flash-next-serve.md#flash-next-4x-piecewise).

## Flash-Next hybrid KV log overstates capacity {#flash-next-hybrid-kv-overstated}

`#vllm-rdna` (Sep 17): Flash-Next **hybrid** KV (main KV + GDN conv/SSM + PLE conv **per page**) can
**overstate usable tokens by ~2.5×**. Community measured vs log:

| What the log said | Measured single-request peak | Notes |
|---|---|---|
| ~411k tokens | ~166k (MTP=0) / ~100k (MTP=3) | One tree on 4× V620 |
| ~468k tokens | ~190k (100k prompt ≈ 52%; 150k ≈ 79%) | `rdna_extras` `50120e13b`, no MTP |

A prompt **between the real pool and `--max-model-len`** can **livelock**: KV fills, resets, refills,
and `num_preemptions` stays **0**. Size `--max-model-len` / `--kv-cache-memory-bytes` from a
**measured** peak (one long request, watch `%` usage), not the advertised token count.

## Flash-Next prefix cache never hits (pre-fix HEAD) {#flash-next-prefix-cache-zero}

`#vllm-rdna` (Sep 17): on `rdna_extras` **before**
[`e45dd5cb`](https://github.com/opengfx1030/vllm-rdna/commit/e45dd5cb2de8218defe19878fd75e39528f0acbc),
`--enable-prefix-caching` reported **0%** hits on repeat Flash-Next / Qwen4Exp prompts (community
18k / 4.7k tests). The QSA compression ring returned length 0 at a group boundary, and the hybrid
coordinator took the **min** across groups.

**Fix (fork-source):** pull `rdna_extras` **at or after** that commit. Maintainer validation on 4×
V620 TP4 PIECEWISE: repeat ~18.8k prompt TTFT **12.2 s → 0.3 s**, server hit rate **0% → ~50%**.

A leapdragon tree without MTP still showed **97–99%** hits from the second request; leapdragon + MTP
needed a second computation and
[vLLM #54044](https://github.com/vllm-project/vllm/pull/54044) (reset Mamba align metadata after
profiling) before MTP + graphs + prefix cache stayed correct. `#55506` /
[`741e5bc3`](https://github.com/opengfx1030/vllm-rdna/commit/741e5bc31ae5a14ab8926e2defaa616fdd87408a)
is a **separate** V2 mamba spec-decode block-table port — community: it did **not** fix the 0% /
`!!!` prefix-cache path on V1.

If you still see 0% hits, confirm the commit, then A/B `--no-enable-prefix-caching` only as a
correctness test (you lose the TTFT win).

`#vllm-rdna` (Sep 19): a PP3 run on `b33f9b6` still logged **0.0%** hits for the whole session.
Maintainer: prefix caching **was fixed**, but the **reported** hit rate can stay wrong. A 0%
counter is not proof the cache is dead — A/B repeat-prompt TTFT. Details:
[PP3 graph KeyError](#flash-next-pp3-graph-keyerror).

## FP8 KV rejected on QSA / Flash-Next {#fp8-kv-rejected-on-qsa}

`#vllm-rdna` (Sep 22): `--kv-cache-dtype fp8` **fails at startup** on current QSA Flash-Next rather
than falling back. Two independent gates:

1. **QSA backends are unquantized-only.** Community traceback: `QSAStateBackend` /
   the AMD QSA owner declare `supported_kv_cache_dtypes = ["auto", "bfloat16", "float16"]`.
   Backend selection records `kv_cache_dtype not supported` and never picks those layers — so
   there is **no** backend that can serve `--kv-cache-dtype fp8` (or `int8_per_token_head`) on
   this model.
2. **ROCm FP8 KV path is AITER / CDNA.** Separate hosts got an **AITER and CDNA only** refusal
   even off the QSA stack. Maintainer: fp8 KV **used to work** on older dense 27B and may be
   re-enableable; it is **not** unlocked on current extras.

Stay on **fp16 KV**. See [Quantization — KV-cache dtype](../vllm/quantization.md#kv-cache-dtype).

## Flash-Next: no P2P, or MTP blocks PLE {#flash-next-no-p2p-or-mtp-blocks-ple}

Two `#vllm-rdna` (Sep 18) reports that look like “this image is slow” but are env / feature
mismatches:

| Symptom | What to try |
|---|---|
| Custom all-reduce / PIX recipe **fails to load** or logs **PYNCCL** | Host has **no GPU↔GPU P2P**. Drop `VLLM_FORCE_CUSTOM_ALL_REDUCE` and the PIX/RCCL P2P block — those are **env flags**, not a different tree. Use `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` — [Configuration](../vllm/configuration.md#custom-all-reduce--p2p-two-community-stacks). |
| Bench ~800 PP / ~50 t/s, then Open WebUI / Hermes ~14 t/s | Client / harness, **or** MTP on while PLE fused decode is required. Community: **MTP off** recovered **~38 t/s** decode and **~95%** prefix reuse on a no-P2P host (plus `--enable-prefix-caching`). Treat tok/s as **Community**. |
| GDN decode on Triton, gfx1030 FP16 | Expected fallback when the HIP GDN path is not selected — confirm `VLLM_RDNA_FORCE_FP16=1` and native GDN in logs before chasing P2P. |
