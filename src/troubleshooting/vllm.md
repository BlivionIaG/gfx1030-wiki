# vLLM troubleshooting

## CUDA graph capture crashes {#cuda-graph-capture-crashes}

Symptom: crash at `GDN _output_projection all-reduce` / `_SimpleCData.__new__`, or OOM during graph capture.

**Root cause (fixed in current extras):** TP comm wrappers were not `allow_in_graph`. Fork fix:
`fix(distributed): allow TP comm ops in torch.compile graph capture`.

**First, pull the latest image:**

```bash
docker pull blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0
```

Then try CUDA graphs (fast path):

```bash
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

Mount cache volumes — see [Configuration](../../vllm/configuration.md#cache-volumes-first-boot-is-slow).

**Fallback:** `--enforce-eager`

On multi-GPU AOT cache issues: `VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1`

Low throughput (~4–5 t/s on 27B)? Check for **older image** where AWQ still used Triton — see
[Quantization](../../vllm/quantization.md).

## First boot is extremely slow

Triton and torch.compile JIT on first run. Typical cache: `~/.triton/cache` (~3 GB),
`~/.cache/vllm/torch_compile_cache` (~700 MB).

## GPTQ/AWQ not using RDNA2 kernels

Check logs for `Using RDNA2W4A16LinearKernel`. If you see Triton/Exllama instead:

```bash
export VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel
```

Confirm `-extras` image from current extras. See [Fork kernel dispatch](../../vllm/fork.md#kernel-dispatch-on-gfx1030).

## vLLM picks the wrong platform / doesn't see my Radeon

Use published [`blivioniag/vllm-rdna`](../../vllm/images.md) images with `patches/*rocm-platform*` fixes rather
than stock upstream builds.

## `Failed to infer device type` / `AMDSMI_STATUS_NOT_INIT`

Symptom: vLLM logs `ROCm platform is not available because no GPU is found` and
`AMDSMI_STATUS_NOT_INIT - Device not initialized` (often right after CUDA/NVML is also missing — that
part is expected on AMD).

Check the Docker device/group block **exactly** — `#vllm-rdna` hits this when `render` is missing:

```yaml
devices: [/dev/kfd, /dev/dri]
group_add: [video, render]    # both — video alone is not enough on many hosts
ipc: host
security_opt: [label=disable]  # or seccomp=unconfined on docker run
```

Host user still needs `render`/`video` as in [General troubleshooting](./general.md). Debug with
`VLLM_LOGGING_LEVEL=DEBUG`. If the same compose worked on an older tag, `docker pull` a known-good
image — a bad rebuild can also fail AMDSMI init.

## `ROCM_ATTN` hangs for hours (Triton compile)

AMD Triton flash-attention compile on gfx1030 can sit there for **hours** (RCCL vs Triton). On `-extras`,
use `--attention-backend RDNA_ATTN` and/or `VLLM_USE_RDNA2_FA=1` instead of `ROCM_ATTN`. See
[Configuration](../../vllm/configuration.md).

`FA_RDNA2` / `RDNA_ATTN` may not appear in the backend list on older `-extras` images or some GPTQ
models (logs only show Triton / ROCM / TurboQuant). Pull the latest `-extras` tag and confirm
`Using RDNA2W4A16LinearKernel` / native FA in startup logs. Qwen3.8-27B AWQ needs **head size 256**
on the fork — see [Quantization](../../vllm/quantization.md#int4-on-gfx1030-no-native-int4-alus).

### `shm_broadcast` / one GPU + one CPU pegged {#shm-broadcast-triton-vs-rccl}

Symptom (`#vllm-rdna` Sep 2026, often Flash-Next / multi-GPU): serve looks wedged; logs repeat
something like:

```text
No available shared memory broadcast block found in 60 seconds.
This typically happens when some processes are hanging or doing some time-consuming work
(e.g. compilation, weight/kv cache quantization).
```

Community diagnosis: classic **Triton JIT compile fighting RCCL** — not necessarily a dead process.
Guidance:

1. **Wait** — first boots can sit like this a long time; keep the Triton / compile cache volumes
   mounted ([Configuration](../../vllm/configuration.md#cache-volumes-first-boot-is-slow)).
2. Stay on **ROCm 7.2.0 or 7.14.x** — avoid mid-7.2.x (same pin as
   [multi-GPU RCCL](#multi-gpu-rccl-hangs-or-cards-drop-offline)).
3. Prefer HIP / `RDNA_ATTN` paths over AMD Triton FA where the fork offers them — less Triton means
   fewer of these stalls.
4. Next cold start with a warm cache should be much shorter; if it never recovers after hours, A/B
   `VLLM_USE_V2_MODEL_RUNNER=0` and the [long-prompt](#flash-next-long-prompt-stalls) notes.

## Multi-GPU RCCL hangs or cards drop offline

If TP works on one image and dies after a host ROCm bump, check the **ROCm version** before the
model. **7.2.1 through ~7.13** are reported to have a multi-card RCCL bug. Stay on **7.2.0** or
**7.14.0** — see [Installing ROCm](../../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

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

Also rule out thermal / power first ([Power tuning](../../tuning/power.md)), and measure expected
prefill time (~1k tok/s class ⇒ ~40 s for 40k tokens, not minutes). Prefill campaign numbers:
[vLLM overview](../../vllm/overview.md#qwen38-flash-next-on-vllm).

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
KV, CPU PLE offload). See [overview](../../vllm/overview.md#intel-autoround-flash-next).

If output is **wrong** rather than just slow, stay on **2048** and switch graphs to **PIECEWISE**
first — [FULL-graph corruption](#flash-next-full-graph-corruption).

If 128k is still slow after 4096, A/B [V2 runner](#flash-next-long-prompt-stalls) and confirm you
are not on quantized PLE (known-good is the embedded BF16 n-gram table).

## Prefill blocks decode / MTP stalls under concurrency

Symptom: with speculative decode (MTP) and multiple in-flight requests, generation stalls while
prefill runs; or graph + MTP3 reaches "Application startup complete" then hangs on PLE lookup /
`sample_tokens` timeout.

Community notes (`#vllm-rdna`):

- Prefer **GPTQ + RDNA2 W4A16** (or AWQ HIP) paths over GGUF-in-vLLM for these cards.
  `#vllm-rdna` (Sep 15): `RDNA2W4A16MoEExperts` was reported to raise **usable concurrency** vs the
  older MoE path — confirm the kernel name in logs; A/B if your image is older.
- **One prefill stream can fully block other decode streams** (community Sep 15). When you bench,
  alternate prefill+decode in flight — concurrency-only or decode-only numbers miss this.
- Concurrent MTP / prefill-vs-decode fixes land in community recipes first — see open PRs on
  [`leapdragon/vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe).
- MTP=0 vs MTP=3 are different bug surfaces; a commit that "works" at MTP=3 can still emit spurious
  tokens at MTP=0. A/B and pin a known-good recipe commit.
- Slow or broken P2P + custom all-reduce can look like MTP latency bugs — A/B the
  [disable vs PIX custom AR](../../vllm/configuration.md#custom-all-reduce--p2p-two-community-stacks)
  stacks.

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
can still use `FULL_AND_PIECEWISE` — see [Configuration](../../vllm/configuration.md#cuda-graphs-preferred-over---enforce-eager).
Sanitized serve line: [Flash-Next PIECEWISE recipe](../../vllm/recipes.md#flash-next-4x-piecewise).

The in-tree Flash-Next launcher (`scripts/serve_gfx1030_flashnext.sh`, comment 18 Sep 2026) uses
`FULL_AND_PIECEWISE` and states that mode **executes as PIECEWISE on ROCm**
(`rocm_full_executes_as_piecewise`). It also attributes an earlier “corrupts at c=8” report to
**probe artifacts** (reasoning-parser field / reasoning-budget), not the graphs. Treat
**`PIECEWISE` and `FULL_AND_PIECEWISE` as equivalent on ROCm**; do **not** switch to **FULL-only**.

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
[4× recipe vision notes](../../vllm/recipes.md#flash-next-4x-piecewise).

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

## Upstream KV offload tanks decode {#upstream-kv-offload-tanks-decode}

`#vllm-rdna` (Sep 15): **upstream vLLM** CPU → SSD **KV cache offload** (not LMCache) failed to
bring blocks back to VRAM usefully. Community: decode fell to the **~1 t/s** class. **Mamba / SSM**
state models are called out as especially unreliable on this path (including on Hopper-class hosts
in-channel).

`#vllm-rdna` (Sep 16): public RAM-offload Flash-Next packs (example:
[`Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound))
are interesting for long context, but recipes that need **`--no-enable-prefix-caching`** are a
**non-starter** on gfx1030 vLLM. Qwen4exp KV is already relatively efficient; parking **non-linear**
state in host RAM fights prefix cache. **LMCache** is still the intended overflow path.

Do not plan production multi-chat overflow on native vLLM KV offload. `#lmcache` is still the
intended gfx1030 path (standalone LMCache server + vLLM connector) but has **no published recipe**
yet. See [fork landscape](../../vllm/fork.md#consolidation-status).

`#vllm-rdna` (Sep 20): upstream [`vllm#57160`](https://github.com/vllm-project/vllm/pull/57160)
(mainline ROCm, **not** v0.29 and **not** confirmed on `rdna_extras`) changes CPU KV offload to
private pinned tensors after `cudaHostRegister` failures on large TP. Treat as a tracking note —
do not assume it restores usable decode on gfx1030.

`#vllm-rdna` / `#general` (Sep 18–19): official **`rocm/pytorch`** images are the **wrong** place to
compile an LMCache connector (missing HIP / developer packages). Intended shape:

1. Run [`lmcache/standalone`](https://hub.docker.com/r/lmcache/standalone) in **CPU** mode — see the
   [standalone starter](https://docs.lmcache.ai/getting_started/quickstart/standalone_starter.html).
2. Add the LMCache connector to a **bare-metal / venv** `rdna_extras` serve, or patch it onto
   [`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) / [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna).
3. Do **not** use `lmcache/vllm-openai` on gfx1030 — that image is the CUDA path.

NVMe-as-KV is the usual motive (low host RAM). Until someone posts a working gfx1030 compose, treat
this as **Needs verify**.

`#vllm-rdna` (Sep 22) + `#lmcache` (Sep 23): still **no working gfx1030 build**. A community prompt
claimed LMCache HIP needs **~143 KB LDS** vs gfx1030 **64 KB** — **Needs verify** (that number came
from an LLM, not a posted compile log). In-channel: **Mamba / SSM alignment** was the last known
real blocker even on Hopper; another host still failed to compile against V620. Keep the
standalone-CPU + connector shape above; do not treat LDS size as settled.

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

Stay on **fp16 KV**. See [Quantization — KV-cache dtype](../../vllm/quantization.md#kv-cache-dtype).

## Leftover vLLM / PLE workers after a restart {#leftover-vllm-workers}

`#vllm-rdna` (Sep 18): killing the OpenAI API process is **not** enough. `VLLM::Worker`,
`VLLM::EngineCore`, and `PleOffloadWorker` can stay up and hold VRAM / the PLE sidecar. Before
relaunch, stop those leftover processes (match the **process names**, not a broad pattern) and wait
a few seconds. The [4× Flash-Next recipe](../../vllm/recipes.md#flash-next-4x-piecewise) already
`cd`s to `/tmp` so GPU coredumps do not land in a checkout.

## Flash-Next: no P2P, or MTP blocks PLE {#flash-next-no-p2p-or-mtp-blocks-ple}

Two `#vllm-rdna` (Sep 18) reports that look like “this image is slow” but are env / feature
mismatches:

| Symptom | What to try |
|---|---|
| Custom all-reduce / PIX recipe **fails to load** or logs **PYNCCL** | Host has **no GPU↔GPU P2P**. Drop `VLLM_FORCE_CUSTOM_ALL_REDUCE` and the PIX/RCCL P2P block — those are **env flags**, not a different tree. Use `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` — [Configuration](../../vllm/configuration.md#custom-all-reduce--p2p-two-community-stacks). |
| Bench ~800 PP / ~50 t/s, then Open WebUI / Hermes ~14 t/s | Client / harness, **or** MTP on while PLE fused decode is required. Community: **MTP off** recovered **~38 t/s** decode and **~95%** prefix reuse on a no-P2P host (plus `--enable-prefix-caching`). Treat tok/s as **Community**. |
| GDN decode on Triton, gfx1030 FP16 | Expected fallback when the HIP GDN path is not selected — confirm `VLLM_RDNA_FORCE_FP16=1` and native GDN in logs before chasing P2P. |

## ROCR idle CPU spin (TheRock 7.14) {#rocr-idle-cpu-spin-therock-714}

Symptom: after starting a multi-GPU vLLM serve on **TheRock / ROCm 7.14** (ROCR **1.21**), the host
holds several CPU cores at high utilization even when the GPUs are idle (~one core per HIP process,
plus more once RCCL initializes).

Root cause: `AsyncEventsLoop` / signal-wait paths busy-spin without backoff
([ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051),
[ROCm/ROCm#6522](https://github.com/ROCm/ROCm/issues/6522)). Stock env knobs
(`HSA_ENABLE_INTERRUPT`, etc.) do not fix multi-GPU cases.

**Fix:** rebuild only `libhsa-runtime64.so` with the poll-backoff patch and `LD_PRELOAD` it. Step-by-step
for host builds (and note that recipe containers already bake the patch):
[`docs/rdna2/ROCR-CPU-FIX.md`](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/ROCR-CPU-FIX.md)
in [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next).
Pull latest recipe / container before re-debugging idle CPU.
