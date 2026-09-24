# Flash-Next serve lines

> **WIP / community.** Copy from here only after you have read [Flash-Next status](./flash-next.md)
> (which card count, which graph mode). Failures: [Flash-Next troubleshooting](../troubleshooting/vllm-flash-next.md).

## Path C — Flash-Next (`vllm-rdna2-qwen`)

For agentic / long-context Flash-Next on **4× V620**, use the Flash-Next vLLM stack — not Hub
`-extras` and not the 27B recipe presets:

- **Source of truth (Sep 14–15):** [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras` (leapdragon work cherry-picked; PLE load path on HEAD)
- **In-tree launcher (flags / vision):** [`scripts/serve_gfx1030_flashnext.sh`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/scripts/serve_gfx1030_flashnext.sh) — `#vllm-rdna` Sep 21 pointed 2-card / GPTQ questions here. Do **not** copy the script’s host paths.
- **Published container / docs (may lag):** [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next) — [`docs/rdna2/`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next/docs/rdna2)
- Weights: [`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
- PLE sidecar: [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant)

Expect large **host DRAM** for the n-gram / PLE store (community: **~64–95 GB** class; **128 GB**
host RAM was **not** enough for KV offload on one 4× host). Long-prompt / intermittent stalls: try
`VLLM_USE_V2_MODEL_RUNNER=0` — [troubleshooting](../troubleshooting/vllm-flash-next.md#flash-next-long-prompt-stalls).
Throughput and KV tightness: [Flash-Next status](./flash-next.md#qwen38-flash-next-on-vllm).

`#vllm-rdna` (Sep 17): there is **no Q3 / GGUF-Q3 path on vLLM**. Stay on W4A16 / AWQ / AutoRound.
EXL3 ~3 bpw is still **experimental** — [Quantization](quantization.md#experimental-exl3-and-quark-vllm-rdna-sep-2026).

### 4× V620 Flash-Next serve (`rdna_extras` `#17`, Sep 23) {#flash-next-4x-pr17}

[`opengfx1030/vllm-rdna#17`](https://github.com/opengfx1030/vllm-rdna/pull/17) **merged** 23 Sep 2026
into `rdna_extras`. This is the public landing of the unpublished ~1950 PP recovery after `#15`.
**Community / Needs verify.** Not a Hub tag. Do **not** copy host paths from the PR body — use the
in-tree docs (`docs/rdna2/V620-GIT-FAST-SERVICE.md`, `docs/rdna2/V620-BASELINE-PORT-20260922.md`)
and [`tunableop/`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/tunableop).

Operational facts that are safe to copy:

- **Resident W4A16 MoE** (`VLLM_RDNA_MOE_RESIDENT=1`): convert the checkpoint’s packed INT4 experts
  into the RDNA2 kernel layout **once at load**, keep that layout, and skip repacking every 4k
  prefill chunk. Precision stays W4A16.
- **Mamba retirement:** backport of [`vllm#55450`](https://github.com/vllm-project/vllm/pull/55450).
  Without it, long prompts (100k+ class) hit a deterministic **preemption loop** (state blocks not
  retired across null gaps).
- **Graphs:** `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12]}'`
  with **MTP-2**. Prefill stays eager. This is the **measured** path, not `PIECEWISE`.
- **TunableOp** is **rocBLAS-hash locked**. A mismatched CSV aborts the first GEMM or silently
  uses default algorithms — [regenerate](../troubleshooting/vllm.md#tunableop-rocblas-mismatch).
- Harness used for the PR numbers: [`GeorgeMA-Strong/llm-context-bench`](https://github.com/GeorgeMA-Strong/llm-context-bench).

PR-head 16k (author) vs second-host confirm (`#vllm-rdna` Sep 23):

| Cell | PR `#17` | Second 4× host |
|---|---:|---:|
| Regular 16k PP | 1,957.8 | 1,983.8 |
| Coding 16k PP | 1,982.8 | 1,985.3 |
| Regular / coding TTFT | 8.56 / 9.11 s | 8.44 / 9.10 s |
| Decode | 69.7 / 68.5 t/s | 63.1 / 70.1 t/s |

Open follow-ups (not the default): [`#20`](https://github.com/opengfx1030/vllm-rdna/pull/20) makes
compiled `FULL_AND_PIECEWISE` **correct** but **slower decode** (~26–34 t/s). Draft
[`#21`](https://github.com/opengfx1030/vllm-rdna/pull/21) tries to keep FULL decode graphs — **unbenched**.

The older Sep 17 PIECEWISE host-venv block below is still useful if you are **not** on `#17` yet.

### 4× V620 Flash-Next serve (`rdna_extras`, PIECEWISE, Sep 17) {#flash-next-4x-piecewise}

Sanitized from a `#vllm-rdna` (Sep 17) host-venv recipe on **4× V620**. **Community / Needs verify**
— not a published Hub tag. Requires [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna)
`rdna_extras` including the **GDN sanitizer** commit `388a61b6f`, and a **rocm_sdk** venv
(`torch 2.12` + **ROCm 7.14**). Paths below are generalized.

**Why this differs from the Hub `-extras` TP4 block:** Flash-Next on current `rdna_extras` **FULL**
decode graphs still **corrupt** output. Community: switch
`--compilation-config` to **`PIECEWISE`**, keep `--max-num-batched-tokens 2048`, and cap
`--max-num-seqs` at **4–6**. See
[FULL-graph corruption](../troubleshooting/vllm-flash-next.md#flash-next-full-graph-corruption).
Do **not** copy this graph mode onto the 27B `-extras` recipe unless you are A/B testing.

Community snapshot on that host (16k prompt / 1k gen, concurrency 8): **~3331 tok/s** prefill,
**~72.7 tok/s** decode, TTFT **~39 s**. Treat as a single-host number.

`#vllm-rdna` (Sep 17–18): pin `rdna_extras` **at or after**
[`e45dd5cb`](https://github.com/opengfx1030/vllm-rdna/commit/e45dd5cb2de8218defe19878fd75e39528f0acbc)
if you want prefix cache to actually hit, and set `VLLM_RDNA_DENSE_GEMV=1` if skinny `wvSplitK`
faults after the profile run. `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is the **stable / no-compile** path
(~39 t/s class in one A/B); `0` plus compile can be faster once graphs are clean — see
[FULL-graph troubleshooting](../troubleshooting/vllm-flash-next.md#flash-next-full-graph-corruption).
Do not trust the logged hybrid KV token count — [overstated pool](../troubleshooting/vllm-flash-next.md#flash-next-hybrid-kv-overstated).

`#vllm-rdna` (Sep 18): maintainer posted an updated **host-venv** production block that still keeps
`VLLM_USE_V2_MODEL_RUNNER=0` (**V2 blocked on Qwen4Exp**), `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, and
`--max-num-batched-tokens 2048`, but:

- uses `--compilation-config` **`FULL_AND_PIECEWISE`**. The in-tree launcher comment (Sep 18) says
  that mode **executes as PIECEWISE on ROCm** (`rocm_full_executes_as_piecewise`) and that an earlier
  “FULL_AND_PIECEWISE corrupts at c=8” report was **probe artifacts** (reasoning-parser field /
  reasoning-budget), **not** the graphs. Still do **not** use **FULL-only** —
  [FULL-graph corruption](../troubleshooting/vllm-flash-next.md#flash-next-full-graph-corruption)
- raises `--kv-cache-memory-bytes` to **7000000000** (7 GiB) and `--gpu-memory-utilization 0.90`
- reported a **16× 16k** concurrency test as fine (earlier `--max-num-seqs` tightness)
- **kills leftover processes** before relaunch — `VLLM::Worker`, `VLLM::EngineCore`, and
  `PleOffloadWorker` can survive an API/port kill ([troubleshooting](../troubleshooting/vllm.md#leftover-vllm-workers))
- custom all-reduce / PIX is **optional** — hosts without GPU↔GPU P2P should drop that block
  (vLLM falls back to PYNCCL) — [no P2P](../troubleshooting/vllm-flash-next.md#flash-next-no-p2p-or-mtp-blocks-ple)

Hub **`v0.28.0-extras`** is the container to A/B against this host-venv line; it is **not** yet the
wiki default.

```bash
MODEL="wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16"
export VLLM_PLE_QUANT_DIR="$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('primitive-ai/Qwen3.8-Flash-Next-PLE-quant'))")/ples_int4"

export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_FORCE_CUSTOM_ALL_REDUCE=1   # only if GPU↔GPU P2P works
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1   # compile off; A/B 0 after output is clean
export VLLM_RDNA_DENSE_GEMV=1           # if wvSplitK faults after profile
export VLLM_RDNA_FUSED_HC=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export PYTORCH_TUNABLEOP_FILENAME="${PYTORCH_TUNABLEOP_FILENAME:-/path/to/tunableop_results.csv}"
export VLLM_BATCH_INVARIANT=0
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=pix
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

cd /tmp   # avoid sys.path collisions with a local vllm checkout
python -m vllm.entrypoints.cli.main serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --max-num-seqs 6 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 5000000000 \
  --gpu-memory-utilization 0.88 \
  --dtype float16 \
  --trust-remote-code \
  --enable-prefix-caching \
  --language-model-only \
  --skip-mm-profiling \
  --enable-expert-parallel \
  --block-size 16 \
  --distributed-timeout-seconds 1800 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","compile_ranges_endpoints":[]}'
```

Shared P2P / all-reduce env is the same as
[Hub `-extras` TP4](./recipes.md#hub--extras-tp4-qwen38-27b-awq). Drop the custom-AR block if P2P is broken.

**Concurrency vs batch:** `#vllm-rdna` (Sep 16): `--max-num-batched-tokens 2048` keeps prefill
turns ~**1 s** at ~1300 tok/s; **8192** can block other streams for ~**6 s**. Raise toward **4096**
only if you hit the [128k prefill cliff](../troubleshooting/vllm-flash-next.md#flash-next-128k-prefill-cliff)
on Intel AutoRound — that 4096 knob and this PIECEWISE stability recipe are **different bugs**.

Vision (`#vllm-rdna` Sep 17–21 + the in-tree launcher): **on by default** in
[`serve_gfx1030_flashnext.sh`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/scripts/serve_gfx1030_flashnext.sh)
and **requires a pixel cap**. `--limit-mm-per-prompt '{"image":1}'` alone is **not** enough (count
is already 1). Without `max_pixels`, mm-profiling can feed a huge dummy image (~24.8M px) and the
vision encoder SDPA math backend materializes a **~64 GiB** L×L fp32 score matrix — **startup OOM**
on 32 GB cards. `max_pixels=1605632` keeps images up to about **1424×1424**.

```bash
--limit-mm-per-prompt '{"image":1}' \
--mm-processor-kwargs '{"max_pixels":1605632}'
```

`#vllm-rdna` (Sep 21): vision **works** on `rdna_extras` with those flags; community **~15–20 s per
image**. Runtime is a second OOM: vLLM may **not reserve** vision memory, so a **tight KV** plus an
image can OOM after a clean start. `#general` (Sep 21): vision + **PP3** is especially tight. Keep
`--language-model-only --skip-mm-profiling` if you do not need images.

Maintainer QA playbook (not a second wiki): [`BlivionIaG/vllm-rdna-qa`](https://github.com/BlivionIaG/vllm-rdna-qa).

### 3× V620 Flash-Next PP3 + MTP (community, Needs verify) {#flash-next-3x-pp3-mtp}

`#vllm-rdna` (Sep 15): one host fitted **Qwen3.8-Flash-Next + MTP** on **3× V620 32 GB** using a
**leapdragon** image (W4A16 + PLE int4 CPU offload), `PP=3` / `TP=1`. Treat as a **single-host**
write-up — some steps were **local source patches**, not a published tag.

Operational facts that are safe to copy:

- Baseline **PP3 without MTP** already sat at **~30 GiB/card** with `--gpu-memory-utilization 0.95`.
- The MTP head (`model_mtp.safetensors`, unquantized **fp16 MoE**, ~**5 GB**) is instantiated on the
  **last pipeline stage only**, so that stage OOMs first.
- Uneven layers: `VLLM_PP_LAYER_PARTITION=17,18,13` (48 layers) to leave the last stage light. The
  **PLE offload worker** can inherit that env, see `pp_size=1`, and refuse
  (`len(partitions)=3` ≠ `pp_size=1`) — **unset** `VLLM_PP_LAYER_PARTITION` in the PLE worker process.
- `VLLM_ROCM_MOE_PADDING=0` (do not pad routed-expert weights).
- Serve knobs on that host: `--enforce-eager`, context **32768**, `--max-num-seqs 4`,
  `--max-num-batched-tokens 2048`.
- Upstream [`vllm#46994`](https://github.com/vllm-project/vllm/pull/46994) (MTP under pipeline
  parallel on the V2 runner) merged days earlier — expect it in a **future** vLLM release, not Hub
  `-extras` 0.27.1.

Community also rewrote `VLLM_RDNA_DENSE_INT8` shadow quant to **row blocks** so a 2.4 GiB fp32
temporary would not OOM a nearly full card, then `VLLM_RDNA_DENSE_INT8_ONLY=1` to drop fp16 copies
(~2 GB/rank). Those int8 changes were **local** — do not assume they are in `rdna_extras` HEAD.

If an older `rdna_extras` pin **corrupts** PP3 decode, stay on a known-good leapdragon container —
[PP3 corruption](../troubleshooting/vllm-flash-next.md#flash-next-pp3-output-corruption). Sep 19 pin
`b33f9b6` is a different failure (graphs boot, small prefill `KeyError`) —
[workaround](../troubleshooting/vllm-flash-next.md#flash-next-pp3-graph-keyerror).

### 3× overlay: leapdragon + org MoE HIP (Sep 17–18) {#flash-next-3x-moe-hip-overlay}

`#vllm-rdna` (Sep 17–18): a **temporary public overlay**
([`alanoo81/flashnext-v620-pp3`](https://github.com/alanoo81/flashnext-v620-pp3)) ports
`opengfx1030`'s fused **W4A16 MoE HIP** kernel (`moe_gptq_gemm_rdna2`) onto a **leapdragon**
Flash-Next image so **3× V620 / PP=3** can keep leapdragon CUDA graphs + MTP while picking up
org-class prefill. **Not** a published Hub tag; treat as **Community / Needs verify**.

Public README snapshot (same harness; 160 W cap; AWQ-W4A16 + int4 PLE in host RAM):

| Stack | Prefill 4K / 16K / 64K / 130K | Decode |
|---|---|---|
| leapdragon image as shipped, graphs, PP3 | 409 / 738 / 1349 / — | ~32 t/s (~48 with dense int8) |
| leapdragon + org MoE HIP + graphs + MTP k=2 + prefix cache | **1078 / 1863 / 1989 / 2493** | **~57–63 t/s** |

Community: use **MTP for one user**; from **two streams** up, leave MTP off (plain graphs scaled
better in that write-up). Also backports [vLLM #46994](https://github.com/vllm-project/vllm/pull/46994)
and [#54044](https://github.com/vllm-project/vllm/pull/54044) (MTP + graphs + prefix cache). Follow
the overlay README — do not copy host paths from Discord.

`#general` (Sep 23): the overlay author is **no longer working on the repo**. Community still
reports useful 3-card numbers (one host **~45 t/s** decode / **~1100 tok/s** PP; overlay write-up
**120+ t/s** under multi-agent load, even **~200 t/s** with 5–6 agents). Long context can trip
**infinite KV-pool dump loops** — the author capped **3 concurrent agents** on long prompts.
Treat the overlay as **unmaintained / snapshot**; prefer `rdna_extras` once PP3 is clean there.

`#vllm-rdna` (Sep 19): that same **HIP MoE** kernel was reported **non-deterministic** (~3.5% of
top tokens change between identical runs; Triton MoE did not). See
[fork MoE note](fork.md#moe-mixture-of-experts). **Needs verify.**

`#vllm-rdna` (Sep 18): the published Flash-Next checkpoint can leave the **MTP draft head's 512
experts in bf16**, so speculative decode runs those through the generic **Triton MoE** kernel.
The overlay README adds `scripts/quant_mtp_experts.py` (offline **W4A16**, RTN int4 g128, same
packing as the main experts) so the draft experts hit the **same HIP kernel**. Community on
**3× V620 PP3**: **+4–7%** single-stream decode (~65–66 t/s), no acceptance loss (~70→75%), and
**+60%** wall throughput at 8 streams (~77 → ~123 t/s). **Community / Needs verify** — run the
script from the overlay repo; do not invent a Hub tag for the derived checkpoint.

### DeepSeek-V4 Flash (community, Needs verify) {#deepseek-v4-flash}

`#vllm-rdna` (Sep 17): public INT4-W4A16 weights
[`yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16`](https://huggingface.co/yiminyuan/DeepSeek-V4-Flash-0731-INT4-W4A16)
plus a gfx1030 vLLM tree
[`yiminyuan/vllm` @ `gfx1030/v0.28.0`](https://github.com/yiminyuan/vllm/tree/gfx1030/v0.28.0).
Publisher intent: **TP=4 / PP=2**. This is **not** Hub `-extras` and **not** wiki-tested. `#llamacpp`
also mentioned a RDNA2 TP4 DeepSeek-V4 serve (~22 t/s, kernel unpublished) — **Needs verify**.

---

## Related

- [Flash-Next status](./flash-next.md) — throughput, KV, AutoRound
- [Hub `-extras` TP4](./recipes.md#hub--extras-tp4-qwen38-27b-awq) — the non-Flash-Next 27B block
- [Configuration](./configuration.md)
- [vLLM forks](./fork.md)

