# Flash-Next serve lines

> **WIP / community.** Copy from here only after you have read [Flash-Next status](./flash-next.md)
> (which card count, which graph mode). Failures: [Flash-Next troubleshooting](../troubleshooting/vllm-flash-next.md).

## Path C — Flash-Next (`rdna_extras` / Hub 0.31)

For agentic / long-context Flash-Next on **4× V620**, use the Flash-Next vLLM stack — not the
27B recipe presets and **not** the old `v0.27.1-extras*` / `v0.28.0-extras*` Hub tags:

- **Source of truth:** [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras`
  is the **v0.31.0** line (`#announcements` 11 Oct 2026). The old 0.28 aggregation lives at
  [`rdna_extra/v0.28.0`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extra/v0.28.0).
  Hub **`blivioniag/vllm-rdna:v0.31.0-extras`** (ROCm 7.14) served Flash-Next MTP-2 (BF16 + INT4-disk).
- **In-tree launcher:** [`tools/rdna/serve_rdna.sh`](serve-scripts.md) `RECIPE=flashnext` (production tools/reasoning) or `RECIPE=flashnext-mtp2` / `flashnext-mtp0`. The old `tools/rdna/serve_gfx1030_flashnext.sh` name is a thin alias of `RECIPE=flashnext`. `#vllm-rdna` Sep 21 pointed 2-card / GPTQ questions at that file. Do **not** copy host paths out of the tree. Prefer the recipe over agent-generated env dumps — drop `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and do **not** force `VLLM_RDNA_FUSED_HC=0` on Flash-Next (`#vllm-rdna` Oct 10).
- **Published container:** prefer Hub **`v0.31.0-extras`**. The older
  [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next)
  tree remains for legacy GHCR / `docs/rdna2` (occasional commits through 11 Oct 2026).
- Weights: [`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16).
  `#vllm-rdna` (Oct 11): [`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4)
  produced **incoherent** output on one 4× host; switching to the `wtdcode` AWQ + primitive-ai PLE
  pair is the documented path.
- PLE / n-gram: launcher `PLE=bf16|int4|int4-disk` + `PLE_QUANT_DIR=…/ples_int4`
  ([`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant)).
  `#announcements` (11 Oct): all **three** modes tested; speed at parity. Docker disk mode:
  `-v /path/to/ples_int4:/ples_int4 -e VLLM_PLE_QUANT_DIR=/ples_int4 -e VLLM_PLE_CPU_OFFLOAD=1`.
  No-P2P hosts: `P2P=0` on the launcher (or `VLLM_RDNA_AR=0`).

Expect large **host DRAM** for the BF16 n-gram table (~**95 GB** class). INT4 pinned is a
**~30 GB** table; `#vllm-rdna` (Oct 11) corrected an inflated **~90 GB** host-RAM figure — INT4
sidecar working set is closer to **~35 GB**. `int4-disk` mmaps shards (`MADV_RANDOM`) so only the
gather/dequant working set hits RAM. That is **weight** offload, not KV. Native **KV RAM overflow**
was broken on one 4× host (Sep 14: **128 GB** not enough); the
fix **merged** as [`#24`](https://github.com/opengfx1030/vllm-rdna/pull/24) (27 Sep) —
[KV offload](../troubleshooting/vllm.md#upstream-kv-offload-tanks-decode).
Long-prompt / intermittent stalls: try `VLLM_USE_V2_MODEL_RUNNER=0` —
[troubleshooting](../troubleshooting/vllm-flash-next.md#flash-next-long-prompt-stalls).
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
  prefill chunk. Precision stays W4A16. `#vllm-rdna` (Sep 29): a latest-`rdna_extras` host saw PP
  fall from ~**2k → ~1450** with this **unset**; setting `1` restored ~2k. Treat a sudden PP cliff
  after an extras pull as a missing-resident-MoE check before blaming kernels.
- **Mamba retirement:** backport of [`vllm#55450`](https://github.com/vllm-project/vllm/pull/55450).
  Without it, long prompts (100k+ class) hit a deterministic **preemption loop** (state blocks not
  retired across null gaps).
- **Graphs (`#17` measured):** `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12]}'`
  with **MTP-2**. Prefill stays eager.
- On **current HEAD**, prefer **`FULL_AND_PIECEWISE`** instead — see
  [FULL_AND_PIECEWISE](#flash-next-4x-full-and-piecewise) (decode uses the same FULL path as this
  baseline; piecewise covers mixed/prefill).
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

### 4× V620 Flash-Next serve (`FULL_AND_PIECEWISE`, Sep 24–25) {#flash-next-4x-full-and-piecewise}

**Current recommended graphs on latest `rdna_extras`.** `#vllm-rdna` (Sep 24): maintainer confirmed
full + piecewise is fixed. Fork-source write-up:
[`docs/rdna2/V620-FULL-AND-PIECEWISE.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/docs/rdna2/V620-FULL-AND-PIECEWISE.md).

What changed:

| Layer | Behavior |
|---|---|
| **Uniform decode** | Replays the **FULL** CUDA graph (same live capture as `#17` `FULL_DECODE_ONLY`) |
| **Mixed / prefill** | Replays **piecewise** graphs |
| **`#20` alone** | Piecewise capture worked, but ROCm still redirected every FULL dispatch → piecewise → decode **~26–34 t/s** |
| **Keep-FULL fix** | Commit [`6c26c78d`](https://github.com/opengfx1030/vllm-rdna/commit/6c26c78d54) (`#21` / `rocm_full_executes_as_piecewise → False`) — on `rdna_extras` HEAD |

Safe compile line (generalized; do not copy host paths):

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[3,6,12]}'
```

In-tree launcher: [`tools/rdna2/serve_v620_piecewise.sh`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/tools/rdna2/serve_v620_piecewise.sh)
(`V620_COMPILE_MODE=3`, `V620_CUDAGRAPH_MODE=FULL_AND_PIECEWISE`). Older comments in that script
still mention the `#20`-era ~2× decode drop — ignore those once you are on HEAD with the keep-FULL
commit. Prefer this over bare `PIECEWISE` (no FULL graphs) and over the Sep 17 breakable/`mode:0`
workaround.

Without compilation / without `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, a `FULL_AND_PIECEWISE` request
**downgrades to `FULL_DECODE_ONLY`** (keeps FULL decode) instead of dropping every graph; a bare
`PIECEWISE` request still becomes `NONE`.

`rdna_extras` HEAD (25 Sep, `700753d9`): **MTP-2 boots** after the amdsmi / local-argmax draft
fix, but author-host decode stays **below MTP-0** (~46–49 vs ~61 t/s). Do not expect MTP to beat
the no-draft line yet — [MTP](quantization.md#mtp-speculative-decoding).

The older Sep 17 PIECEWISE host-venv block below is only for checkouts **before** `#17` / the
keep-FULL fix.

### 4× V620 Flash-Next MonoKernel (`#94`, 16 Oct) {#flash-next-monokernel}

**Merged** [`opengfx1030/vllm-rdna#94`](https://github.com/opengfx1030/vllm-rdna/pull/94) (16 Oct)
into `rdna_extras`. A MonoKernel is a **persistent** HIP kernel that runs a large part of one
Flash-Next (`qwen4_exp`) decoder layer in one launch (k1 hyper-connection preamble, GDN/QSA
decode cores, k2 FFN). Prefill, mixed batches, the MTP drafter, and TP all-reduce stay on
today’s path. **Off by default.** Hub **`v0.31.0-extras` lags**. Recipes do **not** pass the
flag yet.

Startup **refuses** `--monokernel` unless all of these hold: gfx10x GPU, Qwen4Exp (MTP
drafter accepted), **V2 model runner**, **PP=1**, `_rocm_C` ops built, and MoE on
**symmetric W4A16** (`CompressedTensorsWNA16RDNA2MoEMethod`). Asymmetric experts are a
follow-up (open `#115`). PP3 MonoKernel is open backlog (`#116`) — do not combine this
switch with `RECIPE=flashnext-pp3*`.

```sh
# Build the in-tree HIP bits (from an rdna_extras checkout + venv):
bash tools/rdna/mono/build_tree.sh

MODEL=/path/to/flash-next VENV=/path/to/venv \
  EXTRA_ARGS="--monokernel" \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 \
  PLE=int4 PLE_QUANT_DIR=/path/to/ples_int4
```

Same switch: `VLLM_ROCM_MONO_DECODE=1` (alias of `--monokernel`; backport of
[`vllm#61082`](https://github.com/vllm-project/vllm/pull/61082)). A/B without the flag in
the same session. Keep **`GPU_MAX_HW_QUEUES=4`** (Flash-Next TP4 recipes already set it —
[`#107`](https://github.com/opengfx1030/vllm-rdna/pull/107)); two queues share HIP queues
and skew ranks. In-tree notes: [`tools/rdna/mono/README.md`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/tools/rdna/mono/README.md)
and [`docs/rdna/monokernel/`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/docs/rdna/monokernel).
Reproduce A/B write-up: open [`#120`](https://github.com/opengfx1030/vllm-rdna/pull/120).

PR `#94` same-session A/B (4× V620, TP4, PLE int4, `FULL_AND_PIECEWISE`, temperature 0,
`HSA_FORCE_FINE_GRAIN_PCIE=1`, `GPU_MAX_HW_QUEUES=4`) — **Fork-source**. Cells are
prefill tok/s/req / decode tok/s/req / TTFT s / median ITL ms. For MTP-2, prefer **ITL**
(decode tok/s moves with drafter acceptance):

| Recipe / cell | off | on | ITL |
|---|---|---|---|
| MTP-0 1k c=1 | 2034 / 44.97 / 0.50 / 22.2 | 2205 / 50.74 / 0.46 / 19.7 | **−11.4%** |
| MTP-0 1k c=8 | 524 / 25.84 / 1.95 / 35.1 | 574 / 28.56 / 1.78 / 31.7 | **−9.5%** |
| MTP-0 16k c=1 | 2096 / 44.38 / 7.82 / 22.5 | 2113 / 50.03 / 7.75 / 20.0 | **−11.2%** |
| MTP-0 16k c=8 | 1071 / 13.34 / 15.30 / 35.6 | 1083 / 13.97 / 15.13 / 32.6 | **−8.5%** |
| MTP-2 1k c=1 | 2053 / 58.88 / 0.50 / 28.6 | 2284 / 77.66 / 0.45 / 24.8 | **−13.4%** |
| MTP-2 1k c=8 | 905 / 29.47 / 1.13 / 61.0 | 1003 / 32.54 / 1.02 / 55.9 | **−8.3%** |
| MTP-2 16k c=1 | 2022 / 68.37 / 8.10 / 29.1 | 2040 / 104.58 / 8.03 / 25.5 | **−12.4%** |
| MTP-2 16k c=8 | 1397 / 13.92 / 11.73 / 62.3 | 1460 / 14.35 / 11.23 / 58.3 | **−6.4%** |

`#vllm-rdna` (16 Oct) author restated the decode-side win as about **8–10%** ITL at c=1
(with or without MTP), **5–6%** at c=8 without MTP, and **0–3%** at c=8 with MTP-2.
Prefill / TTFT were same or better in the PR table. Serves passed probe / PPL /
prefix / concurrency gates.

**Community / Needs verify** (`#vllm-rdna` 16–17 Oct): a second 4× host called MonoKernel
slightly **slower** than the current HIP path, with one GPU at **100% VRAM** and prefill
under **1200** tok/s on the same cells — TunableOp mismatch suspected; author-host prefill
stayed above **2.1k**. A mixed RDNA2/RDNA3 host saw about **5–10%** after adding the switch
(not a V620 recipe). If PP collapses after `--monokernel`, regenerate TunableOp before
blaming the kernel — [TunableOp](../troubleshooting/vllm.md#tunableop-rocblas-mismatch).

Merged [`#126`](https://github.com/opengfx1030/vllm-rdna/pull/126) (18 Oct) moves the
MonoKernel QSA indexer (store / epilogue / scoring) to HIP. That is **on** with
`--monokernel` on HEAD; it does not change the default path by itself.

### HEAD HIP defaults (`#130`, 19 Oct) {#flash-next-hip-defaults}

**Merged** [`#130`](https://github.com/opengfx1030/vllm-rdna/pull/130) (19 Oct) into
`rdna_extras` (closes `#99`). Flash-Next recipes on **HEAD** now default the HIP set
**on**: `VLLM_RDNA_ALL_HIP`, `VLLM_RDNA_QSA_HIP`, `VLLM_RDNA_HC_PREFILL_HIP`, and
`VLLM_GDN_HIP_PREFILL=1`. EXL3 recipes are unchanged. Hub **`v0.31.0-extras` lags**.
Set those vars to **0** to get the older Triton paths back.

Author-host 4× TP4 (Flash-Next AWQ, PLE int4): HIP-on ITL **−6 to −8%** and prefill
**+3–7%** vs the previous default (no `--monokernel`). With `--monokernel`, MTP-2 ITL
was **−2 to −4%**. Clone HEAD — do not expect this from the 11 Oct Hub bake.

Do **not** pass `num_speculative_tokens_per_batch_size` (merged `#127` opt-in).
`#vllm-rdna` (22 Oct): that schedule caused token-repeat loops. Fixed MTP-2
(`RECIPE=flashnext-mtp2`) stays the daily path. `#vllm-rdna` (25–26 Oct): loops can
**survive MTP off** if **CPU / disk KV offload** is restoring sessions — disable
`--kv-offloading-size` while isolating —
[troubleshooting](../troubleshooting/vllm-flash-next.md#flash-next-dynamic-mtp).

### 4× V620 Flash-Next DFlash2 (`#132`, 21 Oct) {#flash-next-dflash2}

**Merged** [`#132`](https://github.com/opengfx1030/vllm-rdna/pull/132) (21 Oct). Opt-in
drafter path, **not** a replacement for MTP-2.

```sh
MODEL=/path/to/flash-next VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-dflash2 \
  PLE=int4 PLE_QUANT_DIR=/path/to/ples_int4
```

- Target: [`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16).
- Draft: [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  (**BF16** — FP16 draft produced NaNs / near-zero acceptance).
- Draft attention is FA-RDNA2 `BLHNC` (noncausal sliding window). Target stays HIP QSA.
- Default **3** draft tokens (`[4,8,16,32]` capture). For the measured **16k / 1k c=8**
  ladder: `DFLASH_TOKENS=2` (`[3,6,12,24]`). `PRINT=1` shows the argv before launch.

PR paired A/B (4× V620, TP4, same boot): **native MTP-2 faster in all four cells**
(tuned DFlash2 aggregate output **−29 / −12 / −24 / −7%** at 1k c=1 / c=8 and 16k
c=1 / c=8). `#vllm-rdna` (22 Oct): maintainer still has MTP-2 ahead on agentic load;
community: DFlash2 can win short / single-user, MTP-2 under concurrency. **Fork-source**
for the recipe; community contrast **Needs verify**. Not Hub.

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

- uses `--compilation-config` **`FULL_AND_PIECEWISE`**. On **Sep 18** trees that mode still
  **executed as PIECEWISE on ROCm** (`rocm_full_executes_as_piecewise`). On **current HEAD** the
  keep-FULL fix turns that redirect off — [FULL_AND_PIECEWISE](#flash-next-4x-full-and-piecewise).
  Still do **not** use **FULL-only** on old trees —
  [FULL-graph corruption](../troubleshooting/vllm-flash-next.md#flash-next-full-graph-corruption)
- raises `--kv-cache-memory-bytes` to **7000000000** (7 GiB) and `--gpu-memory-utilization 0.90`
- reported a **16× 16k** concurrency test as fine (earlier `--max-num-seqs` tightness)
- **kills leftover processes** before relaunch — `VLLM::Worker`, `VLLM::EngineCore`, and
  `PleOffloadWorker` can survive an API/port kill ([troubleshooting](../troubleshooting/vllm.md#leftover-vllm-workers))
- custom all-reduce / PIX is **optional** — hosts without GPU↔GPU P2P should drop that block
  (vLLM falls back to PYNCCL) — [no P2P](../troubleshooting/vllm-flash-next.md#flash-next-no-p2p-or-mtp-blocks-ple)

Prefer Hub **`v0.31.0-extras`** (or a matching host-venv clone of `rdna_extras`) for this
serve line. Hub **`v0.28.0-extras`** is only for historical A/B against older Sep 2026 pastes.

```bash
MODEL="wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16"
export VLLM_PLE_QUANT_DIR="$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('primitive-ai/Qwen3.8-Flash-Next-PLE-quant'))")/ples_int4"

export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=3600
# Low-RAM hosts only: PLE_OFFLOAD_PREFAULT=0  (skips RAM prefault; ~25% slower — see env-vars)
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

Vision (`#vllm-rdna` Sep 17–21 + the in-tree launcher): Flash-Next recipes pass
the `vision-cap` feature, which still applies a **pixel cap**.
`--limit-mm-per-prompt '{"image":1}'` alone is **not** enough (count
is already 1). Without `max_pixels`, mm-profiling can feed a huge dummy image (~24.8M px) and the
vision encoder SDPA math backend materializes a **~64 GiB** L×L fp32 score matrix — **startup OOM**
on 32 GB cards. `max_pixels=1605632` keeps images up to about **1424×1424**.
[`serve_rdna.sh`](serve-scripts.md) bakes those two flags into `vision-cap`.

```bash
--limit-mm-per-prompt '{"image":1}' \
--mm-processor-kwargs '{"max_pixels":1605632}'
```

`#vllm-rdna` (Sep 21): vision **works** on `rdna_extras` with those flags; community **~15–20 s per
image**. Runtime is a second OOM: vLLM may **not reserve** vision memory, so a **tight KV** plus an
image can OOM after a clean start. `#general` (Sep 21): vision + **PP3** is especially tight. Keep
`--language-model-only --skip-mm-profiling` if you do not need images.

Maintainer QA playbook (not a second wiki): [`BlivionIaG/vllm-rdna-qa`](https://github.com/BlivionIaG/vllm-rdna-qa).

### 3× V620 Flash-Next PP3 (`rdna_extras` HEAD) {#flash-next-3x-pp3}

`#general` (13 Oct 2026) + merged [`#96`](https://github.com/opengfx1030/vllm-rdna/pull/96):
**PP3 / MTP-2** is in-tree on current `rdna_extras`. Hub **`v0.31.0-extras`** (11 Oct bake) **does
not** include these recipes — clone HEAD + [host venv](host-venv.md) +
[serve scripts](serve-scripts.md).

```sh
MODEL=/path/to/flash-next VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-pp3 \
  PLE=int4 PLE_QUANT_DIR=/path/to/ples_int4
```

MTP-2 needs a **W4A16 quantized drafter** (`tools/rdna/quant_mtp_experts.py`); bf16 draft experts
do not fit the last stage:

```sh
MODEL=/path/to/mtpq-model VENV=/path/to/venv \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-pp3-mtp2 \
  PLE=int4 PLE_QUANT_DIR=/path/to/ples_int4
```

Recipe facts (fork comments + `#general` 13 Oct maintainer paste):

| Recipe | Partition | SEQS / MAXBAT / GMEM | Notes |
|---|---|---|---|
| `flashnext-pp3` | `16,16,16` | 8 / 2048 / 0.95 | MTP=0, text-only, 262k, `KV=auto`. Dense INT8 **ONLY** (fp16 copy next to the shadow OOMs a stage). Vision **not supported** under PP3 (encoder on stage 0). |
| `flashnext-pp3-mtp2` | `17,17,14` | 4 / 2112 / 0.975 | Drafter on last stage. `GMEM 0.95` caps length ~188k on 30 GB cards. `MAXBAT 2048` admits only 3 of 4 at 16k. Use up to ~4 streams; beyond that MTP=0 has more aggregate. |

`#vllm-rdna` (17 Oct): later HEAD merges were **not re-benched** against this 13 Oct table —
treat the numbers as a snapshot, not a current HEAD guarantee.

Author-host 16k/1k (`#general` 13 Oct, **Community** — not wiki-tested):

| Recipe | c=1 PP / TG / TTFT / ITL | Concurrent |
|---|---|---|
| `flashnext-pp3` (MTP-0) | **1848** / **38.0** / 8.87 s / 26.4 ms | c=8: **771** (1589 agg) / **7.7** / 21.3 s / 63.4 ms |
| `flashnext-pp3-mtp2` | **1866** / **70.9** / 8.78 s / 37.0 ms | c=4: **1071** (3088 agg) / **10.1** / 15.3 s / 122 ms (all 4 running) |

Also merged [`#95`](https://github.com/opengfx1030/vllm-rdna/pull/95): int8 dense shadows are built in
**row chunks** so load no longer allocates a ~2.4 GiB fp32 temporary. PLE still lives on **stage 0**
— `PLE=bf16` wants ~95 GB host RAM; smaller hosts use `PLE=int4` / `int4-disk`.

Older Sep 15 leapdragon / overlay notes stay below as **historical**. If an older pin still
**corrupts** PP3 decode —
[PP3 corruption](../troubleshooting/vllm-flash-next.md#flash-next-pp3-output-corruption). Sep 19 pin
`b33f9b6` is a different failure (graphs boot, small prefill `KeyError`) —
[workaround](../troubleshooting/vllm-flash-next.md#flash-next-pp3-graph-keyerror).

### 3× V620 Flash-Next PP3 + MTP (leapdragon era, Needs verify) {#flash-next-3x-pp3-mtp}

`#vllm-rdna` (Sep 15): one host fitted **Qwen3.8-Flash-Next + MTP** on **3× V620 32 GB** using a
**leapdragon** image (W4A16 + PLE int4 CPU offload), `PP=3` / `TP=1`. Treat as a **single-host**
write-up — some steps were **local source patches**. Prefer the [HEAD recipes](#flash-next-3x-pp3)
when you can clone current `rdna_extras`.

Operational facts from that era that still match the HEAD recipes:

- Baseline **PP3 without MTP** sat at **~30 GiB/card** with `--gpu-memory-utilization 0.95`.
- The MTP head is instantiated on the **last pipeline stage only**, so that stage OOMs first.
- Uneven layers leave the last stage light. The **PLE offload worker** can inherit
  `VLLM_PP_LAYER_PARTITION`, see `pp_size=1`, and refuse (`len(partitions)=3` ≠ `pp_size=1`) —
  **unset** that env in the PLE worker process.
- `VLLM_ROCM_MOE_PADDING=0` (do not pad routed-expert weights).
- Upstream [`vllm#46994`](https://github.com/vllm-project/vllm/pull/46994) (MTP under pipeline
  parallel on the V2 runner) merged Sep 2026.

The old **local** int8 row-chunk rewrite is now [`#95`](https://github.com/opengfx1030/vllm-rdna/pull/95)
on HEAD. `VLLM_RDNA_DENSE_INT8_ONLY=1` is on both `flashnext-pp3*` recipes.

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
top tokens change between identical runs; Triton MoE did not). `#vllm-rdna` (Sep 30): an **fp32
deterministic** path was described as **opt-in** (throughput cost); no public flag name. See
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

`#vllm-rdna` (Oct 10): on the current extras / **v0.31** test line, **DeepSeek-V4** is expected to
run once mixed **prefill/decode stalls** are fixed (suspected **CUDA-graph capture range** too
narrow). That is **not** a serve recipe yet — stay on the yiminyuan tree or wait for the stall
fix. **GLM-5.3** on the same fork was asked and **not answered**. See
[mixed prefill](../troubleshooting/vllm.md#prefill-blocks-decode).

---

## Related

- [Flash-Next status](./flash-next.md) — throughput, KV, AutoRound
- [Serve scripts](./serve-scripts.md) — `RECIPE=flashnext-mtp2` and the other in-tree configs
- [Hub `-extras` TP4](./recipes.md#hub--extras-tp4-qwen38-27b-awq) — the non-Flash-Next 27B block
- [Configuration](./configuration.md)
- [vLLM forks](./fork.md)

