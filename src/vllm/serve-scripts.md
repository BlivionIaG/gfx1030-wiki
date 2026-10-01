# Fork serve scripts

> **WIP / fork-source.** See [Verification status](../reference/verification.md#serve-scriptsmd).
> Knobs below are the in-tree launcher on
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras`
> (`scripts/serve_rdna.sh`, `scripts/recipes/*.env`, `tunableop/README.md`).
> Build the venv first: [Host venv](host-venv.md). Card-count picks stay on
> [Recipes](recipes.md). Flash-Next graph history stays on
> [Serve lines](flash-next-serve.md).

The fork ships **one** serve entry point. A recipe file holds only that config’s
deltas. The script owns the venv, ROCm library path, TunableOp profile, GPU
selection, cache roots, stale-worker teardown, and the `vllm serve` argv.

Run it from a checkout of `rdna_extras`:

```sh
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  bash scripts/serve_rdna.sh RECIPE=<name> [KEY=value …]
```

`MODEL` and `VENV` (or an already-activated `$VIRTUAL_ENV`) are required. Nothing
in the script or the recipes bakes in a home directory, host, or IP.

`RECIPE` is a **script argument** (`RECIPE=name` on the command line). `MODEL`
and `VENV` are environment variables; a trailing `KEY=value` can set them too.
Precedence, lowest to highest: **recipe file → caller environment → trailing
`KEY=value`**.

The launcher defaults `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Pass a
**local checkpoint** in `MODEL`. To pull a Hub id on this run, export
`HF_HUB_OFFLINE=0` and `TRANSFORMERS_OFFLINE=0` in the caller environment.

## Recipes

Files live in [`scripts/recipes/`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/scripts/recipes).
Copy one and change only the deltas to add a config.

| Recipe | Model / config | Defaults worth knowing |
|---|---|---|
| `flashnext-mtp2` | Flash-Next AWQ-W4A16, **MTP=2**. Recipe note (27 Sep 2026): MTP=2 wins at **8×16k / 1k, c=8**. | TP=4, port **18096**, API server, `MAXLEN=262144`, block size **1024**, KV **4026531840**, `COMPILE_MODE=3`, `FULL_AND_PIECEWISE`. Requires `VLLM_PLE_QUANT_DIR`. |
| `flashnext-mtp0` | Same family, **MTP=0**. Recipe note: MTP=0 wins at **8×1k / 512, c=8** (short prompts). | Same ports and graphs as `flashnext-mtp2`, with `MTP=0`. |
| `flashnext` | Flash-Next **production** stack: V1 runner, tools + reasoning, six in-flight sequences. Separate from `flashnext-mtp0`. | Port **18094**, `--host 0.0.0.0`, CLI entry, KV **7000000000**, `gpu-memory-utilization 0.90`, block size 16. `ATTN=none` (FA is `VLLM_USE_RDNA2_FA=1`, no `--attention-backend`). Requires `VLLM_PLE_QUANT_DIR`. |
| `27b-awq` | Qwen3.8-27B AWQ INT4, compressed-tensors **W4A16**. `W4A8=1` opts into `VLLM_RDNA2_W4A8_SDOT4`. | Port **18210**, MTP=0, KV **8000000000**, no `MAXLEN` (the checkpoint’s own context). `language-model-only`. |
| `27b-exl3` | Qwen3.8-27B EXL3 **3.00bpw mul1**. **Eager** bring-up (`EAGER=1`): no torch.compile / CUDA graphs until you turn that off. | Port **18105**, `MAXLEN=20480`, KV **6000000000**. `HIP_VISIBLE_DEVICES=4,5,6,7` (second PLX on an 8-GPU chassis). On a 4-card host set `HIP_VISIBLE_DEVICES=0,1,2,3`. |
| `full` | The old `serve_gfx1030_full.sh` config (FA-RDNA2 + W4A16 + HIP KV + HIP GDN, breakable `FULL_AND_PIECEWISE`). | TP=**2**, devices `0,1`, `MAXLEN=200000` (floor **32768**), KV **7000000000**. `HOST` is empty so vLLM binds its default address. |

Flash-Next recipes need the int4 PLE sidecar
(`VLLM_PLE_QUANT_DIR=…/ples_int4`). See [Flash-Next serve lines](flash-next-serve.md).

Recipes that omit `HIP_VISIBLE_DEVICES` use the harness default **`0,1,2,3`**
(override with `GPUIDS` or `HIP_VISIBLE_DEVICES`).

## Dry-run

`PRINT=1` (aliases: `DRY_RUN=1`, `--dry-run`, `-n`) prints the recipe path, the
TunableOp profile it would select, the `export` block, and the exact command,
then exits. It does not launch vLLM. Print mode still requires `MODEL` and
`VENV`, and it skips the check that `$VENV/bin/python` exists.

```sh
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  PRINT=1 bash scripts/serve_rdna.sh RECIPE=27b-exl3
```

## Examples

Flash-Next speculative decode (MTP=2). Long context is where this recipe’s own
note says MTP=2 wins:

```sh
MODEL=/path/to/flash-next VENV=/path/to/venv \
VLLM_PLE_QUANT_DIR=/path/to/ples_int4 \
  bash scripts/serve_rdna.sh RECIPE=flashnext-mtp2
```

27B AWQ at MTP=2, with port and KV overrides:

```sh
MODEL=/path/to/qwen3.8-27b-awq VENV=/path/to/venv \
  bash scripts/serve_rdna.sh RECIPE=27b-awq MTP=2 KV=8000000000 PORT=18210
```

EXL3 stays eager unless you clear that flag. `CG_MODE=FULL_AND_PIECEWISE` is
already the recipe default; it applies once `EAGER=0`:

```sh
MODEL=/path/to/qwen3.8-27b-exl3 VENV=/path/to/venv \
  bash scripts/serve_rdna.sh RECIPE=27b-exl3 EAGER=0 CG_MODE=FULL_AND_PIECEWISE \
  HIP_VISIBLE_DEVICES=0,1,2,3
```

Disable TunableOp lookups for an A/B:

```sh
MODEL=/path/to/flash-next VENV=/path/to/venv \
VLLM_PLE_QUANT_DIR=/path/to/ples_int4 \
  bash scripts/serve_rdna.sh RECIPE=flashnext-mtp2 TUNABLEOP=0
```

## Option map

| Knob | Role |
|---|---|
| `TP` | `--tensor-parallel-size` (harness default 4). |
| `PORT` / `HOST` | Bind address. Harness defaults **18094** and **127.0.0.1**. An empty `HOST` (the `full` recipe) omits `--host`. |
| `MTP` | `0` skips spec decode. `1+` adds MTP with `num_speculative_tokens` and local-argmax reduction. Capture sizes become `(MTP+1) × {1,2,4,8}` unless `CG_SIZES` is set. |
| `ATTN` | `fa` → `--attention-backend RDNA_ATTN`. `triton` → `TRITON_ATTN`. `none` omits the flag (the recipe may still set `VLLM_USE_RDNA2_FA`). |
| `MAXLEN` | `--max-model-len`. `full` also enforces `MIN_MAXLEN` (default floor 32768). |
| `BLOCK_SIZE` | `--block-size`. |
| `SEQS` | `--max-num-seqs` (default 8). |
| `MAXBAT` | `--max-num-batched-tokens` (default 2048). |
| `LPTH` | `--long-prefill-token-threshold`. |
| `PREFILL_INTERVAL` | `--prefill-schedule-interval`. |
| `KV` | `--kv-cache-memory-bytes`. **Required** (a recipe with no KV budget aborts). |
| `GMEM` | `--gpu-memory-utilization`. |
| `COMPILE_MODE` | Compilation `mode` inside `--compilation-config`. |
| `CG_MODE` | `cudagraph_mode` (default `FULL_AND_PIECEWISE`). |
| `CG_CAPTURE_SIZES` | `1` (default) writes `cudagraph_capture_sizes` into the compile JSON. `0` leaves capture sizes to vLLM. |
| `CG_SIZES` | Replaces the capture-size list itself (for example `[1,2,4,8]`). |
| `EAGER` | `1` passes `--enforce-eager` and skips `--compilation-config`. |
| `TUNED_CONFIG` | `1` points `VLLM_TUNED_CONFIG_FOLDER` at `<tree>/tuned-moe`. |
| `W4A8` | Maps to `VLLM_RDNA2_W4A8_SDOT4` (`27b-awq` default `0`). |
| `RDNA_AR` | Maps to `VLLM_RDNA_AR`. |
| `HIP_VISIBLE_DEVICES` | GPU set. Alias: `GPUIDS`. |
| `TUNABLEOP` | `1` (default) loads a profile. `0` sets `PYTORCH_TUNABLEOP_ENABLED=0`. |
| `FEATURES` | Space-separated tokens (table below). |
| `EXTRA_ARGS` | Extra words appended to the vLLM command. |

Legacy names the entry point still accepts: `MODEL`←`MODEL_PATH`,
`KV`←`KVCAP`←`KV_CACHE_MEMORY`, `SEQS`←`MAX_NUM_SEQS`,
`MAXBAT`←`MAX_BATCHED_TOKENS`, `MAXLEN`←`MAX_MODEL_LEN`, `GMEM`←`GPU_MEM`.

`FEATURES` tokens the script knows:

| Token | Flag |
|---|---|
| `prefix-caching` | `--enable-prefix-caching` (skipped when `ENABLE_PREFIX_CACHING=0`) |
| `language-model-only` | `--language-model-only` |
| `skip-mm-profiling` | `--skip-mm-profiling` |
| `mamba-align` | `--mamba-cache-mode align` |
| `reasoning` | `--reasoning-parser qwen3` |
| `auto-tool` | `--enable-auto-tool-choice --tool-call-parser qwen3_coder` |
| `expert-parallel` | `--enable-expert-parallel` |
| `trust-remote-code` | `--trust-remote-code` |
| `prompt-tokens-details` | `--enable-prompt-tokens-details` |
| `vision-cap` | `--limit-mm-per-prompt '{"image":1}'` and `--mm-processor-kwargs '{"max_pixels":1605632}'` |

An unknown token aborts. `vision-cap` is how Flash-Next recipes keep the pixel
cap that prevents the vision-encoder startup OOM — see
[Flash-Next serve lines](flash-next-serve.md).

## TunableOp

Rows are **named profiles**, one folder per rocBLAS build, registered in
[`tunableop/profiles.json`](https://github.com/opengfx1030/vllm-rdna/blob/rdna_extras/tunableop/profiles.json).
Solver IDs are build-specific. The helper
(`tools/rdna2_028/tunableop_env.sh`) selects the folder whose
`lib_sha256[:12]` matches the loaded `librocblas.so.5`.

| Profile | rocBLAS / torch | `librocblas.so.5` sha256[:12] | Rows / rank |
|---|---|---|---|
| `rocm7.14-rocblas5.5/` | 5.5.0 / `2.12.0+rocm7.14.0` | `f30bb442e9b5` | **783** (canonical). Covers Flash-Next MTP=0/2, 27B AWQ, and EXL3 27B. |
| `rocm10-rocblas5.6/` | 5.6.0 / `2.13.0+rocm10.0.0` | `c27e2252cc7a` | **70**, thin. Those shapes are tuned; every other GEMM uses the rocBLAS heuristic. |

- Auto-select is the default. No recipe or launcher change is required when the venv’s `librocblas.so.5` hash changes. A ROCm 7.14 start logs `TunableOp profile rocm7.14-rocblas5.5 auto-selected for rocBLAS build f30bb442e9b5`.
- `TUNABLEOP_PROFILE=<name>` forces a registered profile. A hash mismatch **fails the launch**. `TUNABLEOP_ALLOW_MISMATCH=1` is the experiment-only override (hard warning, then it uses the rows anyway). Solver IDs are build-specific, so a mismatched profile is never applied silently.
- Lookup is read-only (`PYTORCH_TUNABLEOP_TUNING=0`). The helper does not write rows to `/tmp` or the process CWD. If no profile matches, it uses a legacy `rocblas-<hash>/` folder when that folder exists, then `$HOME/.cache/tunableop/tunableop_results.csv`.
- `TUNABLEOP=0` turns lookups off (`PYTORCH_TUNABLEOP_ENABLED=0`) for an A/B against the heuristic.
- `PRINT=1` reports which profile it would pick, including “NOT registered” if `TUNABLEOP_PROFILE` is unknown.

### ROCm 10 and rocBLAS 5.6

A venv whose torch loads rocBLAS **5.6** (`librocblas.so.5` sha256[:12] `c27e2252cc7a`) selects `rocm10-rocblas5.6` by that hash. Same recipe, same script:

```sh
MODEL=/path/to/checkpoint VENV=/path/to/rocm10-venv \
  bash scripts/serve_rdna.sh RECIPE=27b-awq

MODEL=/path/to/checkpoint VENV=/path/to/rocm10-venv \
  PRINT=1 bash scripts/serve_rdna.sh RECIPE=27b-awq
```

Print mode (library present in that venv) reports:

```text
# tunableop: rocm10-rocblas5.6 (auto-selected for librocblas.so.5 c27e2252cc7a)
```

A real launch logs `TunableOp profile rocm10-rocblas5.6 auto-selected for rocBLAS build c27e2252cc7a.`

**Thin** means the shipped CSV covers **70 shapes per rank**. Those GEMMs use the recorded solver. Every other shape falls back to the rocBLAS heuristic: safe, and untuned. The validated serving numbers on this wiki were captured on **gfx1030 + ROCm 7.14** (`rocm7.14-rocblas5.5`, 783 rows). Recipe knobs (`TP`, `KV`, `MTP`, graphs) do not depend on the ROCm version. Treat the first ROCm 10 boot as bring-up: run `PRINT=1` and confirm the profile line before leaving it up.

To fill the gap on that machine, tune there and register the result. The in-tree pipeline keeps scratch under `<tree>/cache/tunableop-rows` (never `/tmp`) and does not overwrite the repo profile:

```sh
VENV=/path/to/rocm10-venv \
  bash tools/rdna2_028/tunableop_rows_pipeline.sh tune
```

Curate with `tools/rdna2_028/curate_tunableop_rows.py`, freeze the CSVs into `tunableop/<profile-name>/`, and add that name to `profiles.json` with this library’s sha256. Then prove the lookup actually hits:

```sh
python tools/rdna2_028/verify_tunableop_lookup.py \
  --rows tunableop/<profile-name>/tunableop_results0.csv \
  --out hits.json
```

A hit means the solver TunableOp recorded equals the shipped row. Contributing that profile back is how the 70-row set grows. The older qualify flow is still documented under [Regenerate for another rocBLAS build](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/tunableop#regenerate-for-another-rocblas-build).

Guardrails for a ROCm 10 (or any other) library:

| Situation | What happens |
|---|---|
| Loaded sha is `c27e2252cc7a` | Auto-selects `rocm10-rocblas5.6`. |
| Loaded sha matches neither profile | Legacy `tunableop/rocblas-<sha12>/` if that folder exists, otherwise `$HOME/.cache/tunableop/tunableop_results.csv`. The 7.14 rows are not loaded. |
| `TUNABLEOP_PROFILE=rocm10-rocblas5.6` but the loaded library is a different sha | Launch **refuses to start**. |
| `TUNABLEOP_ALLOW_MISMATCH=1` | Proceeds with a hard warning. Experiments only. |
| `TUNABLEOP=0` | Lookups off. |

Mismatch symptoms and the historical ~25% prefill gap:
[TunableOp troubleshooting](../troubleshooting/vllm.md#tunableop-rocblas-mismatch).

## Foreground, aliases, stale workers

`serve_rdna.sh` **execs in the foreground**. Background it yourself and poll
readiness:

```sh
MODEL=/path/to/checkpoint VENV=/path/to/venv \
  nohup bash scripts/serve_rdna.sh RECIPE=27b-awq >serve.log 2>&1 &
# port follows the recipe (27b-awq is 18210)
curl -sf http://127.0.0.1:18210/v1/models
```

Before `exec`, the harness signals processes whose environment has the same
`VLLM_CACHE_ROOT` (API server, `VLLM::Worker`, `VLLM::EngineCore`,
`PleOffloadWorker`). The default cache root is `<tree>/cache/vllm`. A second
serve that shares that root will tear the first one down.

The old filenames are thin aliases. They forward the environment and any
trailing `KEY=value`:

| Old script | Calls |
|---|---|
| `scripts/serve_gfx1030_flashnext.sh` | `RECIPE=flashnext` |
| `scripts/serve_gfx1030_flashnext_mtp.sh` | `RECIPE=flashnext-mtp2` |
| `scripts/serve_gfx1030_27b_dense.sh` | `RECIPE=27b-awq` |
| `scripts/serve_gfx1030_exl3_27b.sh` | `RECIPE=27b-exl3` |
| `scripts/serve_gfx1030_full.sh` | `RECIPE=full` |

## Related

- [Host venv](host-venv.md) — build the tree this script expects
- [Recipes](recipes.md) — Hub image vs recipe container vs this checkout
- [Flash-Next serve lines](flash-next-serve.md) — graph modes and PLE
- [Environment variables](../reference/env-vars.md)
- [TunableOp mismatch](../troubleshooting/vllm.md#tunableop-rocblas-mismatch)
