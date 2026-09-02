# RDNA2 fork — Serving

## Recommended env stack {#recommended-env-stack}

`HSA_OVERRIDE_GFX_VERSION=10.3.0` is **required** for the tested V620/`gfx1030` native profile.

Prefer this **short** stack (`#llamacpp`, Aug 2026). Older long lists of `GGML_HIP_GFX1030_*`
knobs are mostly redundant with the override and can **clash with the RCCL autotune path**:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_RDNA2_AUTO=1 \
GGML_HIP_SAFE_STATE_IO=1 \
GGML_TP_SHARDED_OUTPUT=1   # TP2+ only
```

- `GGML_HIP_SAFE_STATE_IO=1` — recommended default; mitigates a known ROCm FA crash class.
- `GGML_CUDA_ALLREDUCE=nccl` — community reports **+10% tgen / +20% prefill** on Qwen 122B when RCCL
  is healthy.
- Optional TP4 mode: `GGML_HIP_GFX1030_P2P_ALLREDUCE=auto-expanded` (topology-gated).
- If you see `internal AllReduce init failed (n_devices != 2)` or wild PP variance, strip custom
  `GGML_HIP_GFX1030_*` / P2P knobs back to the short stack and re-test on **ROCm 7.2.0 or 7.14.0**
  (not mid-7.2.x such as 7.2.4). See [Installing ROCm](../../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

## Launch

**TP2+** (four-GPU example):

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_RDNA2_AUTO=1 \
GGML_HIP_SAFE_STATE_IO=1 \
GGML_TP_SHARDED_OUTPUT=1 \
./build/bin/llama-server \
  -m /path/to/main.gguf \
  -ngl all \
  --split-mode tensor \
  --tensor-split 1,1,1,1 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080
```

**TP1** — omit tensor split and `GGML_TP_SHARDED_OUTPUT`:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_SAFE_STATE_IO=1 \
./build/bin/llama-server -m /path/to/main.gguf -ngl all --flash-attn on --host 0.0.0.0 --port 8080
```

### Batch / ubatch tips

For tensor-split prefill, `#llamacpp` often does better with **larger ubatch** than the old
`2048/256` bench defaults — roughly **~1024 ubatch per GPU** (e.g. TP4 → `--ubatch-size 4096`) while
keeping `--batch-size` ≥ ubatch. Small prompts may regress slightly; long prompts usually win.

Recent long-context community recipes commonly use `--batch-size 16384 --ubatch-size 1024` on TP4
(see [Benchmarks](../rdna2-benchmarks.md#long-context-community-sweeps-aug-2930-2026)).

### Host tips that affect llama-server

- **CPU governor:** if any hot path stays on the host (Flash-Next n-gram tables, `--override-tensor …=CPU`,
  MoE/KV offload), `powersave` can lag bursty PP. Community: switching Intel `intel_pstate` to
  `performance` improved Flash-Next PP ~33% while VRAM-resident 27B was unchanged. Check
  `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`. See
  [General troubleshooting](../../troubleshooting/general.md#cpu-governor-hurts-host-resident-models).
- **DAX / Optane model store:** fast loads are nice for swap-testing; **never mmap** GGUFs from
  `dax=always` mounts into ROCm — use `--no-mmap` / `--load-mode none`. See
  [troubleshooting](../../troubleshooting/llama-cpp.md#dax-backed-mmap-oopses-amdgpu-svm).

### Full validated example (4× V620, Qwen3.5-122B-A10B-MTP)

```bash
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
./build/bin/llama-server \
  -m /path/to/Qwen3.5-122B-A10B-MTP-UD-Q4_K_M.gguf \
  -ngl all --split-mode tensor --tensor-split 1,1,1,1 --main-gpu 0 \
  --ctx-size 262144 --cache-type-k f16 --cache-type-v f16 --kv-unified \
  --batch-size 2048 --ubatch-size 256 --parallel 4 \
  --host 0.0.0.0 --port 8080 --temp 0.6 --metrics --jinja --numa distribute \
  --reasoning on --reasoning-format deepseek --chat-template-kwargs '{"preserve_thinking": true}' \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --spec-draft-p-min 0.0 --spec-draft-p-split 0.10 --no-spec-draft-backend-sampling --cache-ram 0
```

## Docker example {#docker-example}

TP2 + MTP4 from `#llamacpp` (ROCm 7.14, MXFP4 Qwen3.8-27B):

```yaml
environment:
  HSA_NO_SCRATCH_RECLAIM: "1"
  GGML_HIP_RDNA2_AUTO: "1"
  GGML_HIP_SAFE_STATE_IO: "1"
  GGML_TP_SHARDED_OUTPUT: "1"
  HSA_OVERRIDE_GFX_VERSION: "10.3.0"
command: >
  llama-server
  -hf quark75/Qwen3.8-27B-MXFP4-GGUF:MXFP4
  -ngl 999 -fa on -sm tensor -ts 1,1 -fit off
  --flash-attn on --ctx-size 32768
  --cache-type-k f16 --cache-type-v f16
  --batch-size 4096 --ubatch-size 4096 --parallel 1 --cont-batching
  --host 0.0.0.0 --port 8091 --temp 0.7 --jinja --metrics -kvu
  --reasoning-preserve
  --chat-template-kwargs '{"preserve_thinking": true}'
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-ngl 999
  --ctx-checkpoints 0
```

Build with `./scripts/build-rdna2-portable.sh`. RCCL needs working [PCIe P2P](../../tuning/p2p.md). If P2P
is enabled but slower, set `NCCL_P2P_DISABLE=1`. If all-reduce fails, try
`GGML_HIP_GFX1030_P2P_ALLREDUCE=off` or `GGML_CUDA_ALLREDUCE=none`.

The RDNA2 fork's recent HIP/RCCL work is validated primarily against **ROCm 7.14** (`#llamacpp`,
Sep 2026). Mid-7.2.x (e.g. 7.2.4) is not a confident target — upgrade or pin per
[Installing ROCm](../../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

## Single-GPU community ballpark (Qwen3.8-27B)

`#llamacpp` (Sep 2026), **1× V620**, RDNA2 fork, short env stack — treat as single-host snapshots:

| Quant | Decode (community) | Notes |
|---|---|---|
| **Q4_0** | ~39–49 t/s | Fastest native MMVQ path on the fork |
| **Q5_K_XL** | ~24 t/s | Slower unpack path |
| **Q8_0** | often ≥ Q5 | Less unpacking/shuffling; needs the VRAM (easier on TP2+) |
| **MXFP4** (~14.5 GB) | ≈ Q4_0 (slightly slower in one report) | Fits 1× 32 GB with reduced context |

On **4× V620**, long-context MTP benches (community suite) saw decode hold ~60–65 t/s at 32k on coding
prompts but drop harder by 64k — ngram / sidecar MTP work aims to lift acceptance; A/B with
[`llm-context-bench`](https://github.com/GeorgeMA-Strong/llm-context-bench).

### Idle power with embedded servers

Running an **embedding** server (nomic, etc.) alongside chat on the same cards can add ~**40 W per GPU**
even when the embedder looks idle (`#llamacpp`). Park embedders on a spare card or stop them when not
needed if you care about power.

## Notable limits

- Validated primarily on **4× V620 gfx1030, ROCm 7.14**; other systems use conservative fallbacks.
  **TP4 does work** on known-good server boards (e.g. Gigabyte **MC62-G40**) and on some dual-socket
  Broadwell-EP hosts. Most successful TP4 + P2P reports are on **AMD CPUs**.
- **Multi-socket** hosts can hurt TP — single-socket EPYC preferred. Dual-Xeon UPI traffic has
  **halved prefill** in `#general` even when theoretical cross-socket bandwidth looked fine.
- **KV checkpoints** crash on tensor split — use `--ctx-checkpoints 0`.
- `GGML_TP_SHARDED_OUTPUT` and `GGML_TP_VOCAB_SHARDED_OUTPUT` are incompatible modes.
- **Tensor-split prefill spikes current** — layer split can look fine while TP shuts the PSU off at
  prefill start (transient on the +12 V rail, classic with old miner PSUs). Try **160 W** (~2–4%
  slower) or **140 W** (~8–10% slower vs unlocked) before blaming the fork. See
  [Power Tuning](../../tuning/power.md).
- **TP3** (three cards) has caused driver crashes; stick to 2 or 4.
- Most `GGML_HIP_GFX1030_*` flags are redundant with `HSA_OVERRIDE_GFX_VERSION=10.3.0` unless A/B
  testing — prefer the [short env stack](#recommended-env-stack).
- **FA / q8 KV:** quantized V cache needs FA on; FA occupancy asserts on some head-256 models —
  see [troubleshooting](../../troubleshooting/llama-cpp.md#flashattention-abort-max_blocks_per_sm--0).
- **GPU sampling** (`--spec-draft-backend-sampling` and related): install `hipcub-devel` (or distro
  equivalent) at build time. Without it, expect `device 'Meta()' does not have support for op TOP_K`
  and fall back to slower CPU sampling.

For stock builds see [Building & running](../building.md). Speculative decoding configs:
[RDNA2 speculative decoding](../rdna2-speculative.md).
