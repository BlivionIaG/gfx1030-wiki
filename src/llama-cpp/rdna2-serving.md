# RDNA2 fork — Serving

## Launch

`HSA_OVERRIDE_GFX_VERSION=10.3.0` is **required** for the tested V620/`gfx1030` native profile.

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

- `GGML_CUDA_ALLREDUCE=nccl` — author reports **+10% tgen / +20% prefill** on Qwen 122B.
- Optional TP4 mode: `GGML_HIP_GFX1030_P2P_ALLREDUCE=auto-expanded` (topology-gated).

### Full validated example (4× V620, Qwen3.5-122B-A10B-MTP)

```bash
GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_ALLREDUCE=nccl HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
~/llama-cpp-mirrored/build/bin/llama-server \
  -m ~/models/Qwen3.5-122B-A10B-MTP-UD-Q4_K_M/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf \
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

## Notable limits

- Validated primarily on **4× V620 gfx1030, ROCm 7.14**; other systems use conservative fallbacks.
  **TP4 does work** (author: Gigabyte MC62-G40; others on Broadwell-EP across two NUMA nodes). Most
  successful TP4 + P2P reports are on **AMD CPUs**.
- **Multi-socket** hosts can hurt TP — single-socket Epyc preferred. Dual-Xeon UPI traffic has
  **halved prefill** in `#general` even when theoretical cross-socket bandwidth looked fine.
- **KV checkpoints** crash on tensor split — use `--ctx-checkpoints 0`.
- `GGML_TP_SHARDED_OUTPUT` and `GGML_TP_VOCAB_SHARDED_OUTPUT` are incompatible modes.
- **Tensor-split prefill spikes current** — layer split can look fine while TP shuts the PSU off at
  prefill start (transient on the +12 V rail, classic with old miner PSUs). Try **160 W** (~2–4%
  slower) or **140 W** (~8–10% slower vs unlocked) before blaming the fork. See
  [Power Tuning](../../tuning/power.md).
- **TP3** (three cards) has caused driver crashes; stick to 2 or 4.
- Most `GGML_HIP_GFX1030_*` flags are redundant with `HSA_OVERRIDE_GFX_VERSION=10.3.0` unless A/B testing.

For stock builds see [Building & running](../building.md). Speculative decoding configs:
[RDNA2 speculative decoding](../rdna2-speculative.md).
