# vLLM Configuration

> **WIP:** Env recipes and throughput numbers are **community-reported**. See
> [Verification status](../../reference/verification.md#vllm-configurationmd).

## Recommended environment (`#vllm-rdna`)

These settings are commonly used in the gfx1030 Discord for `-extras` images on ROCm 7.14:

```bash
export VLLM_TARGET_DEVICE=rocm
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0          # or 1 for autotuning (see compose below)
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=0
export HIP_FORCE_DEV_KERNARG=1
export RCCL_MSCCL_ENABLE=0
export VLLM_USE_RDNA2_FA=1                  # extras images: native RDNA2 FlashAttention
export VLLM_USE_V2_MODEL_RUNNER=1           # +17% vs V1 reported on gfx1030
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export SAFETENSORS_FAST_GPU=1               # faster safetensors → GPU load (`#vllm-rdna` Sep 2026)
```

### Custom all-reduce / P2P (two community stacks)

Pick **one** path — do not mix both:

| Topology | Env |
|---|---|
| **No working GPU↔GPU P2P** (common on Ice Lake / ACS-blocked hosts) | `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` (previous wiki default) |
| **P2P works** and you want the fast path (`#vllm-rdna` Aug 2026 benches) | `VLLM_FORCE_CUSTOM_ALL_REDUCE=1`, `NCCL_P2P_LEVEL=pix`, `RCCL_P2P_NET_DISABLE=1`, `RCCL_P2P_BATCH_ENABLE=1`, `NCCL_PROTO=Simple` |

If custom all-reduce misbehaves (stalls, bad latency on V620), fall back to the disable path. Some
community recipes now run a short all-reduce self-test at boot and auto-fallback — pull latest recipe /
fork notes.

Prefer **`--attention-backend RDNA_ATTN`** (or `VLLM_USE_RDNA2_FA=1`) over **`ROCM_ATTN`** on `-extras`.
`#vllm-rdna` reports `ROCM_ATTN` sitting in AMD Triton flash-attention compile for **hours** (RCCL and
Triton also fight each other). `FA_RDNA2` may not show up on older `-extras` images or GPTQ models that
still auto-select `ROCM_ATTN` — that is expected on hybrid GDN; see [fork](../fork.md#attention-backends).

For **multi-GPU TP** on current `-extras` images, if AOT compile cache replay causes device-bound
errors, add:

```bash
export VLLM_USE_AOT_COMPILE=0
export VLLM_DISABLE_COMPILE_CACHE=1
```

`RDNA_ATTN` / `VLLM_USE_RDNA2_FA` steer vLLM away from the generic AMD Triton flash-attention path,
which can be slower or crash on some Qwen head sizes. See [rdna_extras fork](../fork.md) for kernel
details. Full env cheat-sheet: [Reference](../../reference/env-vars.md).

**Don't force backends or quantization** unless you're A/B testing — or avoiding a
[`ROCM_ATTN` Triton hang](../../troubleshooting/vllm.md#rocm_attn-hangs-for-hours-triton-compile).
Let vLLM read the model's `config.json` unless that auto-selects the slow AMD Triton FA path.

## CUDA graphs (preferred over `--enforce-eager`)

On current `-extras` images, **CUDA graphs are the fast path** — you generally should **not** use
`--enforce-eager`. Graph capture can take a while on first boot (Triton JIT + torch-compile cache), but
steady-state throughput is much higher once warmed up.

Recommended graph config (fork-author reported on Qwen3.8-27B-AWQ-INT4, TP4, 4× V620 — **not wiki-reproduced**):

```bash
vllm serve /path/to/model \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --language-model-only \
  --skip-mm-profiling \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

Reported steady-state throughput on current `-extras` (1024/512 bench, TP4):

| Graph mode | Total tok/s | Output tok/s |
|---|---|---|
| PIECEWISE 1024/512 | 277.99 | 92.66 |
| FULL 1024/512 | 280.34 | 93.45 |
| PIECEWISE 16k/512, c=8 | 330.97 | — (prefill peak ~1573 tok/s) |

Alternative that also avoids `--enforce-eager`:

```bash
--compilation-config '{"mode": "NONE", "cudagraph_mode": "FULL", "compile_ranges_endpoints": []}'
```

To disable graphs entirely (debugging only):

```bash
--compilation-config '{"cudagraph_mode": "NONE"}'
```

If graph capture still crashes, fall back to `--enforce-eager` — but try the updated image first
(`docker pull blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0`). See
[vLLM troubleshooting](../../troubleshooting/vllm.md#cuda-graph-capture-crashes).

### Cache volumes (first boot is slow)

Mount these so Triton and torch-compile artifacts persist across container restarts:

```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
  - ~/.triton/cache:/root/.triton/cache          # ~3 GB compiled Triton kernels
  - ~/.triton/dump:/root/.triton/dump
  - ~/.triton/llvm:/root/.triton/llvm
  - ~/.cache/vllm/torch_compile_cache:/root/.cache/vllm/torch_compile_cache  # ~700 MB
```

First startup compiles kernels and can take many minutes. Subsequent boots reuse the cache.

For **recipe / Flash-Next** stacks that *want* the compile cache (not the multi-GPU AOT-disable
workaround above), `#vllm-rdna` (Sep 2026) recommends keeping the cache **on** and pointing it at a
persistent directory:

```bash
export VLLM_DISABLE_COMPILE_CACHE=0
export VLLM_CACHE_ROOT=/path/to/persistent/vllm-cache   # mount this in Docker
```

Community reports ~5 min vs ~10 min subsequent startups once the cache is warm — still slow cold, but
better than rebuilding every boot. Do **not** mix this with the TP `VLLM_DISABLE_COMPILE_CACHE=1`
workaround unless you have verified your image needs that disable path.

`SAFETENSORS_FAST_GPU=1` is also commonly set (and already present in some `vllm-rdna` Dockerfiles) to
speed weight load into GPU memory — see [AMD vLLM optimization notes](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/vllm-v1-optimization.html).

## Docker Compose example (GPTQ + MTP + CUDA graphs)

This `#vllm-rdna` setup reached ~24 output t/s on TP2 with a GPTQ model. Key points: **GPTQ** hits the
native `RDNA2W4A16LinearKernel`, CUDA graphs via `--compilation-config`, and `VLLM_DISABLED_KERNELS`
forces the RDNA2 quant path:

```yaml
services:
  server:
    image: blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0
    network_mode: host
    ipc: host
    devices: [/dev/kfd, /dev/dri]
    group_add: [video, render]
    security_opt: [label=disable]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.triton/cache:/root/.triton/cache
      - ~/.triton/dump:/root/.triton/dump
      - ~/.triton/llvm:/root/.triton/llvm
      - ~/.cache/vllm/torch_compile_cache:/root/.cache/vllm/torch_compile_cache
    environment:
      VLLM_TARGET_DEVICE: rocm
      VLLM_ROCM_USE_AITER: "0"
      VLLM_USE_RDNA2_FA: "1"
      PYTORCH_TUNABLEOP_ENABLED: "1"
      PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED: "0"
      VLLM_WORKER_MULTIPROC_METHOD: spawn
      GPU_MAX_HW_QUEUES: "2"
      VLLM_USE_V2_MODEL_RUNNER: "1"
      VLLM_DISABLED_KERNELS: ExllamaLinearKernel,TritonW4A16LinearKernel
    command: >
      serve btbtyler09/Qwen3.8-27B-GPTQ-4bit
      --served-model-name "Qwen 27B"
      --host 0.0.0.0 --port 8091
      --max-model-len 262144
      --gpu-memory-utilization 0.85
      --kv-cache-dtype float16
      --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
      --tensor-parallel-size 2
      --dtype float16
      --max-num-seqs 4
      --language-model-only --skip-mm-profiling --trust-remote-code
      --enable-auto-tool-choice --tool-call-parser qwen3_coder
      --enable-prefix-caching --enable-chunked-prefill
      --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

Confirm the kernel is active in logs: `Using RDNA2W4A16LinearKernel for AutoGPTQLinearMethod`.
