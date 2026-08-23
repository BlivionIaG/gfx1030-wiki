# Running vLLM (Docker)

The quickest way to serve LLMs on gfx1030 / RDNA is the prebuilt
[`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) images. They are built on a
[`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) ROCm + PyTorch base and target
seven RDNA architectures, so a single image runs on RDNA2 (gfx1030) through RDNA4.

## Image matrix

### Base images — `blivioniag/rocm-rdna`

| Tag | ROCm | PyTorch | Triton |
|---|---|---|---|
| `7.2.0` | 7.2.0 | 2.12.0 | 3.5.1 |
| `7.14.0` | 7.14.0 | 2.13.0 | 3.7.1 |

These are a general-purpose **ROCm PyTorch base for RDNA** — useful on their own if you just want a
working `torch` on a Radeon card.

### vLLM images — `blivioniag/vllm-rdna`

| Tag | vLLM | Base | Variant |
|---|---|---|---|
| `v0.27.1` | v0.27.1 | rocm-rdna:7.2.0 | upstream |
| `v0.27.1-rocm7.14.0` | v0.27.1 | rocm-rdna:7.14.0 | upstream |
| `v0.27.1-extras` | v0.27.1 | rocm-rdna:7.2.0 | [`rdna2_extras` fork](./vllm_fork.md) |
| `v0.27.1-extras-rocm7.14.0` | v0.27.1 | rocm-rdna:7.14.0 | [`rdna2_extras` fork](./vllm_fork.md) |
| `v0.26.0` | v0.26.0 | rocm-rdna:7.2.0 | upstream |
| `v0.22.1` | v0.22.1 | — | upstream |

The **`-extras`** tags use the [`rdna2_extras` fork](./vllm_fork.md), which adds hand-written RDNA2 HIP
kernels (FlashAttention, quantized GEMM, MoE, GDN, …). Use them if you want the RDNA-tuned kernels; use
the plain tags for stock upstream vLLM. Check
[Docker Hub](https://hub.docker.com/r/blivioniag/vllm-rdna/tags) for the current tag list — new vLLM
releases and ROCm bases are added over time.

Every image bakes these `PYTORCH_ROCM_ARCH` targets:
`gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201`.

## Run it

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc host \
  -p 8000:8000 \
  docker.io/blivioniag/vllm-rdna:v0.27.1-extras \
  vllm serve Qwen/Qwen2.5-7B-Instruct --dtype float16 --max-model-len 8192
```

- Give the container the GPU with `--device /dev/kfd --device /dev/dri` and the `video`/`render` groups.
- **Prefer `--dtype float16`.** RDNA2 has weak/emulated BF16; letting vLLM pick bf16 from a model's
  `config.json` can trigger slow float32 fallbacks.
- For a **non-Navi-21** RDNA2 card (gfx1031/1032/…), add `-e HSA_OVERRIDE_GFX_VERSION=10.3.0`. See
  [HSA_OVERRIDE for RDNA2 Cousins](./hsa_override.md).
- Multi-GPU: add `--tensor-parallel-size N`; enabling [PCIe P2P](./tuning_p2p.md) helps a lot here.

Query the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hi from gfx1030!"}]}'
```

## Recommended environment (`#vllm-rdna`)

These settings are commonly used in the gfx1030 Discord for `-extras` images on ROCm 7.14:

```bash
export VLLM_ROCM_USE_AITER=0
export VLLM_RDNA_FORCE_FP16=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=0          # or 1 for autotuning (see compose below)
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export GPU_MAX_HW_QUEUES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export VLLM_USE_RDNA2_FA=1                  # extras images: native RDNA2 FlashAttention
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

`RDNA_ATTN` / `VLLM_USE_RDNA2_FA` steer vLLM away from the generic AMD Triton flash-attention path,
which can be slower or crash on some Qwen head sizes. See [The rdna2_extras Fork](./vllm_fork.md) for
kernel details.

## Docker Compose example (GPTQ + MTP, community-validated)

This `#vllm-rdna` setup reached ~24 output t/s on TP2 with a GPTQ model. Key points: **GPTQ** (not AWQ)
hits the native `RDNA2W4A16LinearKernel`, `--enforce-eager` is required for hybrid GDN models on
v0.27.1, and `VLLM_DISABLED_KERNELS` forces the RDNA2 quant path:

```yaml
services:
  server:
    image: blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0
    network_mode: host
    ipc: host
    devices: [/dev/kfd, /dev/dri]
    group_add: [video]
    security_opt: [label=disable]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - ~/.triton/cache:/root/.triton/cache
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
      --enforce-eager
```

Confirm the kernel is active in logs: `Using RDNA2W4A16LinearKernel for AutoGPTQLinearMethod`.

## Quantization: GPTQ vs AWQ on gfx1030

| Format | `-extras` kernel path | Notes |
|---|---|---|
| **GPTQ** (e.g. `btbtyler09/Qwen3.8-27B-GPTQ-4bit`) | `RDNA2W4A16LinearKernel` — native gfx1030 HIP | Best `-extras` throughput today. Force with `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel`. |
| **AWQ** | Triton AWQ / Exllama fallback | Does **not** use `RDNA2W4A16LinearKernel`. Community work is ongoing for AWQ-native RDNA2 kernels. |
| **compressed-tensors** (e.g. `cyankiwi/Qwen3.8-27B-AWQ-INT4`) | Mixed — test with `--quantization compressed-tensors` | Useful for custom int4 re-quants; benchmark against GPTQ. |

If AWQ performance on v0.27.1 is poor, try `v0.26.0` or `v0.26.0-extras` as a fallback while AWQ
kernel support matures. Hybrid GDN models (Qwen3.5+/3.8) on v0.27.1 **require `--enforce-eager`** —
CUDA-graph capture crashes on the GDN all-reduce regardless of attention backend.

## Tips

- Lower `--gpu-memory-utilization` (e.g. `0.9` → `0.8`) if KV-cache allocation OOMs on 16 GB cards.
- If a Triton/flash-attention path crashes on an upstream image, try an `-extras` image (RDNA2 kernels)
  or fall back to the default attention backend.
- Hybrid GDN models (Qwen3.5+, Qwen3.8): always add `--enforce-eager` on v0.27.1 — graph capture fails
  on the GDN linear-attention all-reduce.
- For GPTQ on `-extras`: set `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel` and
  watch logs for `RDNA2W4A16LinearKernel`.
- AWQ on gfx1030 is still catching up to GPTQ on `-extras`. If throughput is unexpectedly low (~4–5 t/s
  on a 27B), check whether you're on AWQ (Triton path) vs GPTQ (native RDNA2 kernel).
- Mount Triton and torch-compile caches as volumes (see compose above) — first boot compiles kernels and
  is much slower.
- To rebuild the images yourself or add a new vLLM version, see [Building the Images](./vllm_images.md).
