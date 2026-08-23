# The rdna2_extras Fork

The `-extras` images are built from the
[`blivioniag/vllm`](https://github.com/blivioniag/vllm) fork on the **`rdna2_extras`** branch. It adds
hand-written **RDNA2 HIP kernels** and the vLLM plumbing to dispatch to them.

Why it exists: RDNA2 (gfx1030) has **no matrix/WMMA cores** — those arrived with RDNA3 (`gfx11xx`). So
quantized GEMM, attention, and MoE have to be implemented efficiently on RDNA2's regular vector ALUs.
This fork does exactly that (with RDNA3 WMMA variants where it makes sense).

## Branches

| Branch | Purpose |
|---|---|
| `main` | Fork baseline tracking upstream vLLM. |
| `feat/enable-gfx1030` | Baseline enablement so vLLM recognizes and runs on gfx1030. |
| `perf/rdna2_w4a16` | Performance work on the W4A16 path. |
| `rdna2_extras` | The aggregated branch with all the extra kernels — what the `-extras` images build. |

## What's in it

The custom device code lives under `csrc/rocm/*_rdna2.cu` (with `_rdna3` variants where WMMA applies),
wired into vLLM through Python kernel/layer modules and covered by targeted tests under
`tests/kernels/`.

### Attention

- `fa_rdna2.cu` — a **FlashAttention** kernel tuned for RDNA2, exposed via
  `vllm/v1/attention/ops/fa_rdna2_backend.py` and the `vllm/v1/attention/backends/rdna_attn.py` backend.
- `sparse_mla_rdna2.cu` + `rocm_rdna2_mla_sparse.py` — **sparse MLA** (multi-head latent attention,
  DeepSeek-style).
- `indexer_paged_mqa_rdna2.cu` — paged **MQA** indexer.

### Quantized GEMM (mixed-precision linear)

- `q_gemm_rdna2.cu` / `q_gemm_rdna2_prefill.cu` (+ `q_gemm_rdna2_common.cuh`) — **W4A16** quantized GEMM
  for decode and prefill.
- `q_gemm_w8a16_fp8_rdna2.cu`, `gemm_w8a8_fp8_dense_rdna2.cu` — **FP8** W8A16 / W8A8 paths.
- `qdq_4_rdna2.cuh`, `qdq_8_rdna2.cuh`, `qdq_fp8_rdna2.cuh` — quant/dequant helpers.
- Python: `model_executor/kernels/linear/mixed_precision/rdna2_w4a16.py`,
  `.../scaled_mm/rdna2_w8a16_fp8*.py`, `rdna2_w8a8_fp8.py`, and the `rdna_hybrid_w4a16.py` selector.

### MoE (mixture of experts)

- `moe_q_gemm_rdna2.cu`, `moe_w8a16_rdna2.cu`, `moe_w8a16_fp8_rdna2.cu` — quantized expert GEMMs.
- Python experts `fused_moe/experts/rdna2_mxfp4_moe.py`, `rdna2_w8a16_fp8_moe.py`, and
  `compressed_tensors` MoE glue (`..._fp8_rdna2`, `..._w4a4_mxfp4_rdna2`, `..._wna16_rdna2`).

### GDN (gated delta-net / linear attention)

Kernels for gated-delta-net models (e.g. Qwen3-Next-style linear attention):

- `gdn_decode_rdna2.cu` and the prefill pipeline `gdn_prefill_prep_rdna2.cu`,
  `gdn_prefill_kkt_rdna2.cu`, `gdn_prefill_delta_h_rdna2.cu`, `gdn_prefill_solve_wy_rdna2.cu`,
  `gdn_prefill_o_rdna2.cu`.

## Using it

The easiest path is a prebuilt **`-extras`** image — no compilation required:

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc host -p 8000:8000 \
  docker.io/blivioniag/vllm-rdna:v0.27.1-extras \
  vllm serve <model> --dtype float16
```

See [Running vLLM (Docker)](./vllm.md) for the full run recipe and [Building the Images](./vllm_images.md)
for how the `-extras` variant is produced (`VLLM_VARIANT=extras-fork`, `VLLM_REF=rdna2_extras`).

## Kernel dispatch on gfx1030

On `-extras` images, vLLM picks kernels based on quantization format:

| Quant method | Kernel selected | How to force |
|---|---|---|
| GPTQ (`AutoGPTQLinearMethod`) | `RDNA2W4A16LinearKernel` | `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel` |
| AWQ | Triton AWQ / Exllama | No native RDNA2 AWQ kernel yet — community patches in progress |
| FP8 W8A16 / W8A8 | `gemm_w8a16_fp8_rdna2` etc. | Automatic on `-extras` when model uses FP8 |

Check startup logs for lines like `Using RDNA2W4A16LinearKernel for AutoGPTQLinearMethod`. If you see
`TritonW4A16LinearKernel` or `ExllamaLinearKernel` instead, the RDNA2 quant path isn't active.

### Attention backends

- `VLLM_USE_RDNA2_FA=1` — enables the custom `fa_rdna2.cu` FlashAttention backend.
- `--attention-backend RDNA_ATTN` — alternative RDNA-tuned attention path (useful for Qwen models with
  head size 256 where generic AMD Triton FA is slow or broken).
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` — enables AMD Triton FA as a fallback; often slower on gfx1030.

On v0.27.1, hybrid GDN models may still auto-select `ROCM_ATTN` even with `VLLM_USE_RDNA2_FA=1`. That's
expected — the GDN layers use Triton FLA kernels regardless. The critical flag for these models is
`--enforce-eager` (see [Running vLLM](./vllm.md)).

### Disabling fallback kernels

```bash
export VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel
```

This is the main lever for forcing GPTQ onto the native RDNA2 W4A16 path. The variable accepts a
comma-separated list of kernel class names registered in vLLM's linear-kernel registry.

## Building from source (advanced)

```bash
git clone -b rdna2_extras https://github.com/blivioniag/vllm.git
cd vllm
export PYTORCH_ROCM_ARCH=gfx1030
pip install -r requirements/rocm.txt
pip install --no-build-isolation -e .
```

The kernels have their own tests, e.g.:

```bash
pytest tests/kernels/quantization/test_rdna2_w4a16.py
pytest tests/kernels/attention/test_fa_rdna2_shape_sweep.py
```

> These kernels are actively evolving. Treat the fork as experimental, pin to a known-good image tag for
> reproducible serving, and file issues on the fork if you hit correctness or performance problems.
