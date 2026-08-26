# vLLM Quantization on gfx1030

> **WIP:** Throughput numbers are **community-reported**. See
> [Verification status](../../reference/verification.md#vllm-quantizationmd).

## GPTQ vs AWQ

| Format | `-extras` kernel path | Notes |
|---|---|---|
| **GPTQ** (e.g. `btbtyler09/Qwen3.8-27B-GPTQ-4bit`) | `RDNA2W4A16LinearKernel` — native gfx1030 HIP | Best `-extras` throughput. Force with `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel`. |
| **AWQ** (e.g. `Qwen3.8-27B-AWQ-INT4`) | `RDNA2W4A16LinearKernel` on gfx10x | As of Aug 2026 `rdna2_extras`, AWQ dense routes through the same native W4A16 kernel as GPTQ (fork-author reported ~151 output t/s; **needs verify** on your image). |
| **compressed-tensors** (e.g. `cyankiwi/Qwen3.8-27B-AWQ-INT4`) | Mixed — use `--quantization compressed-tensors` | Custom int4 re-quants; benchmark against GPTQ/AWQ. |
| **AWQ-vd** (e.g. `ikantkode/Qwen3.8-27B-AWQ-vd`) | `RDNA2W4A16LinearKernel` when dense | Community-tuned AWQ variant; confirm kernel in logs. |

On **older images** (before the AWQ→RDNA2 dispatch fix), AWQ fell through to Triton/Exllama and could
stall at ~4–5 t/s on a 27B. Pull the latest `-extras` image and confirm
`Using RDNA2W4A16LinearKernel` in startup logs. Qwen3.8-27B AWQ also needs the fork's
**`head_size=256`** FlashAttention path — without it, FA falls back or never lists `RDNA_ATTN`.

Kernel dispatch details: [rdna2_extras fork](../fork.md#kernel-dispatch-on-gfx1030).

## KV-cache dtype

| Dtype | When people use it | `#vllm-rdna` notes |
|---|---|---|
| **`float16`** | Long-context / agents / tool calling | Default recommendation. `VLLM_USE_FA_RDNA2=1` currently needs fp16 KV. |
| **`int8_per_token_head`** | Throughput on GPTQ | Reported **5–10 t/s above fp8** in TG (and higher PP) in limited testing. One report that it misbehaves with chunked prefill. |
| **`fp8`** | VRAM savings | Often slower than `int8_per_token_head` on these cards. Quality drops on long sessions. |
| **KVarN** | Third-party KV compression | Raised concurrency on Qwen, **broke tool calling**, failed on Gemma 4. Community verdict: skip for agents. |

Prefer **`float16`** unless you are A/B testing a quantized KV for a non-agentic workload.

## MTP speculative decoding

MTP (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`) can boost throughput on GPTQ
models with CUDA graphs enabled. Acceptance rates dropped after a v0.27.1 speculator update (~0.25), but
base decode speed remains good — worth testing on your model. Example in
[Configuration](../configuration.md#docker-compose-example-gptq--mtp--cuda-graphs).

MTP is **not free at high concurrency**. A `#vllm-rdna` TP4 matrix on **Qwen3.6-35B-A3B-FP16**
(4× V620, `--enforce-eager`, 16k/1k-style bench) reported MTP-2 **+17%** output tok/s at `c=1`, but
**−53%** at `c=8`. Use MTP for latency-critical single-stream; leave it off for batched throughput.

## INT4 on gfx1030 (no native int4 ALUs)

RDNA2 has no hardware int4 matrix units. The `-extras` W4A16 kernels use **vdot2 on fp16 with on-the-fly
dequant** — int4 weights packed and processed via `dp4a`-style instructions. Both GPTQ and AWQ dense now
hit the same native HIP kernel on current `rdna2_extras` images.

Recent fork work on hybrid GDN models (Qwen3.8-27B-AWQ-INT4, TP4) **reported** **~93 output tok/s** with CUDA
graphs (1024/512), **~331 total tok/s** at 8 concurrent requests (16k/512), and prefill peaks of
**1450–1573 tok/s** — with the full HIP GDN prefill + decode chain replacing Triton JIT.

## Tips

- Lower `--gpu-memory-utilization` (e.g. `0.9` → `0.8`) if KV-cache allocation OOMs on 16 GB cards.
- For GPTQ or AWQ on `-extras`: set `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel` and
  watch logs for `RDNA2W4A16LinearKernel`.
- On older images, AWQ could fall through to Triton (~4–5 t/s on a 27B). Pull latest `-extras` and confirm
  the native kernel is active before blaming the quant format.
- Don't force `--attention-backend` or `--quantization` — let vLLM auto-select unless A/B testing.
- Mount Triton and torch-compile caches (see [Configuration](../configuration.md#cache-volumes-first-boot-is-slow)).
