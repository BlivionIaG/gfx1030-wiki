# Running vLLM (Docker)

> **WIP:** See [Verification status](../reference/verification.md#vllm).

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
| `v0.31.0-extras` / `v0.31.0-extras-rocm7.14.0` | v0.31.0 | ROCm **7.14** (same digest) | [`rdna_extras` fork](fork.md) — **current** extras bake. `#announcements` (11 Oct 2026): torch **2.12**, Triton **3.7.1**. Flash-Next MTP-2 (BF16 + INT4-disk) served from this image. |
| `v0.28.0-extras` / `v0.28.0-extras-rocm7.14.0` | v0.28.0 | ROCm 7.14 | Historical extras. Source now [`rdna_extra/v0.28.0`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extra/v0.28.0). Prefer `v0.31.0-extras` unless you need the old line. |
| `v0.27.1` | v0.27.1 | rocm-rdna:7.2.0 | upstream |
| `v0.27.1-rocm7.14.0` | v0.27.1 | rocm-rdna:7.14.0 | upstream — `#vllm-rdna` Sep 24: **can remove**; prefer `-extras` or host build |
| `v0.27.1-extras` | v0.27.1 | rocm-rdna:7.2.0 | Older [`rdna_extras`](fork.md) bake (Aug 2026). Keep only for A/B. |
| `v0.27.1-extras-rocm7.14.0` | v0.27.1 | rocm-rdna:7.14.0 | Same as `v0.27.1-extras` (same digest). |
| `v0.26.0` | v0.26.0 | rocm-rdna:7.2.0 | upstream |
| `v0.22.1` | v0.22.1 | — | upstream |

The **`-extras`** tags use the [`rdna_extras` fork](fork.md) lineage (historical bake still clones
`blivioniag/vllm` `rdna2_extras`), which adds hand-written RDNA2 HIP
kernels (FlashAttention, quantized GEMM, MoE, GDN, …). Check
[Docker Hub](https://hub.docker.com/r/blivioniag/vllm-rdna/tags) for the current tag list. Tags are
refreshed in place — `docker pull` before debugging.

`#announcements` (11 Oct 2026): **`v0.31.0-extras`** is the current extras image — `rdna_extras`
is now the **v0.31.0** port, and that bake served Flash-Next MTP-2. Older Hub tags (`v0.27.1-extras*`,
`v0.28.0-extras*`) do **not** track HEAD. For kernels newer than the bake, **clone the org repo
and follow [Host venv](host-venv.md)**. Use `rocm-rdna:7.14.0` if you only need a ROCm packaging base.

**Multi-GPU:** pick a **7.2.0** or **7.14.0** tag, not a host ROCm in the 7.2.1–7.13 gap. RCCL on
those in-between releases is reported broken with more than one card — see
[Installing ROCm](../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

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
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  docker.io/blivioniag/vllm-rdna:v0.31.0-extras \
  vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
    --dtype float16 \
    --max-model-len 8192 \
    --language-model-only --skip-mm-profiling --trust-remote-code
```

- Give the container the GPU with `--device /dev/kfd --device /dev/dri` and the `video` **and**
  `render` groups. Missing `render` is a common `#vllm-rdna` cause of
  [`Failed to infer device type`](../troubleshooting/vllm.md#failed-to-infer-device-type--amdsmi_status_not_init).
- **Prefer `--dtype float16`.** RDNA2 has weak/emulated BF16; letting vLLM pick bf16 from a model's
  `config.json` can trigger slow float32 fallbacks.
- The **27B AWQ** example is the current `#vllm-rdna` day-to-day dense pick (Sep 2026). For a smaller
  smoke test, swap in any instruct model that fits VRAM. Model / card-count matrix:
  [Recipes](recipes.md).
- For a **non-Navi-21** RDNA2 card (gfx1031/1032/…), add `-e HSA_OVERRIDE_GFX_VERSION=10.3.0`. See
  [HSA_OVERRIDE](../setup/hsa-override.md).
- Multi-GPU: add `--tensor-parallel-size N`; enabling [PCIe P2P](../tuning/p2p.md) helps a lot here.

Query the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cyankiwi/Qwen3.8-27B-AWQ-INT4","messages":[{"role":"user","content":"Hi from gfx1030!"}]}'
```

## Next steps

- Choose Hub vs recipe container vs Flash-Next: [Recipes](recipes.md)
- Tune env vars and CUDA graphs: [Configuration](configuration.md)
- Pick a quant format: [Quantization](quantization.md)
- Kernel details: [rdna_extras fork](fork.md)
