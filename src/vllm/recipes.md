# vLLM recipes (pick a path)

> **WIP:** Stack picks and tok/s figures are **community-reported** (`#vllm-rdna`, Sep 2026). See
> [Verification status](../../reference/verification.md#vllm-recipesmd). Prefer this page when you want
> “which image / which model / how many cards” — not a kernel deep-dive.

Discord keeps asking the same questions: which Docker stack, which Hugging Face weights, and whether
**1× V620** is enough. This page consolidates those answers. Details for env vars and CUDA graphs live
on [Configuration](../configuration.md); fork history on [vLLM forks](../fork.md).

## Which stack?

| Goal | Use | Image / repo |
|---|---|---|
| Day-to-day serving with RDNA HIP kernels | Hub **`-extras`** | [`blivioniag/vllm-rdna:v0.27.1-extras`](https://hub.docker.com/r/blivioniag/vllm-rdna) (or `-extras-rocm7.14.0`) — [Running](../running.md) |
| Tuned **27B / 122B** presets, host needs only `amdgpu` + Docker | Recipe book container | [`ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030`](https://github.com/leapdragon/vllm-rdna2-recipe) (`preset:…`) — mirror [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe) |
| **Qwen3.8 Flash-Next** on **4×** V620 | `rdna_extras` + Flash-Next | [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) HEAD; published container may still be [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen) — [overview](../overview.md#qwen38-flash-next-on-vllm) |

`#vllm-rdna` (Sep 2026): the recipe repo is treated as a **parts pile** (compose/env/patches against
pristine vLLM 0.27.1). New Flash-Next work lives on **`opengfx1030/vllm-rdna` `rdna_extras`** (Sep
14–15 cherry-picks); `vllm-rdna2-qwen` is the older published container. Official kernel
development is the same org extras repo — Hub `-extras` tags may lag HEAD until bake retargets.

**Do not mix** host ROCm userspace into the recipe container — the image carries its own stack.
Mounting host ROCm into it is a common break (recipe `TROUBLESHOOTING.md`).

## Pick by card count

| Cards | Community starting point (`#vllm-rdna`) |
|---|---|
| **1× V620 (32 GB)** | Prefer **MoE** (e.g. Qwen3.6 **35B-A3B**, Ornith-class) over dense 27B when prefill matters. Dense **Qwen3.8-27B AWQ** works for day-to-day chat; expect weaker PP than MoE. **Flash-Next is not a 1-card path** without heavy CPU/DRAM offload (weights ~60+ GB class + PLE). |
| **2× V620** | Recipe **TP=2** presets for 27B GPTQ / AWQ / MixedInt4, or Hub `-extras` with `--tensor-parallel-size 2`. **Flash-Next** on 2 cards needs host RAM / MoE offload — prefer [llama.cpp](../../llama-cpp/rdna2-speculative.md#flash-next-2x-iq4) over vLLM (`#vllm-rdna` Sep 14). |
| **3× V620** | vLLM **tensor parallel needs an even world size**. `#vllm-rdna` (Sep 14): use **pipeline parallel 3** (`PP=3`) on vLLM; llama.cpp can still report TP across three cards (TP3 has [crash notes](../../llama-cpp/rdna2-serving.md#notable-limits)). |
| **4× V620** | Best vLLM path for **Flash-Next**; also TP=4 dense 27B on `-extras` (see [TP4 AWQ recipe](#hub--extras-tp4-qwen38-27b-awq) below). |

Also see [What fits well on V620](../overview.md#what-fits-well-on-v620).

### Gemma 4 note

Community reports **Gemma 4 ~26B** still fails or is unfinished on current gfx1030 vLLM paths
(`#vllm-rdna`, Sep 13 2026). Prefer Qwen / Ornith until someone posts a working recipe.

## Model cheat sheet

| Model | Cards | Stack | Notes |
|---|---|---|---|
| [`cyankiwi/Qwen3.8-27B-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | 1–4× | Hub `-extras` or recipe preset | Current day-to-day **27B** pick in `#vllm-rdna` (Sep 13 2026). compressed-tensors AWQ. |
| [`btbtyler09/Qwen3.8-27B-GPTQ-4bit`](https://huggingface.co/btbtyler09/Qwen3.8-27B-GPTQ-4bit) | 2×+ | Recipe `preset:qwen38-27b-gptq` or Hub `-extras` | Recipe **reference** preset; native GPTQ → `RDNA2W4A16` on `-extras`. |
| [`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound`](https://huggingface.co/Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound) | 2× | Recipe builds/ | AutoRound MixedInt4 sibling in the recipe book. |
| [`Intel/Qwen3.5-122B-A10B-int4-AutoRound`](https://huggingface.co/Intel/Qwen3.5-122B-A10B-int4-AutoRound) | 3–4× | Recipe builds/ (not a one-line preset) | MoE 122B — needs recipe wrapper / weight prep; see build `BUILD.md`. |
| Qwen3.6 **35B-A3B** (FP16 / community quants) | 1–4× | Hub `-extras` | MoE sweet spot; MTP helps **c=1**, hurts high concurrency — [Quantization](../quantization.md#mtp-speculative-decoding). |
| [`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) + [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) | **4×** | Flash-Next fork | Production Flash-Next weights + PLE sidecar in `#vllm-rdna`. |
| [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | **4×** | Draft org PR / experimental | Separate track — [Intel AutoRound](../overview.md#intel-autoround-flash-next). |
| [`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) | **4×** | Experimental | Mentioned as a possible switch (`#vllm-rdna`); not a drop-in Hub `-extras` path yet. |

Small-VRAM experiment: Ornith **9B** EXL3 (~6.8 GB) vs AWQ (~9 GB) — **experimental**, not in published
`-extras` tags yet. See [Quantization](../quantization.md#experimental-exl3-and-quark-vllm-rdna-sep-2026).

---

## Path A — Hub `-extras` (recommended default)

```bash
docker pull docker.io/blivioniag/vllm-rdna:v0.27.1-extras

docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc host \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  docker.io/blivioniag/vllm-rdna:v0.27.1-extras \
  vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
    --dtype float16 \
    --max-model-len 8192 \
    --language-model-only --skip-mm-profiling --trust-remote-code
```

- Add `--tensor-parallel-size N` for multi-GPU. Enable [PCIe P2P](../../tuning/p2p.md) when possible.
- Prefer `--dtype float16`. Mount Triton / torch-compile caches for faster restarts —
  [Configuration](../configuration.md#cache-volumes-first-boot-is-slow).
- Force the native W4A16 path when needed:
  `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel` — confirm
  `Using RDNA2W4A16LinearKernel` in logs.

Full env block and Compose (GPTQ + MTP): [Configuration](../configuration.md).

### Hub `-extras` TP4 Qwen3.8-27B AWQ {#hub--extras-tp4-qwen38-27b-awq}

Sanitized from a `#vllm-rdna` (Aug 31 2026) bench recipe on **4× V620** with working P2P / custom
all-reduce. Drop the custom-AR block if P2P is broken on your board — use
`VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` instead ([Configuration](../configuration.md#custom-all-reduce--p2p-two-community-stacks)).

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export VLLM_RDNA_FORCE_FP16=1
export VLLM_USE_RDNA2_FA=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=0
export GPU_MAX_HW_QUEUES=2

# Only when GPU↔GPU P2P works:
export VLLM_FORCE_CUSTOM_ALL_REDUCE=1
export NCCL_P2P_LEVEL=pix
export RCCL_P2P_NET_DISABLE=1
export RCCL_P2P_BATCH_ENABLE=1
export NCCL_PROTO=Simple
export RCCL_MSCCL_ENABLE=0

cd /tmp   # avoid sys.path collisions with a local vllm checkout
vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 20480 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.88 \
  --dtype float16 \
  --language-model-only --skip-mm-profiling --trust-remote-code \
  --enable-prefix-caching --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

---

## Path B — Recipe container (27B / 122B presets)

Fastest path when you want the recipe book's measured knobs without installing ROCm on the host:

```bash
IMG=ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030
docker pull "$IMG"
docker run --rm "$IMG" list-presets

docker run -d --name vllm-rdna2 --network=host \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMG" preset:qwen38-27b-gptq
```

| Knob | Meaning |
|---|---|
| `preset:qwen38-27b-gptq` | Reference GPTQ 27B (TP=2). AWQ / MixedInt4 siblings are also presets. |
| `ROCR_VISIBLE_DEVICES` | Exactly **two** indices for TP=2 presets. |
| `MTP=0..3` | Speculative depth (default 2 in the recipe image). |
| `list-presets` / `DRYRUN=1` | Enumerate presets or print the resolved `vllm serve` line. |

First boot cold-compiles for **~10–15 minutes**; mount `/compile-cache` and Triton caches for ~3 min
warm boots — see the recipe [`containers/README.md`](https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/containers/README.md).
Community ballpark on **2× V620**: ~40–49 tok/s decode when TunableOp rows are seeded; ~27 t/s flat if
they are missing (recipe troubleshooting).

122B builds need the repo wrapper / one-time weight prep — not a bare `preset:` one-liner.

---

## Path C — Flash-Next (`vllm-rdna2-qwen`)

For agentic / long-context Flash-Next on **4× V620**, use the Flash-Next vLLM stack — not Hub
`-extras` and not the 27B recipe presets:

- **Source of truth (Sep 14–15):** [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras` (leapdragon work cherry-picked; PLE load path on HEAD)
- **Published container / docs (may lag):** [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next) — [`docs/rdna2/`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next/docs/rdna2)
- Weights: [`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
- PLE sidecar: [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant)

Expect large **host DRAM** for the n-gram / PLE store (community: **~64–95 GB** class; **128 GB**
host RAM was **not** enough for KV offload on one 4× host). Long-prompt / intermittent stalls: try
`VLLM_USE_V2_MODEL_RUNNER=0` — [troubleshooting](../../troubleshooting/vllm.md#flash-next-long-prompt-stalls).
Throughput and KV tightness: [overview](../overview.md#qwen38-flash-next-on-vllm).

---

## Related

- [Running (Docker)](../running.md) — image matrix and minimal `docker run`
- [Configuration](../configuration.md) — env vars, CUDA graphs, Compose
- [Quantization](../quantization.md) — GPTQ/AWQ, KV, MTP
- [vLLM forks](../fork.md) — `rdna_extras` vs Flash-Next vs recipe ports
- [Useful resources](../../meta/resources.md) — external links
