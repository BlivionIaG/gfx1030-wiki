# vLLM recipes (pick a path)

> **WIP:** Stack picks and tok/s figures are **community-reported** (`#vllm-rdna`, Sep 2026). See
> [Verification status](../reference/verification.md#vllm-recipesmd). Prefer this page when you want
> “which image / which model / how many cards” — not a kernel deep-dive.

Discord keeps asking the same questions: which Docker stack, which Hugging Face weights, and whether
**1× V620** is enough. This page consolidates those answers. Details for env vars and CUDA graphs live
on [Configuration](configuration.md); fork history on [vLLM forks](fork.md).

## Which stack?

| Goal | Use | Image / repo |
|---|---|---|
| Day-to-day serving with RDNA HIP kernels | Hub **`-extras`** | [`blivioniag/vllm-rdna:v0.27.1-extras`](https://hub.docker.com/r/blivioniag/vllm-rdna) (or `-extras-rocm7.14.0`) — [Running](running.md). `#vllm-rdna` Sep 18 also posted **`v0.28.0-extras`** as a **test** pull (`docker.io/blivioniag/vllm-rdna:v0.28.0-extras`); A/B before replacing 0.27.1. |
| Latest Flash-Next / `#17`+ / `#24` HEAD | Host **venv** build | Clone [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) + [Host venv](host-venv.md), then [`scripts/serve_rdna.sh`](serve-scripts.md). `#vllm-rdna` Sep 24–26: Hub tags **lag** and are **not auto-rebuilt**. |
| Tuned **27B / 122B** presets, host needs only `amdgpu` + Docker | Recipe book container | [`ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030`](https://github.com/leapdragon/vllm-rdna2-recipe) (`preset:…`) — mirror [`opengfx1030/vllm-rdna2-recipe`](https://github.com/opengfx1030/vllm-rdna2-recipe) |
| **Qwen3.8 Flash-Next** on **4×** V620 | `rdna_extras` + Flash-Next | [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) HEAD; published container may still be [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen) — [overview](flash-next.md#qwen38-flash-next-on-vllm) |

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
| **2× V620** | Recipe **TP=2** presets for 27B GPTQ / AWQ / MixedInt4, or Hub `-extras` / a [host-venv](host-venv.md) clone with `--tensor-parallel-size 2`. `#vllm-rdna` (Sep 26): one host built `vllm-rdna` from source and served **cyankiwi 27B AWQ-INT4** TP=2 — [community table](#host-venv-tp2-qwen38-27b-awq). `#vllm-rdna` (Oct 6): **Qwen3.8-27B fp16** TP=2 + MTP **~40–48 t/s** single-stream; a **Swift 1.5** 27B tune on the same layout **~48 t/s** single / **~70 t/s** across two instances — **not** Flash-Next. **Flash-Next** on 2 cards needs host RAM / MoE offload — prefer [llama.cpp](../llama-cpp/rdna2-speculative.md#flash-next-2x-iq4) over vLLM (`#vllm-rdna` Sep 14). |
| **3× V620** | vLLM **tensor parallel needs an even world size**. `#vllm-rdna` (Sep 14–15): use **pipeline parallel 3** (`PP=3`, `TP=1`) on vLLM. Older `rdna_extras` pins corrupted PP3 output — [corruption](../troubleshooting/vllm-flash-next.md#flash-next-pp3-output-corruption). Pin `b33f9b6` (Sep 19) boots graphs but dies on small captured prefill — [KeyError](../troubleshooting/vllm-flash-next.md#flash-next-pp3-graph-keyerror). MTP on 3 cards is **Needs verify** — [PP3 + MTP](./flash-next-serve.md#flash-next-3x-pp3-mtp). llama.cpp can still report TP across three cards (TP3 has [crash notes](../llama-cpp/rdna2-serving.md#notable-limits)). |
| **4× V620** | Best vLLM path for **Flash-Next** is still **TP=4**. Prefer **latest `rdna_extras`** with merged [`vllm-rdna#17`](https://github.com/opengfx1030/vllm-rdna/pull/17) (23 Sep) — [fast stack](./flash-next-serve.md#flash-next-4x-pr17). `#vllm-rdna` (Sep 19): **PP=4** can look great on 32k prefill then **tank decode**. `#vllm-rdna` (Sep 21): `#15`-only public merge reproduced **~1.1–1.45k** PP. Prefer TP4 over PP4. Hub `-extras` still lags. **Needs verify.** |
| **6× V620** | Flash-Next **minimum is 3** cards. `#vllm-rdna` (Sep 24): start with **`PP=3` + `TP=2`** or **`PP=6`**; keep **TP power-of-2** (avoid `PP=2` + `TP=3` as the first try). Clone HEAD + [host venv](host-venv.md) — Hub tags lag. Disagg prefill/decode across switches is open [`#23`](https://github.com/opengfx1030/vllm-rdna/pull/23) (**Needs verify**). |
| **8× V620** | Not a Flash-Next recipe. `#vllm-rdna` (Sep 30): community **GLM-5.3-Flash** AWQ on a **local vLLM 0.30 fork**, **TP=8**, **P2P off**, **150 W** — [GLM-5.3-Flash](#glm-53-flash). Org extras stay **0.28**; **Needs verify**. |

Also see [What fits well on V620](../choose-a-stack.md#what-fits-well-on-v620).

`#vllm-rdna` (Oct 8), for llama.cpp migrants: **`--pipeline-parallel-size N`** is the vLLM analogue
of llama.cpp **layer split** (`-sm layer`); **`--tensor-parallel-size N`** is the analogue of
**tensor split** (`-sm row` / `-ts`). vLLM can **combine** them (example: **TP=2 + PP=2** on four
cards). Prefer **power-of-2 TP**. Flash-Next on four cards still wants **TP=4**, not PP=4
(PP=4 can look great on long prefill then **tank decode** — table above).

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
| Qwen3.6 **35B-A3B** (FP16 / community quants) | 1–4× | Hub `-extras` | MoE sweet spot; MTP helps **c=1**, hurts high concurrency — [Quantization](quantization.md#mtp-speculative-decoding). |
| [`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) + [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) | **4×** | Flash-Next fork | Production Flash-Next weights + PLE sidecar in `#vllm-rdna`. |
| [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | **4×** | `rdna_extras` HEAD (QSA `#15`) / draft `#5` | W4A16 + original BF16 CPU PLE. QSA live-context merge: [overview](flash-next.md#intel-autoround-flash-next). Not Hub `-extras`. |
| [`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4) | **4×** | Experimental | Mentioned as a possible switch (`#vllm-rdna`); not a drop-in Hub `-extras` path yet. |
| [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ) | **3–4×** | `rdna_extras` HEAD / host venv | `#vllm-rdna` Sep 25–26 — shorter-reasoning AWQ. 3× community **~1300 PP / ~45 TG**. Two-card fit unconfirmed. [Swift](flash-next.md#swift-15-flash-next). |
| [`wtdcode/GLM-5.3-Flash-AWQ-W4A16`](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16) | **8×** | Local vLLM **0.30** fork (not Hub / not `rdna_extras` HEAD) | `#vllm-rdna` Sep 30 — TP=8, MTP-2, P2P off. [Notes](#glm-53-flash). |

Small-VRAM experiment: Ornith **9B** EXL3 (~6.8 GB) vs AWQ (~9 GB) — **experimental**, not in published
`-extras` tags yet. See [Quantization](quantization.md#experimental-exl3-and-quark-vllm-rdna-sep-2026).

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

- Add `--tensor-parallel-size N` for multi-GPU. Enable [PCIe P2P](../tuning/p2p.md) when possible.
- Prefer `--dtype float16`. Mount Triton / torch-compile caches for faster restarts —
  [Configuration](configuration.md#cache-volumes-first-boot-is-slow).
- Force the native W4A16 path when needed:
  `VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel` — confirm
  `Using RDNA2W4A16LinearKernel` in logs.

Full env block and Compose (GPTQ + MTP): [Configuration](configuration.md).

### Hub `-extras` TP4 Qwen3.8-27B AWQ {#hub--extras-tp4-qwen38-27b-awq}

Sanitized from a `#vllm-rdna` (Aug 31 2026) bench recipe on **4× V620** with working P2P / custom
all-reduce. Drop the custom-AR block if P2P is broken on your board — use
`VLLM_DISABLE_CUSTOM_ALL_REDUCE=1` instead ([Configuration](configuration.md#custom-all-reduce--p2p-two-community-stacks)).

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

### Host-venv TP=2 Qwen3.8-27B AWQ (community, Sep 26) {#host-venv-tp2-qwen38-27b-awq}

`#vllm-rdna` (Sep 26): one host **cloned `opengfx1030/vllm-rdna` and built from source** (not a Hub
tag), then served **cyankiwi** AWQ-INT4 at **TP=2**. Same short test on both models; answers
coherent, no NaNs. **Community / Needs verify.**

| | Qwen3-14B, TP=2 | Qwen3.8-27B, TP=2 |
|---|---|---|
| Weights per GPU | 4.69 GiB | 9.73 GiB |
| KV cache (logged) | 266k tokens (4k context) | 159k tokens (32k context) |
| Engine start | 39 s | 425 s (**first run only**) |
| 4 prompts at once | 88.3 tok/s | 46.9 tok/s |
| One prompt, decode | 59.5 tok/s | 29.5 tok/s |

Day-to-day 27B on **4 cards** still uses the [TP4 block](#hub--extras-tp4-qwen38-27b-awq) or a
recipe preset. This table is only the **two-card / source-build** snapshot.

`#vllm-rdna` (Oct 6): a different 2× host ran **unquantized fp16** Qwen3.8-27B at **TP=2** with
**MTP** and reported **~40–48 t/s** single-stream. A shorter-reasoning **Swift 1.5** 27B (not the
[Flash-Next Swift packs](flash-next.md#swift-15-flash-next)) on the same card count was **~48 t/s**
single-stream and **~70 t/s** across two instances. **Community / Needs verify.** Concurrency
notes for wvSplitK vs stacked LLMM1: [troubleshooting](../troubleshooting/vllm.md#wvsplitk-concurrency).

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

## Path C — Flash-Next

Flash-Next (Qwen3.8, PLE sidecar, 3×/4× V620 serve lines, AutoRound, DeepSeek-V4) lives on
[Flash-Next](./flash-next.md). This page stays the picker for Hub `-extras` and the recipe-container presets.

## GLM-5.3-Flash (community, Needs verify) {#glm-53-flash}

`#vllm-rdna` (Sep 30): a community **8× V620** host served public
[`wtdcode/GLM-5.3-Flash-AWQ-W4A16`](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16)
(AWQ W4A16 of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash); routed
experts INT4, attention / vision / MTP left BF16) on a **local vLLM 0.30 fork**. **TP=8**,
**MTP-2**, **P2P off** (PLX), **150 W** caps, dual-socket Broadwell-EP class. This is **not**
Hub `-extras` and **not** `rdna_extras` HEAD.

Org draft [`opengfx1030/vllm-rdna#2`](https://github.com/opengfx1030/vllm-rdna/pull/2) is a
**runtime-unverified** Glm5Next load path on a **side branch** (`later/glm53-flash-awq`); HIP
gates default **off**. `#vllm-rdna` (Sep 30): day-to-day extras stay on **0.28**; the
maintainer **paused** the 0.30 rebase and invited a PR of the community 0.30 work. V2 model
runner **can** work on that line.

Community snapshot (one host — **Needs verify**):

| Metric | Before local-fork work | After |
|---|---|---|
| Decode | ~2.2 t/s | **~32 t/s** |
| Prefill | ~50 t/s | **~320–500 t/s** (others called this **low** vs Flash-Next; hoped **~1000**) |
| Context | — | **262k** with vision; needle tests to **180k** |
| Reliability | — | 377 agent requests at 40–125k tokens: 0 failures / 0 GPU faults; 5–8 s TTFT with prefix cache |

gfx1030-relevant lessons (do **not** treat the unpublished kernel list as a Hub recipe):

- Serve **`--dtype float16`**, not bf16 — no bf16 hardware. Same quality; community **+68%**
  decode on this host.
- **Vision:** MIOpen compiled the downsample conv **per image size** (3–4 min; killed the
  engine once). Community used **conv-as-matmul** (158 s → 4 ms). Separate from the Flash-Next
  `max_pixels` OOM — [troubleshooting](../troubleshooting/vllm.md#glm-vision-miopen-compile).
- Indexer / tiled-fp8 / fused mHC / skinny-int4 MoE work lived on the **local fork**.

Not a drop-in for 4× Flash-Next hosts.

## Related

- [Serve scripts](./serve-scripts.md) — in-tree `serve_rdna.sh` after a host-venv build
- [Running (Docker)](./running.md) — image matrix and minimal `docker run`
- [Flash-Next](./flash-next.md) — Path C serve lines
- [Configuration](./configuration.md) — env vars, CUDA graphs, Compose
- [Quantization](./quantization.md) — GPTQ/AWQ, KV, MTP
- [vLLM forks](./fork.md) — `rdna_extras` vs Flash-Next vs recipe ports
- [Useful resources](../meta/resources.md)

