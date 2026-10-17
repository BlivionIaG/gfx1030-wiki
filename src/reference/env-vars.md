# Environment Variables & Quick Reference

> **WIP:** vLLM and llama.cpp tables include Discord-sourced defaults — see
> [Verification status](./verification.md#reference--troubleshooting).

A cheat-sheet of the settings that matter most when running ML workloads on gfx1030 / RDNA2.

## Key environment variables

| Variable | Example | What it does |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | Makes non-Navi-21 RDNA2 cards run gfx1030 kernels. See [override guide](../setup/hsa-override.md). |
| `HIP_VISIBLE_DEVICES` | `0` | Restrict which GPUs a HIP program sees (hide an iGPU or pick one card). |
| `ROCR_VISIBLE_DEVICES` | `0` | Same idea, at the ROCr runtime level. |
| `PYTORCH_ROCM_ARCH` | `gfx1030;gfx1100;…` | Target arch(es) when **building** PyTorch/vLLM/extensions from source. The RDNA images bake `gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201`. |
| `AMDGPU_TARGETS` | `gfx1030` | Target arch(es) for CMake/HIP builds. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | Force PyTorch to use rocBLAS instead of hipBLASLt (works around Navi 21 hipBLASLt gaps). |
| `ROCM_PATH` | `/opt/rocm` | Where ROCm is installed; used by many build systems. |
| `HSA_ENABLE_SDMA` | `0` | Occasionally needed to work around DMA issues on some setups. |

### vLLM on gfx1030 (`#vllm-rdna`)

| Variable | Example | What it does |
| --- | --- | --- |
| `VLLM_TARGET_DEVICE` | `rocm` | Pin the ROCm platform (Compose / CI; avoids CUDA autodetection). |
| `VLLM_ROCM_USE_AITER` | `0` | Disable aiter fused kernels (CDNA-oriented; community default on RDNA2). |
| `VLLM_ROCM_USE_AITER_MOE` | `0` | Disable AITER MoE (same reason). |
| `VLLM_RDNA_FORCE_FP16` | `1` | Force FP16 compute paths — avoids slow BF16 emulation on RDNA2. |
| `VLLM_USE_RDNA2_FA` | `1` / `0` | Enable native RDNA2 FlashAttention on Hub `-extras`. Flash-Next / `rdna_extras` HEAD: community treated FA as **WIP** and A/B'd `0` (`#vllm-rdna` Sep 15). Flash-Next **QSA** layers ignore this — [partial FA](../troubleshooting/vllm-flash-next.md#flash-next-fa-rdna2-partial). |
| `VLLM_FA_RDNA2_GQA_MODE` | `subgroup` | FA-RDNA2 prefill GQA path on `rdna_extras` (`rdna_attn.py`). Default `subgroup` uses `fa_rdna2_prefill_paged_varlen_gqa`; other values fall back. A/B only. |
| `VLLM_FA_RDNA2_GQA_DECODE` | `1` | FA-RDNA2 GQA **decode** path. `serve_rdna.sh` sets this when `ATTN=fa`. Not a Flash-Next QSA lever. |
| `VLLM_FA_RDNA2_SPLIT_DECODE` | `1` | FA-RDNA2 split-K decode (default on in `rdna_attn.py`). A/B only — not a Flash-Next QSA lever. |
| `VLLM_GDN_HIP_PREFILL` | `0` | Disable HIP GDN prefill (falls back to Triton/FLA). Used while chasing Flash-Next PP3 corruption — did **not** fix it. |
| `VLLM_PP_LAYER_PARTITION` | `17,18,13` | Uneven pipeline-parallel layer split. Community 3× Flash-Next + MTP (48 layers) left the last stage light for the drafter. **Unset** this in the PLE worker or it refuses (`pp_size=1`). |
| `VLLM_ROCM_MOE_PADDING` | `0` | Do not pad routed-expert weights (3× MTP VRAM squeeze). |
| `VLLM_RDNA_AR` | `1` | RDNA all-reduce. Library default **off**. Merged [`#22`](https://github.com/opengfx1030/vllm-rdna/pull/22) (27 Sep) adds two-shot so serve scripts can leave `VLLM_FORCE_CUSTOM_ALL_REDUCE=0`. Failed self-test falls back to RCCL. A/B on your topology. Optional knobs: `VLLM_RDNA_AR_MAX_KB`, `VLLM_RDNA_AR_ONESHOT_KB`, `VLLM_RDNA_AR_ALGO`, `VLLM_RDNA_AR_BLOCKS`, `VLLM_RDNA_AR_PACE`. `#vllm-rdna` (Oct 11): on a host **without** PCIe P2P, set `0` or pass launcher `P2P=0` (also sets `NCCL_P2P_DISABLE=1`). |
| `VLLM_ROCM_MOE_SKINNY_MAX_M` | `16` | Opt-in HIP MoE row cap for concurrent MTP (9–16 token rows). Default **8**. Merged [`#26`](https://github.com/opengfx1030/vllm-rdna/pull/26). No single-chat gain. |
| `VLLM_RDNA_DENSE_GEMV` | `1` | Skip skinny `wvSplitK` GEMM. `#vllm-rdna` Sep 17: **mandatory** on hosts that fault `wvSplitK_hf_sml_*` right after the profile run (`HSA_STATUS_ERROR_EXCEPTION`). GPU core dumps can be **several GB** in the process CWD — [troubleshooting](../troubleshooting/vllm-flash-next.md#wvsplitk-gpu-fault). Oct 6: on a no-fault **4× W6800X** 27B fp16 host, **leaving wvSplitK on** beat stacked LLMM1 for decode concurrency — [vLLM](../troubleshooting/vllm.md#wvsplitk-concurrency). |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | `1` | Sets compilation `mode=NONE` (no torch.compile); GDN/FA/all-reduce run eagerly between graph segments. `#vllm-rdna` Sep 17: community **~39 t/s** vs **~55 t/s** with `0` + compile. The [4× PIECEWISE recipe](../vllm/flash-next-serve.md#flash-next-4x-piecewise) still sets `1` for the stable path — A/B `0` only after graphs are clean. |
| `VLLM_PLE_QUANT_DIR` | path/`ples_int4` | Flash-Next INT4 PLE (`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`). Launcher `PLE=int4` loads the table in pinned RAM; `PLE=int4-disk` **mmap**s shards (`MADV_RANDOM`) so only the gather/dequant working set hits RAM. Directory must contain `META.json`. `#announcements` (11 Oct 2026): all three PLE modes tested. |
| `VLLM_PLE_CPU_OFFLOAD` | `1` | PLE **CPU sidecar** (launcher `PLE=int4-disk`). Pair with `VLLM_PLE_OFFLOAD_READY_TIMEOUT` (launcher default **3600**). Not the same as KV RAM offload (`#24`). |
| `PLE_OFFLOAD_PREFAULT` | `0` / `1` | After GPU workers load, **fault PLE table pages into RAM** (default **on**). `#vllm-rdna` Oct 8: set **`0`** on low-RAM hosts so the sidecar does not compete with swap (community: **~22–28%** slower tok/s vs prefault=1 on prose). Overlay / leapdragon trees use this unprefixed name; some other trees expose `VLLM_PLE_OFFLOAD_PREFAULT`. Not on org extras HEAD. Community: **~1:2 RAM:VRAM** is enough for typical Flash-Next PLE; **32 GiB** host RAM with 2× V620 is tight. |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` / `0` | V2 runner; `#vllm-rdna` reported **+17%** vs V1 on gfx1030 `-extras`. On **Flash-Next**, try `0` if long prompts stall — [troubleshooting](../troubleshooting/vllm-flash-next.md#flash-next-long-prompt-stalls). |
| `VLLM_DISABLED_KERNELS` | `ExllamaLinearKernel,TritonW4A16LinearKernel` | Force GPTQ onto `RDNA2W4A16LinearKernel`. |
| `VLLM_DISABLE_CUSTOM_ALL_REDUCE` | `1` | Disable custom all-reduce (safer when P2P is broken / Ice Lake). |
| `VLLM_FORCE_CUSTOM_ALL_REDUCE` | `1` | Force custom all-reduce when P2P works (`#vllm-rdna` PIX stack). `#vllm-rdna` Sep 24: **crashed one whole server** — avoid as default. Merged [`#22`](https://github.com/opengfx1030/vllm-rdna/pull/22) is the two-shot `rdna_ar` replacement. Mutually exclusive intent with disable. |
| `NCCL_P2P_LEVEL` | `pix` / `PXB` / `PHB` | RCCL P2P topology level — `pix` used with the force-custom stack. |
| `RCCL_P2P_NET_DISABLE` | `1` | Pair with PIX custom all-reduce benches. |
| `RCCL_P2P_BATCH_ENABLE` | `1` | Pair with PIX custom all-reduce benches. |
| `NCCL_PROTO` | `Simple` | Protocol pin used in PIX custom all-reduce benches. |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Worker spawn method — avoids fork issues with ROCm. |
| `VLLM_BATCH_INVARIANT` | `0` | Batch-invariant mode forces hipBLASLt; keep off on gfx1030. |
| `GPU_MAX_HW_QUEUES` | `2` | Launcher default **2**. Merged [`#107`](https://github.com/opengfx1030/vllm-rdna/pull/107) (16 Oct): Flash-Next **TP4** recipes set **4** (neutral with fork kernels off; removes per-rank skew with `rdna_mono_*` / MonoKernel `#94` in the graph). Other recipes keep 2 until measured. |
| `VLLM_ROCM_MONO_DECODE` | `1` | Alias of `--monokernel`. Merged [`#94`](https://github.com/opengfx1030/vllm-rdna/pull/94) (16 Oct). Opt-in Flash-Next persistent decode on gfx1030. Off by default. Needs V2 runner, PP=1, symmetric W4A16. Not Hub. [MonoKernel](../vllm/flash-next-serve.md#flash-next-monokernel). |
| `HSA_KERNARG_POOL_SIZE` | `262144` | `#vllm-rdna` (17 Oct): community **TP=8** host reclaimed ~**14 GB** of host RAM vs the **4 MB** default (padded kernarg pool). c=8 @ 16k speed change was noise (under 1%). Maintainer: legacy; usual path is `HIP_FORCE_DEV_KERNARG=1` (already in the common stack). **Needs verify** on 4×. |
| `ROCPROFILER_REGISTER_ENABLED` | `0` | Merged [`#108`](https://github.com/opengfx1030/vllm-rdna/pull/108) (16 Oct): launchers default **0** so HIP/HSA do not register a rocprofiler client on every serve process. `#vllm-rdna` (16 Oct): that registration spun **≥1 idle core per GPU**. Profiling runs set **`1`**. Distinct from [ROCR busy-spin](../troubleshooting/vllm.md#rocr-idle-cpu-spin-therock-714). |
| `HIP_FORCE_DEV_KERNARG` | `1` | HIP kernel-arg in device memory (common `#vllm-rdna` stack). `#vllm-rdna` (17 Oct): prefer this over shrinking `HSA_KERNARG_POOL_SIZE` unless you are reclaiming host RAM on a large-TP box. |
| `RCCL_MSCCL_ENABLE` | `0` | Disable MSCCL (stream-hungry; conflicts with Triton on some TP hosts). |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | Enable AMD Triton FA fallback. Prefer `RDNA_ATTN` / `VLLM_USE_RDNA2_FA` on `-extras`. |
| `PYTORCH_TUNABLEOP_ENABLED` | `0` / `1` | `0` for reproducible benches; `1` to **load** a qualified table (`TUNING=0` in the `#17` launcher). |
| `PYTORCH_TUNABLEOP_TUNING` | `0` | Keep **off** in serve. Online tuning is a separate regenerate step. |
| `PYTORCH_TUNABLEOP_FILENAME` | set by `serve_rdna.sh` | **Hash-locked** named profile (`rocm7.14-rocblas5.5`, 783 rows, or thin `rocm10-rocblas5.6`). The launcher picks it from `librocblas.so.5`. Override with `TUNABLEOP_PROFILE`; mismatch aborts — [serve scripts](../vllm/serve-scripts.md#tunableop), [troubleshooting](../troubleshooting/vllm.md#tunableop-rocblas-mismatch). |
| `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED` | `0` | Disable hipBLASLt in tunableop (pair with `TORCH_BLAS_PREFER_HIPBLASLT=0`). |
| `VLLM_RDNA_MOE_RESIDENT` | `1` | `#17` resident W4A16 MoE path (pack once at load). `#vllm-rdna` Sep 29: leaving this **off** on latest `rdna_extras` dropped Flash-Next PP from ~**2k → ~1450**; restoring `1` recovered ~2k. |
| `VLLM_RDNA_DYNAMIC_PREFILL` | `1` | Opt-in adaptive mixed prefill+decode (`#36`, merged 5 Oct). Must be the string **`1`**. Default **off** — static `--prefill-schedule-interval` / `--long-prefill-token-threshold` stay as set. Leave off for max single-stream PP. — [troubleshooting](../troubleshooting/vllm.md#prefill-blocks-decode). |
| `VLLM_RDNA_DYN_FLOOR_TPS` | `5.0` | Per-decoder tok/s floor the `#36` controller must not break (default **5**). Other `VLLM_RDNA_DYN_*` knobs (window, cap ladder, interval ceiling) are documented on the PR; restart required to change them. |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | Reduces CUDA/HIP allocator fragmentation. |
| `VLLM_USE_DEEP_GEMM` | `0` | Disable DeepGEMM (NVIDIA-oriented). |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | Disable FlashInfer sampler (not useful on RDNA2). |
| `VLLM_USE_AOT_COMPILE` | `0` | Disable AOT compile on multi-GPU if cache replay causes device-bound errors. |
| `VLLM_DISABLE_COMPILE_CACHE` | `1` | Disable torch.compile cache (pair with `VLLM_USE_AOT_COMPILE=0` for TP stability). For **faster recipe startups**, invert: `0` + set `VLLM_CACHE_ROOT` to a persistent mount (`#vllm-rdna` Sep 2026). |
| `VLLM_CACHE_ROOT` | `/path/to/vllm-cache` | Persistent vLLM compile cache root (Docker volume). |
| `SAFETENSORS_FAST_GPU` | `1` | Faster safetensors → GPU weight load (common in `#vllm-rdna` / some Dockerfiles). |
| `TORCHINDUCTOR_COMPILE_THREADS` | `1` | Limit inductor threads (stability during first-boot compile). |

### llama.cpp RDNA2 fork (`#llamacpp`)

Prefer the **short stack** below. Long lists of `GGML_HIP_GFX1030_*` knobs are usually redundant with
`HSA_OVERRIDE_GFX_VERSION=10.3.0` and can clash with RCCL autotune — see
[Serving](../llama-cpp/rdna2-serving.md#recommended-env-stack).

| Variable | Example | What it does |
| --- | --- | --- |
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | **Required** to activate the V620/gfx1030 native RDNA2 profile. |
| `HSA_NO_SCRATCH_RECLAIM` | `1` | Avoid scratch reclaim issues on long-context runs. |
| `GGML_HIP_RDNA2_AUTO` | `1` | Enable automatic RDNA2 kernel selection. |
| `GGML_HIP_SAFE_STATE_IO` | `1` | Safer HIP state I/O; mitigates a known ROCm FA crash class (recommended default). |
| `GGML_TP_SHARDED_OUTPUT` | `1` | Sharded output head for tensor parallel (TP2+). |
| `GGML_CUDA_ALLREDUCE` | `nccl` | Use RCCL for tensor-parallel all-reduce (+10% tgen reported). |
| `GGML_HIP_GFX1030_P2P_ALLREDUCE` | `off` / `auto-expanded` | P2P all-reduce tuning; set `off` if RCCL misbehaves. Optional — not part of the short stack. |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` | Disable HIP graphs (some benchmark profiles use this). |
| `SPEC_SIDECAR` | `1` | Opt-in MTP/DFlash sidecar outside the main process (`#llamacpp` Sep 2026; pull latest fork). |
| `NCCL_P2P_LEVEL` | `PXB` / `PHB` | RCCL P2P topology level for all-reduce. |
| `NCCL_P2P_DISABLE` | `0` / `1` | Disable P2P in RCCL (fallback when topology is broken). |

### vLLM compilation / CUDA graphs

| Flag | Example | What it does |
| --- | --- | --- |
| `--compilation-config` | `'{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_ranges_endpoints":[]}'` | Enable CUDA graphs (preferred fast path on current **`-extras` 27B** images). |
| `--compilation-config` | `'{"cudagraph_mode":"PIECEWISE","compile_ranges_endpoints":[]}'` | Flash-Next on `rdna_extras` (`#vllm-rdna` Sep 16–17): **FULL** decode graphs corrupt; use **PIECEWISE**. |
| `--compilation-config` | `'{"mode":"NONE","cudagraph_mode":"FULL","compile_ranges_endpoints":[]}'` | Alternative graph mode without torch.compile. |
| `--compilation-config` | `'{"cudagraph_mode":"NONE"}'` | Disable graphs entirely. |
| `--enforce-eager` | — | Fallback: disable all graph capture. Use only when graphs crash. |

> `10.3.0` is the magic value for gfx1030 because the target decodes as
> `gfx` + `10` (major) `3` (minor) `0` (stepping) → `gfx1030`.

## V620 / gfx1030 PCI identity

Used by the [tuning](../tuning/power.md) scripts to match the right board:

| Board | PCI device | Subsystem | 4-tuple |
| --- | --- | --- | --- |
| Radeon PRO V620 (reference) | `1002:73a1` | `1002:0e34` | `1002:73a1:1002:0e34` |
| RX 6900 XT / 6800 (gfx1030) | `1002:73bf` | varies | — |

```sh
lspci -nn | grep '1002:73a1'   # find V620 reference boards
```

## Handy commands

```sh
rocminfo                              # full agent/GPU info; look for "Name: gfx1030"
rocminfo | grep -m1 -o 'gfx[0-9]*'    # just the target name
rocm-smi                              # live clocks, temps, VRAM, power, utilization
rocm-smi --showmeminfo vram           # VRAM usage
clinfo | grep -i board                # OpenCL board name
/opt/rocm/bin/rocminfo | grep -i wavefront   # confirm wave32 on RDNA2
```

## VRAM rules of thumb (LLMs)

| Card VRAM | Comfortable 4-bit model size |
| --- | --- |
| 16 GB (RX 6800/6800 XT/6900 XT/6950 XT) | 7B–13B, some 14B |
| ~30 GB (PRO W6800 / V620, **ECC on**) | up to ~30B–34B |
| 32 GB (same cards, [ECC off](../tuning/ecc.md)) | same class, extra KV / longer context |

Pro cards show ~30 GB until you disable ECC.

## FP16 vs BF16

RDNA2 has **no fast BF16**. Always prefer **FP16** for hot paths:

- PyTorch: pass `dtype=torch.float16`.
- vLLM: `--dtype float16` (see [Running vLLM](../vllm/overview.md)).
- The [`rdna_extras`](../vllm/fork.md) fork adds quantized (W4A16 / FP8) RDNA2 kernels to cut VRAM and
  sidestep BF16 entirely.
