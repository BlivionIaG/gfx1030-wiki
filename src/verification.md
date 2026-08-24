# Verification status (WIP)

> **This wiki is work in progress.** Pages expanded from Discord (`#vllm-rdna`, `#llamacpp`, `#general`)
> and fork release notes mix **solid facts**, **fork-source claims**, and **community benchmarks** that
> have not been independently reproduced in this repo. Use this page to decide what to trust before you
> tune production setups.

## Legend

| Status | Meaning |
|---|---|
| **Solid** | Hardware fact, upstream/ROCm docs, or fork source code confirms the behavior exists. |
| **Fork-source** | Confirmed in `blivioniag/vllm` `rdna2_extras` commits; may still need a fresh `-extras` image build. |
| **Community** | Reported in Discord or by a fork author; plausible but not wiki-verified on our hardware. |
| **Needs verify** | Should be re-tested before treating as guidance (image lag, single-host bench, or opinion). |
| **Opinion** | Subjective quality/speed trade-off from community testing, not a hard technical fact. |

## Global gaps (read first)

| Topic | Status | Notes |
|---|---|---|
| Docker `-extras` images vs `rdna2_extras` HEAD | **Needs verify** | Wiki documents Aug 2026 fork commits (AWQ dispatch, GDN HIP, TP graph fix). Images are refreshed in place but **we have not confirmed** the published tag includes commit `0fdb1884` (or later). Re-pull and check logs before trusting new claims. |
| Performance numbers (tok/s, × speedups) | **Community** | Unless marked **Solid**, treat all throughput tables as snapshots on specific hosts (often 4× V620, ROCm 7.14). Run matched before/after on your box. |
| Env-var recipes | **Community** | Common stacks from `#vllm-rdna` / `#llamacpp`; not every flag is required on every topology. A/B test. |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` on V620 | **Solid** | Required for the tested llama.cpp-rdna2 native profile; wrong override on other ASICs is harmful. |

---

## `vllm.md`

| Statement | Status | Verify how |
|---|---|---|
| Image tag matrix (ROCm / PyTorch / Triton versions) | **Needs verify** | Check [Docker Hub tags](https://hub.docker.com/r/blivioniag/vllm-rdna/tags) and image labels. |
| `PYTORCH_ROCM_ARCH` list baked into images | **Solid** | Documented in `vllm-rdna-docker` build; matches multi-RDNA intent. |
| Prefer `--dtype float16` on RDNA2 | **Solid** | RDNA2 BF16 is slow/emulated; widely documented. |
| Recommended env block (`VLLM_ROCM_USE_AITER=0`, etc.) | **Community** | Discord default stack; try with/without on your model. |
| `VLLM_USE_AOT_COMPILE=0` / `VLLM_DISABLE_COMPILE_CACHE=1` for multi-GPU TP | **Fork-source** | Workaround from fork testing; confirm it fixes your AOT replay errors. |
| CUDA graphs preferred over `--enforce-eager` | **Fork-source** | TP `allow_in_graph` fix in fork; still **needs verify** on your image tag. |
| Throughput table (277.99 / 92.66 / 330.97 tok/s, prefill ~1573) | **Community** | Fork author bench, Qwen3.8-27B-AWQ-INT4, TP4, 4× V620; not wiki-reproduced. |
| Docker Compose ~24 output t/s (GPTQ, TP2) | **Community** | `#vllm-rdna` report; model and tunableop settings matter. |
| GPTQ → `RDNA2W4A16LinearKernel` | **Fork-source** | Long-standing fork path; confirm in startup logs. |
| AWQ dense → `RDNA2W4A16LinearKernel` on gfx10x | **Fork-source** | Commits `73eb04a` / `5ac31e4`; **needs verify** on pulled image + `~151 output t/s` claim. |
| compressed-tensors / AWQ-vd routing | **Needs verify** | Format-dependent; benchmark per model. |
| Older AWQ ~4–5 t/s on 27B (Triton path) | **Community** | Historical; still true on **old** images without AWQ dispatch fix. |
| KV cache: prefer `float16` for long context | **Opinion** | Community quality report; int8/fp8 KV may be fine for some workloads. |
| `VLLM_USE_FA_RDNA2=1` requires fp16 KV | **Needs verify** | Discord note; confirm against current fork/env names. |
| MTP acceptance ~0.25 after v0.27.1 speculator update | **Community** | Model- and image-dependent. |
| INT4 uses vdot2 fp16 dequant (no native int4 ALUs) | **Solid** | Matches RDNA2 ISA and fork implementation. |
| Cache sizes ~3 GB Triton / ~700 MB torch_compile | **Community** | Order-of-magnitude from one host; yours will vary. |
| Image tags refreshed in place | **Community** | Operational policy; always `docker pull` before debugging. |

---

## `vllm_fork.md`

| Statement | Status | Verify how |
|---|---|---|
| RDNA2 has no WMMA (RDNA3+) | **Solid** | Architecture fact. |
| vdot2 fp16 dequant beats INT8 HIP path on V620 | **Community** | Fork author/community testing; not independent microbench here. |
| Kernel file list under `csrc/rocm/*_rdna2.cu` | **Solid** | Matches fork tree layout. |
| `fa_rdna2` supports `head_size=256` on gfx10x | **Fork-source** | Commit `03b2d91`; needed for Qwen3.8-27B-AWQ-INT4. |
| GDN decode ~9.3× faster than Triton at B=1 | **Community** | Fork microbench (`47d92b6c`); reproduce with bundled test. |
| Full GDN prefill chain in HIP (5 kernels + decode) | **Fork-source** | Commits `69d2efe`, `b53a7a2c` + follow-ups; replaces Triton FLA on hot path. |
| AWQ dense uses `RDNA2W4A16LinearKernel` | **Fork-source** | Same as `vllm.md`; confirm on image. |
| TP comm `allow_in_graph` fixes `_SimpleCData.__new__` | **Fork-source** | Commit `b583d64`; **needs verify** on your image. |
| Hybrid GDN may still use `ROCM_ATTN` for attention layers | **Community** | Expected layering; check vLLM backend logs per model. |
| `--linear-backend exllama` still relevant | **Community** | `#general` reports; A/B on your quant. |
| `pytest tests/kernels/...` commands | **Solid** | Standard fork test entry points (require ROCm build env). |

---

## `llama_cpp.md`

| Statement | Status | Verify how |
|---|---|---|
| Fedora + ROCm 7.2.0 build recipe | **Solid** | Standard llama.cpp HIP build; versions may drift. |
| `--spec-draft-device ROCm0` on TP2+ | **Community** | Discord-validated pattern; confirm devices match your enumeration. |
| Multi-GPU examples (122B MTP) | **Community** | Example commands; VRAM and P2P requirements vary. |

---

## `llama_cpp_rdna2_fork.md`

| Statement | Status | Verify how |
|---|---|---|
| Benchmark tables (pp512 / tg128, MTP peaks) | **Community** | Author-reported; page already warns to run your own before/after. |
| RCCL all-reduce +10% tgen / +20% prefill | **Community** | Author-reported on Qwen 122B; topology-dependent. |
| DFlash2 ~52 t/s coding / ~40 t/s at 64k | **Community** | Author + Discord; workload-specific. |
| DFlash2 draft quant: Q4_K_M not Q8_0 | **Community** | Discord consensus; quick A/B on acceptance + speed. |
| DFlash2 vs MTP on 20k bench (~30 vs ~40 t/s) | **Community** | Synthetic bench caveat is documented on page. |
| Full DFlash2 TP4 serve command | **Community** | Author production recipe; paths and batch sizes are host-specific. |
| Docker compose (MXFP4, TP2, MTP4) | **Community** | `#llamacpp` working example; not wiki-reproduced. |
| P2P ~25 GB/s between V620 pairs | **Community** | Bandwidth test report; does not imply faster inference. |
| Multi-socket hurts TP | **Community** | Topology advice from Discord. |
| KV checkpoint crash with tensor split | **Community** | Repro reported; `--ctx-checkpoints 0` is the workaround. |
| Tensor-split power spike / 160 W cap workaround | **Community** | PSU-specific; see [Power Tuning](./tuning_power.md). |
| PR #10 / #12 status and peak tables | **Needs verify** | PRs move; re-check fork README before relying on env block. |

---

## `troubleshooting.md`

| Statement | Status | Verify how |
|---|---|---|
| `TORCH_BLAS_PREFER_HIPBLASLT=0` for Navi 21 | **Solid** | Long-standing ROCm workaround. |
| Graph crash root cause = TP `allow_in_graph` | **Fork-source** | Fixed in fork; **needs verify** image includes fix. |
| AWQ on gfx10x should hit RDNA2 kernel (Aug 2026) | **Fork-source** | Same image-lag caveat as above. |
| Typical cache sizes on first boot | **Community** | Approximate. |
| RCCL / ACS / P2P troubleshooting order | **Community** | Sensible playbook; your topology may differ. |

---

## `reference.md`

| Statement | Status | Verify how |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` decoding to gfx1030 | **Solid** | Documented override semantics. |
| vLLM / llama.cpp env tables | **Community** | Cheat-sheet from Discord + fork docs; not exhaustive upstream API. |
| `VLLM_USE_AOT_COMPILE` / `VLLM_DISABLE_COMPILE_CACHE` | **Fork-source** | Multi-GPU workaround; confirm against current vLLM fork env names. |
| VRAM rules of thumb | **Opinion** | Rough guide; model arch and context dominate. |

---

## `tuning_p2p.md`

| Statement | Status | Verify how |
|---|---|---|
| Validated Fedora 43 + EPYC 7452 + 4× V620 | **Solid** | From `v620_toolbox` repo validation notes. |
| `rocminfo` 5 agents / 12/12 P2P ENABLED | **Solid** | Expected outcome when toolbox recipe succeeds. |
| P2P enabled but inference slower (gen3 x4, ACS) | **Community** | Discord reports; A/B with `NCCL_P2P_DISABLE=1`. |
| ~25 GB/s P2P bandwidth | **Community** | Bandwidth ≠ inference speed. |

---

## `resources.md`

| Statement | Status | Verify how |
|---|---|---|
| External links (repos, PR #52391, Docker Hub) | **Solid** | Links only; upstream PR status changes over time. |

---

## Recommended verification checklist

Before trusting a **community** or **fork-source** claim in production:

1. **Image / commit** — `docker pull blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0` (or rebuild from `rdna2_extras` HEAD). Check fork commit in image metadata or build logs.
2. **Kernel dispatch** — grep startup logs for `Using RDNA2W4A16LinearKernel` (GPTQ **and** AWQ dense on current fork).
3. **Graph capture** — run with `--compilation-config` first; only add `--enforce-eager` if capture still fails after image update.
4. **Throughput** — one matched A/B on your hardware (same prompt length, concurrency, power cap, P2P on/off).
5. **Contribute back** — if you confirm or refute a claim, open a PR updating this page (see [Contributing](./contributing.md)).

## How to help

- Mark rows **Solid** after you reproduce on documented hardware (note ROCm version, image tag, model).
- Flag **Outdated** claims with the image tag or date they failed.
- Prefer linking to fork commits, Discord thread dates, or bench scripts over bare tok/s numbers.
