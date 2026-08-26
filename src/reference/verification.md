# Verification status (WIP)

> **This wiki is work in progress.** Pages expanded from Discord (`#vllm-rdna`, `#llamacpp`, `#general`)
> and fork release notes mix **solid facts**, **fork-source claims**, and **community benchmarks** that
> have not been independently reproduced in this repo.

## Legend

| Status | Meaning |
|---|---|
| **Solid** | Hardware fact, upstream/ROCm docs, or fork source code confirms the behavior exists. |
| **Fork-source** | Confirmed in `blivioniag/vllm` `rdna2_extras` commits; may still need a fresh `-extras` image build. |
| **Community** | Reported in Discord or by a fork author; plausible but not wiki-verified on our hardware. |
| **Needs verify** | Should be re-tested before treating as guidance (image lag, single-host bench, or opinion). |
| **Opinion** | Subjective quality/speed trade-off from community testing. |

## Global gaps (read first)

| Topic | Status | Notes |
|---|---|---|
| Docker `-extras` images vs `rdna2_extras` HEAD | **Needs verify** | Wiki documents Aug 2026 fork commits. Images refreshed in place but **not confirmed** to include `0fdb1884+`. Re-pull and check logs. |
| Performance numbers (tok/s, × speedups) | **Community** | Unless marked **Solid**, treat throughput tables as single-host snapshots. |
| Env-var recipes | **Community** | Common stacks from Discord; A/B on your topology. |
| ROCm 7.2.1+ multi-card RCCL bug | **Community** | `#vllm-rdna` (fork author): stay on **7.2.0** or **7.14.0** |

---

## vLLM (`vllm/`)

### `running.md`

| Statement | Status | Verify how |
|---|---|---|
| Image tag matrix | **Needs verify** | [Docker Hub tags](https://hub.docker.com/r/blivioniag/vllm-rdna/tags) |
| `PYTORCH_ROCM_ARCH` list | **Solid** | `vllm-rdna-docker` build |
| Prefer `--dtype float16` | **Solid** | RDNA2 BF16 limitation |

### `configuration.md`

| Statement | Status | Verify how |
|---|---|---|
| Recommended env block | **Community** | Discord default stack |
| `VLLM_USE_V2_MODEL_RUNNER` +17% vs V1 | **Community** | `#vllm-rdna` bench comment |
| `VLLM_USE_AOT_COMPILE=0` / `VLLM_DISABLE_COMPILE_CACHE=1` | **Fork-source** | Multi-GPU TP workaround |
| CUDA graphs preferred over `--enforce-eager` | **Fork-source** | TP `allow_in_graph` fix; **needs verify** on image |
| `ROCM_ATTN` Triton hang | **Community** | Hours-long compile; use `RDNA_ATTN` |
| Throughput table (277/93/331 tok/s) | **Community** | Fork author, TP4, 4× V620 |
| Docker Compose ~24 t/s TP2 | **Community** | `#vllm-rdna` report |
| Cache sizes ~3 GB / ~700 MB | **Community** | Order-of-magnitude |

### `quantization.md`

| Statement | Status | Verify how |
|---|---|---|
| GPTQ → `RDNA2W4A16LinearKernel` | **Fork-source** | Startup logs |
| AWQ dense → `RDNA2W4A16LinearKernel` | **Fork-source** | Commits `73eb04a`/`5ac31e4`; **needs verify** on image |
| Older AWQ ~4–5 t/s (Triton) | **Community** | True on old images only |
| KV cache prefer float16 | **Opinion** | Community quality / agents |
| `int8_per_token_head` vs fp8 | **Community** | Faster TG than fp8 in limited `#vllm-rdna` testing |
| KVarN | **Community** | Skip for tool calling |
| MTP acceptance ~0.25 | **Community** | Model-dependent |
| MTP-2 hurts high concurrency (35B-A3B) | **Community** | TP4 `#vllm-rdna` 4-cell matrix |
| INT4 vdot2 fp16 dequant | **Solid** | ISA + fork code |
| Qwen3.8-27B AWQ needs `head_size=256` | **Fork-source** | Same as `fa_rdna2` commit; confirm on image |
| GDN hybrid ~93/331 tok/s | **Community** | Fork author bench |

### `fork.md`

| Statement | Status | Verify how |
|---|---|---|
| No WMMA on RDNA2 | **Solid** | Architecture |
| Kernel file list | **Solid** | Fork tree |
| `fa_rdna2` head_size=256 | **Fork-source** | Commit `03b2d91` |
| GDN decode ~9.3× vs Triton | **Community** | Fork microbench |
| GDN full HIP prefill chain | **Fork-source** | Commits `69d2efe`, `b53a7a2c` |
| TP `allow_in_graph` fix | **Fork-source** | Commit `b583d64` |

---

## llama.cpp (`llama-cpp/`)

### `building.md`

| Statement | Status | Verify how |
|---|---|---|
| Fedora + ROCm 7.2.0 build | **Solid** | Standard recipe |
| `--spec-draft-device` on TP2+ | **Community** | Discord pattern |

### `rdna2-benchmarks.md`

| Statement | Status | Verify how |
|---|---|---|
| All benchmark tables | **Community** | Author-reported; run your own before/after |
| Stock vs fork +56.8% (Qwen3.8-27B Q6_K MTP) | **Community** | `#llamacpp` matched A/B, byte-identical outputs |
| Q4_0 / Q8_0 fastest on fork | **Community** | `#harnesses` / `#llamacpp` |
| PR #10 / #12 status | **Needs verify** | Re-check fork PRs |

### `rdna2-speculative.md`

| Statement | Status | Verify how |
|---|---|---|
| DFlash2 draft Q4_K_M not Q8_0 | **Community** | Discord consensus |
| DFlash2 vs MTP bench (~30 vs ~40) | **Community** | Synthetic bench caveat documented |
| Full DFlash2 TP4 command | **Community** | Author production recipe |

### `rdna2-serving.md`

| Statement | Status | Verify how |
|---|---|---|
| Docker compose TP2 MTP4 | **Community** | `#llamacpp` example |
| KV checkpoint workaround | **Community** | `--ctx-checkpoints 0` |
| Multi-socket hurts TP | **Community** | Topology advice |
| 160 W / 140 W PSU workaround | **Community** | Transients on TP prefill; miner PSU / P620 cables |
| TP3 driver crash | **Community** | Prefer 2 or 4 GPUs |

---

## Reference & troubleshooting

| Area | Status | Notes |
|---|---|---|
| `reference/env-vars.md` tables | **Community** | Cheat-sheet; not exhaustive upstream API |
| `troubleshooting/vllm.md` graph fix | **Fork-source** | Same image-lag caveat |
| `troubleshooting/vllm.md` AMDSMI / missing `render` | **Community** | `#vllm-rdna` Docker compose |
| `tuning/power.md` Fedora path | **Solid** | From `v620_toolbox` powertuning |
| `tuning/power.md` Ubuntu 26.04 path | **Community** | `ubuntu_powertuning/` — validated kernel `7.0.0-30-generic`; re-verify after kernel upgrades |
| `tuning/power.md` Fedora Server 44 / kernel 6.19 | **Community** | Forum *Ice Lake + 4× V620* — powerfix + 180 W + ~7 W idle |
| `tuning/power.md` V620 slot-power / `setperflevel` | **Community** | `#llamacpp` — TDP from slot, not 8-pin |
| `tuning/ecc.md` two-reboot `ras_enable=0` | **Community** | lunnova on W6800; Discord reports V620 ECC-on (~30 GB). Confirm `rocm-smi` after two reboots |
| `tuning/p2p.md` validation | **Solid** | From `v620_toolbox` on Fedora + AMD CPU |
| `tuning/p2p.md` ~25 GB/s bandwidth | **Community** | Bandwidth ≠ inference speed |
| `tuning/p2p.md` host topology table | **Community** | Dual-socket, gen3 x4, PLX, TP3 — `#general` / `#llamacpp` |
| `tuning/p2p.md` Ice Lake P2P no-op / ~4% regression | **Community** | Ice Lake 4× V620 host — llama.cpp + vLLM |
| `tuning/p2p.md` Intel IOMMU-off breaks P2P | **Community** | Same thread; opposite of some generic docs |
| `tuning/p2p.md` PLX daisy-chain / heatsink fan | **Community** | `#general` PLX 88096 |
| `troubleshooting/llama-cpp.md` RADV crash / AMDVLK slow | **Community** | `#llamacpp` — prefer ROCm for TP |

---

## Checklist before production

1. `docker pull` latest `-extras` image; confirm fork commit in build metadata.
2. Grep logs for `Using RDNA2W4A16LinearKernel`.
3. Try CUDA graphs before `--enforce-eager`.
4. One matched A/B on your hardware.
5. Update this page when you confirm or refute a claim.

See [Contributing](../meta/contributing.md) and [Wiki structure](../meta/structure.md).
