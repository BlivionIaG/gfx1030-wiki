# Verification status (WIP)

> **This wiki is work in progress.** Pages expanded from Discord (`#vllm-rdna`, `#llamacpp`, `#general`)
> and fork release notes mix **solid facts**, **fork-source claims**, and **community benchmarks** that
> have not been independently reproduced in this repo.

## Legend

| Status | Meaning |
|---|---|
| **Solid** | Hardware fact, upstream/ROCm docs, or fork source code confirms the behavior exists. |
| **Fork-source** | Confirmed in `opengfx1030/vllm-rdna` `rdna_extras` (and historical `blivioniag/vllm` `rdna2_extras`) commits; may still need a fresh `-extras` image build. |
| **Community** | Reported in Discord or by a fork author; plausible but not wiki-verified on our hardware. |
| **Needs verify** | Should be re-tested before treating as guidance (image lag, single-host bench, or opinion). |
| **Opinion** | Subjective quality/speed trade-off from community testing. |

## Global gaps (read first)

| Topic | Status | Notes |
|---|---|---|
| Docker `-extras` images vs extras HEAD | **Needs verify** | Wiki documents Aug 2026 fork commits. Hub bake still clones historical `blivioniag/vllm` `rdna2_extras`. Re-pull and check logs. Official HEAD is `opengfx1030/vllm-rdna` `rdna_extras`. |
| Performance numbers (tok/s, × speedups) | **Community** | Unless marked **Solid**, treat throughput tables as single-host snapshots. |
| Env-var recipes | **Community** | Common stacks from Discord; A/B on your topology. |
| ROCm 7.2.1+ multi-card RCCL bug | **Community** | `#vllm-rdna` (fork author): stay on **7.2.0** or **7.14.0** |

---

## vLLM (`vllm/`)

Flash-Next narrative lives in `flash-next.md`, serve commands in `flash-next-serve.md`, and
Flash-Next failure modes in `troubleshooting/vllm-flash-next.md`. Rows below still name the page
they were written against (`recipes.md`, `overview.md`, `configuration.md`).



### `running.md`

| Statement | Status | Verify how |
|---|---|---|
| Image tag matrix | **Needs verify** | [Docker Hub tags](https://hub.docker.com/r/blivioniag/vllm-rdna/tags) |
| Hub `v0.28.0-extras` | **Needs verify** | `#vllm-rdna` Sep 18 test tag; CI not fully tracked — A/B vs `v0.27.1-extras` |
| `PYTORCH_ROCM_ARCH` list | **Solid** | `vllm-rdna-docker` build |
| Prefer `--dtype float16` | **Solid** | RDNA2 BF16 limitation |
| Default smoke model `cyankiwi/Qwen3.8-27B-AWQ-INT4` | **Community** | `#vllm-rdna` Sep 13 2026 day-to-day 27B pick |

### `recipes.md`

| Statement | Status | Verify how |
|---|---|---|
| Three stacks (Hub `-extras` / recipe GHCR / Flash-Next fork) | **Community** | `#vllm-rdna` Sep 2026; public READMEs |
| Recipe image `0.27.1-rocm7.2.3-gfx1030` + `preset:` | **Community** | Recipe `containers/README.md` |
| 1× V620: prefer MoE; Flash-Next not 1-card | **Community** | `#vllm-rdna` Sep 13 2026 |
| Gemma 4 unfinished on gfx1030 vLLM | **Needs verify** | `#vllm-rdna` Sep 13 — single-thread reports |
| Flash-Next weights `wtdcode` + `primitive-ai` PLE | **Community** | `#vllm-rdna` Aug 30 / Sep 13 |
| Sep 18 4× serve deltas (7 GiB KV, leftover workers, V2 blocked on Qwen4Exp) | **Community** | `#vllm-rdna` Sep 18 host-venv paste |
| MTP draft experts W4A16 (`quant_mtp_experts.py`) | **Community** | `#vllm-rdna` Sep 18 + overlay README; 3× PP3 tok/s **Needs verify** |
| PP3 `b33f9b6` graph KeyError / 1.5e9 KV | **Needs verify** | `#vllm-rdna` Sep 19 — public overlay patches; not a 7 GiB 4× cap |
| TP4 cyankiwi AWQ env block | **Community** | `#vllm-rdna` Aug 31 bench paste (paths sanitized) |
| Flash-Next 4× PIECEWISE serve (Sep 17) | **Community** | `#vllm-rdna` — GDN sanitizer `388a61b6f`; ~3331/73 @ 16k/1k c=8 |
| Flash-Next 4× `#17` `FULL_DECODE_ONLY` (Sep 23) | **Community** | Merged `opengfx1030/vllm-rdna#17`; second host confirmed 16k PP |
| Flash-Next vLLM PP4 ~1950 PP / decode collapse | **Needs verify** | `#vllm-rdna` Sep 19 — same host returned to TP4 (~1550 PP; ~30–35 t/s MTP-2) |
| QSA live-context `#15` ~1.8–2.0k PP / ~54–73 TG (TP4 PP1) | **Needs verify** | `#vllm-rdna` Sep 20 author-host table + merged `opengfx1030/vllm-rdna#15`. Sep 21: other 4× hosts on `#15` only got **~1.1–1.45k** PP. The unpublished recovery is **`#17`** (merged 23 Sep) |
| Recipe decode ~40–49 vs ~27 without TunableOp | **Community** | Recipe container README / troubleshooting |

### `configuration.md`

| Statement | Status | Verify how |
|---|---|---|
| Recommended env block | **Community** | Discord default stack |
| Custom AR disable vs PIX force | **Community** | `#vllm-rdna` Aug 2026 — pick by P2P topology |
| `VLLM_USE_V2_MODEL_RUNNER` +17% vs V1 | **Community** | `#vllm-rdna` bench comment (`-extras`) |
| `VLLM_USE_V2_MODEL_RUNNER=0` for Flash-Next long prompts | **Community** | `#vllm-rdna` Sep 2026 — stalls/timeouts fixed; ~68 t/s dense INT8 |
| `VLLM_USE_AOT_COMPILE=0` / `VLLM_DISABLE_COMPILE_CACHE=1` | **Fork-source** | Multi-GPU TP workaround |
| CUDA graphs preferred over `--enforce-eager` | **Fork-source** | TP `allow_in_graph` fix; **needs verify** on image |
| `ROCM_ATTN` Triton hang | **Community** | Hours-long compile; use `RDNA_ATTN` |
| Throughput table (277/93/331 tok/s) | **Community** | Fork author, TP4, 4× V620 |
| Docker Compose ~24 t/s TP2 | **Community** | `#vllm-rdna` report |
| Cache sizes ~3 GB / ~700 MB | **Community** | Order-of-magnitude |
| `SAFETENSORS_FAST_GPU=1` | **Community** | `#vllm-rdna` Sep 2026 + AMD optimization docs |
| `VLLM_CACHE_ROOT` + compile cache on | **Community** | Flash-Next / recipe startups (~5 vs ~10 min) |
| Flash-Next ~580–700 PP / ~50–53 decode | **Community** | Early Sep recipe; **superseded** by prefill campaign below |
| Flash-Next ~1000–1400 PP / ~53–100+ decode | **Community** | `#vllm-rdna` Sep 4–7 + Sep 13 suite + Flash-Next `RESULTS.md`; host-dependent |
| Flash-Next container ~40 t/s vs llama.cpp ~18–19 | **Community** | `#vllm-rdna` Sep 2026 same-host comparison |
| Intel AutoRound Flash-Next ~962 PP / ~41–56 decode | **Community** | `#vllm-rdna` Sep 10–11 + draft `opengfx1030/vllm-rdna#5`; 4× V620; dirty PR |
| `--max-num-batched-tokens 4096` fixes 128k PP cliff | **Community** | `#vllm-rdna` Sep 11 — 375 → ~950 PP; A/B on your scheduler |
| Intel AutoRound ~75 GB / BF16 PLE ~95 GiB | **Community** | Publisher checkpoint + PR validation notes |
| Quantized group-16 INT4 PLE incoherent | **Needs verify** | `#vllm-rdna` — not the published PR path (BF16 PLE) |
| GDN varlen second-seq output bug | **Fork-source** | `gdn_prefill_o_rdna2.cu` in draft PR #5; concurrent prefill |
| Recipe ports / Hybrid gfx10 opt-in | **Needs verify** | Draft `opengfx1030/vllm-rdna#6` — RDNA2 stays auto default |
| vLLM 0.29 defaults V2 runner | **Community** | `#vllm-rdna` Sep 11; Hub day-to-day still 0.27.1; Sep 18 `rdna_extra/v0.29.0` rebase |
| No MTP on Hub `v0.28.0-extras` | **Community** | `#vllm-rdna` Sep 18 — MTP aimed at 0.29 / Qwen4Exp |
| MTP can block int4 PLE fused decode | **Community** | `#vllm-rdna` Sep 18 — A/B MTP off on no-P2P hosts |
| ROCR 1.21 idle CPU spin on 7.14 | **Community** | TheRock#7051; patch in Flash-Next fork `ROCR-CPU-FIX.md` |
| EXL3 / Quark on gfx1030 | **Needs verify** | Experimental; not in published `-extras` tags yet |
| Intel AutoRound W4A16 Flash-Next | **Community** | `#vllm-rdna` Sep 10–11 — draft PR #5; not in Hub `-extras` |
| Official extras source is `opengfx1030/vllm-rdna` `rdna_extras` | **Solid** | `#vllm-rdna` Sep 2026 move; default branch `rdna_extras`; PRs go to the org. Hub `-extras` still from historical clone |
| Draft PRs `#5` / `#6` on org extras | **Needs verify** | Flash-Next AutoRound + recipe ports; do not treat as released |
| `opengfx1030/vllm-rdna` ready as published Docker | **Needs verify** | Bake repo is now `opengfx1030/vllm-rdna-docker`; Hub `-extras` still historical 0.27.1 |
| Flash-Next source is `rdna_extras` | **Community** | `#vllm-rdna` Sep 14–15 — leapdragon branch not updated; published containers may lag |
| Org 0.28 rebase / Flash-Next cherry-picks | **Community** | `#vllm-rdna` Sep 3–15: PLE load path on HEAD; older PR `#1` was first `rdna_ar` pass |
| Flash-Next KV ~24 GiB ≈ 300k tokens | **Community** | `#vllm-rdna` Sep 14 — 4× 32 GB still tight; 128 GB host RAM offload not enough |
| Flash-Next concurrency splits TG / tanks PP | **Community** | `#vllm-rdna` Sep 14 — ~70–80 t/s total vs ~70 single; PP ~500 |
| 2× V620 Flash-Next: llama.cpp not vLLM | **Community** | `#vllm-rdna` Sep 14 — RAM offload |
| vLLM 3-card → PP=3 (even TP) | **Community** | `#vllm-rdna` Sep 14 |
| Flash-Next V2=0: 100% util / ~40 W stall | **Community** | `#vllm-rdna` Sep 15 — TP4 leapdragon container; MoE wants V1 before 0.29 |
| Flash-Next PP3: leapdragon OK, `rdna_extras` garbage | **Needs verify** | `#vllm-rdna` Sep 15–16 — fresh pull still corrupt; GDN decode suspect. Sep 19 pin `b33f9b6` is a different bug |
| Flash-Next PP3 `b33f9b6` small-prefill KeyError | **Needs verify** | `#vllm-rdna` Sep 19 — capture sizes `[1,2,4,8]`; NaN+concurrent gone on that pin only |
| Flash-Next FULL graphs corrupt; PIECEWISE holds | **Needs verify** | `#vllm-rdna` Sep 16–17 — TP `rdna_extras`; `--max-num-seqs` 4–6 + batched 2048 |
| Flash-Next 4× PIECEWISE serve ~3331 PP / ~73 TG | **Community** | `#vllm-rdna` Sep 17 — 16k/1k c=8; GDN sanitizer `388a61b6f`; not Hub |
| `VLLM_RDNA_DENSE_GEMV=1` avoids wvSplitK fault | **Community** | `#vllm-rdna` Sep 17 — `wvSplitK_hf_sml_*` after profile; multi-GB coredumps |
| Flash-Next hybrid KV log ~2.5× high | **Needs verify** | `#vllm-rdna` Sep 17 — livelock between real pool and max-model-len |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` disables compile | **Community** | `#vllm-rdna` Sep 17 — ~39 vs ~55 t/s A/B |
| Prefix cache 0% before `e45dd5cb` | **Fork-source** | `#vllm-rdna` Sep 17 + `opengfx1030/vllm-rdna` commit; TTFT 12.2s → 0.3s |
| `vllm#55506` port `741e5bc3` | **Fork-source** | V2 mamba spec-decode tables; did not fix V1 0% / `!!!` |
| 3× leapdragon + org MoE HIP overlay | **Community** | `#vllm-rdna` Sep 17–18 — `alanoo81/flashnext-v620-pp3`; ~1078–2493 PP / ~57–63 TG |
| DeepSeek-V4 Flash INT4 + gfx1030 tree | **Needs verify** | `#vllm-rdna` Sep 17 — `yiminyuan` HF + `gfx1030/v0.28.0`; intended TP=4 PP=2 |
| No Q3 on vLLM Flash-Next | **Community** | `#vllm-rdna` Sep 17 — stay W4A16 / AWQ; EXL3 3bpw experimental |
| `--no-enable-prefix-caching` KV offload non-starter | **Community** | `#vllm-rdna` Sep 16 — QSA/RAM packs; prefer LMCache |
| Intel AutoRound 32–128k ~1296–1393 PP / ~56–73 TG | **Community** | `#vllm-rdna` Sep 16 — 4× V620; pin `4c67bf6…`; int8 decode-shadow quality |
| QSA live-context bound merged on `rdna_extras` | **Fork-source** | `opengfx1030/vllm-rdna#15` merged 20 Sep 2026; Hub `-extras` lags |
| QSA `#15` combined-stack ~1830–2040 PP / ~54–73 TG | **Needs verify** | PR recorded table + `#vllm-rdna` Sep 20 author-host; **not reproduced** on two other 4× hosts Sep 21 (~1.07–1.45k PP). The unpublished recovery is **`#17`** |
| QSA `#15` public-merge prefill ~1.1–1.45k PP | **Community** | `#vllm-rdna` Sep 21 — two 4× hosts after `#15` only |
| V620 fast stack `#17` merged | **Fork-source** | `opengfx1030/vllm-rdna#17` merged 23 Sep 2026; resident W4A16 + Mamba `#55450` + `FULL_DECODE_ONLY` |
| `#17` 16k PP ~1958 / ~1983; second host ~1984 / ~1985 | **Community** | PR-head table + `#vllm-rdna` Sep 23 confirm after TunableOp regen |
| TunableOp rows locked to rocBLAS hash | **Fork-source** | `rdna_extras/tunableop` README; mismatch ~25% PP drop / first GEMM abort |
| `#20` PIECEWISE capture (FULL→piecewise redirect) | **Fork-source / Community** | Merged `opengfx1030/vllm-rdna#20` (24 Sep); decode ~26–34 t/s **while redirect was on** |
| Keep-FULL `FULL_AND_PIECEWISE` (`6c26c78d`) | **Fork-source** | `rocm_full_executes_as_piecewise → False` on `rdna_extras` HEAD; `#vllm-rdna` Sep 24; docs `V620-FULL-AND-PIECEWISE.md` |
| `#19` unquantized MTP drafter load | **Fork-source** | Merged `opengfx1030/vllm-rdna#19` (24 Sep) |
| `#18` GPTQ `BLOCK_KN_SIZE` 128→256 | **Needs verify** | Open PR; hold for M>32…2048 sweep (`#vllm-rdna` Sep 24) |
| `#22` two-shot `rdna_ar` vs force custom AR | **Needs verify** | Open PR; force custom AR crashed one host |
| `#23` PCIe P2P KV disagg prefill/decode | **Needs verify** | Open PR; local PLX-switch split — not a recipe |
| Flash-Next 6× `PP=3`+`TP=2` / `PP=6` | **Needs verify** | `#vllm-rdna` Sep 24 — min 3 cards; no published 6× table |
| Qwen3.8-27B fp16 stock 0.29.0 TP4 W6800X Duo | **Community** | `#benchmarks` Sep 24 — 1k–255k ctx, no MTP; ~1394→560 PP / ~29→19 TG |
| Flash-Next vision pixel cap (`max_pixels=1605632`) | **Fork-source** | `rdna_extras` `scripts/serve_gfx1030_flashnext.sh`; `--limit-mm-per-prompt` alone is not enough |
| Flash-Next vision ~15–20 s/image + runtime OOM | **Community** | `#vllm-rdna` / `#general` Sep 21 — works with launcher flags; no reserved vision pool; PP3 especially tight |
| Flash-Next `FULL_AND_PIECEWISE` executes as PIECEWISE on ROCm | **Fork-source (historical)** | Launcher comment 18 Sep 2026 + `#20`-era redirect. **Superseded** on HEAD by `6c26c78d` (`rocm_full_executes_as_piecewise → False`) — prefer `FULL_AND_PIECEWISE` again |
| QSA fused MTP-3 draft decode no gain | **Needs verify** | `#vllm-rdna` Sep 20 — one host; prefill held ~2k |
| INT8 decode shadows excluded from `#15` | **Fork-source** | PR body; Discord ~42 → ~55 t/s without MTP is **Needs verify** / quality risk |
| vllm-rdna-qa playbook | **Needs verify** | `#vllm-rdna` Sep 17 — public repo; maintainer QA only |
| Flash-Next PP3 + MTP on 3× V620 | **Needs verify** | `#vllm-rdna` Sep 15 — leapdragon image; uneven `VLLM_PP_LAYER_PARTITION`; local int8 patches |
| Upstream MTP under PP (`vllm#46994`) | **Fork-source** | Merged Sep 2026; not in Hub `-extras` 0.27.1 |
| `RDNA2W4A16MoEExperts` raises concurrency | **Community** | `#vllm-rdna` Sep 15 — confirm kernel in logs |
| Native RDNA2 FA WIP on Flash-Next HEAD | **Community** | `#vllm-rdna` Sep 15 — A/B `VLLM_USE_RDNA2_FA=0`; Hub `-extras` still uses `1` |
| `rdna_ar` not a recommended default | **Community** | `#vllm-rdna` Sep 15–24 — prefer simpler `RDNA_AR` over force custom AR; open `#22`; one host crash on force custom AR |
| hippihx kernel zoo | **Needs verify** | `#hippihx` Sep 16 — public repo; not in published images |
| Upstream vLLM KV CPU/SSD offload ~1 t/s | **Community** | `#vllm-rdna` Sep 15 — worse on Mamba/SSM; prefer LMCache when it lands |
| `vllm#57160` private pinned CPU KV tensors | **Needs verify** | `#vllm-rdna` Sep 20 — mainline ROCm; not confirmed on `rdna_extras` |
| LMCache RDNA Docker integration | **Needs verify** | `#lmcache` Sep 23–24: 0.5.6.dev wheel built; official kernels CDNA-only; hybrid Qwen HMA + `LMCacheConnectorV1` not HMA-capable. Still no working compose |
| Upstream vLLM 0.28 / 0.30 gfx1030 support | **Needs verify** | Official 0.28–0.30 still omit Navi 21 (`#general` Sep 22: 0.30 + ROCm 10.0). Keep extras |

### `quantization.md`

| Statement | Status | Verify how |
|---|---|---|
| GPTQ → `RDNA2W4A16LinearKernel` | **Fork-source** | Startup logs |
| AWQ dense → `RDNA2W4A16LinearKernel` | **Fork-source** | Commits `73eb04a`/`5ac31e4`; **needs verify** on image |
| Older AWQ ~4–5 t/s (Triton) | **Community** | True on old images only |
| KV cache prefer float16 | **Opinion** | Community quality / agents |
| `int8_per_token_head` vs fp8 | **Community** | Faster TG than fp8 in limited older `#vllm-rdna` testing; **not** the QSA path |
| QSA / Flash-Next `--kv-cache-dtype fp8` startup fail | **Community** | `#vllm-rdna` Sep 22 — QSA backends unquantized-only + AITER/CDNA gate |
| KVarN | **Community** | Skip for tool calling |
| MTP acceptance ~0.25 | **Community** | Model-dependent |
| MTP-2 hurts high concurrency (35B-A3B) | **Community** | TP4 `#vllm-rdna` 4-cell matrix |
| INT4 vdot2 fp16 dequant | **Solid** | ISA + fork code |
| Qwen3.8-27B AWQ needs `head_size=256` | **Fork-source** | Same as `fa_rdna2` commit; confirm on image |
| GDN hybrid ~93/331 tok/s | **Community** | Fork author bench |
| EXL3 9B / Quark W4A16 | **Needs verify** | `#vllm-rdna` Sep 2026 — experimental; Sep 19 `rdna_extras` is 3inst-only; Sep 24 `rdna_extra/v0.29.0` adds **mul1** |
| INT8 KV (not decode shadows) | **Needs verify** | `#vllm-rdna` Sep 23–24 — community testing; QSA backends still unquantized-only |
| HIP MoE non-deterministic vs Triton | **Needs verify** | `#vllm-rdna` Sep 19 — ~3.5% top-token drift; `global_atomic_add_f32` not landed (blocked on 0.29.0) |
| Intel AutoRound W4A16 Flash-Next | **Community** | `#vllm-rdna` Sep 10–11 + draft `opengfx1030/vllm-rdna#5`; Sep 20 QSA `#15` used this pack + BF16 PLE (not `wtdcode` + PLE-quant) |
| INT8 decode shadows not in QSA `#15` | **Fork-source** | `#vllm-rdna` Sep 20 — excluded from merge; quality loss already noted Sep 16 |

### `fork.md`

| Statement | Status | Verify how |
|---|---|---|
| Fork landscape: official `opengfx1030/vllm-rdna` `rdna_extras`; Flash-Next still separate; Hub still historical | **Solid** | Org repo + default branch; `#vllm-rdna` Sep 2026 |
| Consolidation / 0.28 gap audit | **Community** | `#vllm-rdna` Sep 3–7 channel notes; treat as WIP |
| No WMMA on RDNA2 | **Solid** | Architecture |
| Kernel file list | **Solid** | Fork tree (`rdna_extras`) |
| `fa_rdna2` head_size=256 | **Fork-source** | Commit `03b2d91` |
| Public extras `fa_rdna2` dropped FP16 pieces | **Community** | `#vllm-rdna` Sep 22 — maintainer focused on W4A16 validation |
| `2kiss/flash-attention-rdna2` | **Needs verify** | `#vllm-rdna` Sep 22–24 — independent kernel; **stale**; org `fa_rdna2` currently disabled after `#17` |
| GDN decode ~9.3× vs Triton | **Community** | Fork microbench |
| GDN full HIP prefill chain | **Fork-source** | Commits `69d2efe`, `b53a7a2c` |
| TP `allow_in_graph` fix | **Fork-source** | Commit `b583d64` |
| HIP MoE non-deterministic (~3.5% top-token drift) | **Needs verify** | `#vllm-rdna` Sep 19 — Triton MoE bit-identical; f32 atomic not landed |
| QSA live-context prefill bound | **Fork-source** | Merged `opengfx1030/vllm-rdna#15`; Python-only scoring width change |
| Upstream `vllm#57160` CPU KV offload pin | **Needs verify** | `#vllm-rdna` Sep 20 — mainline, not 0.29; may not apply to current `rdna_extras` |

### `host-venv.md`

| Statement | Status | Verify how |
|---|---|---|
| 7.2.0 index `download.pytorch.org/whl/rocm7.2`, torch 2.12.0, triton 3.5.1 | **Solid** | `vllm-rdna-docker` `base-rocm720` |
| 7.2.0 host paste also had `triton-rocm==3.7.1` | **Community** | `#general` 1 Aug 2026 |
| 7.14.0 index `repo.amd.com/rocm/whl-multi-arch/`, torch `2.12.0+rocm7.14.0`, triton 3.7.1 | **Solid** | `docker-bake.hcl` `base-rocm714` |
| Published `7.14.0` tag described as PyTorch 2.13.0 | **Needs verify** | README + [Running](../vllm/running.md) vs current bake pin 2.12.0 |
| `pytorch.org/whl/rocm7.14` has no cp312 x86_64 torchaudio | **Community** | `#vllm-rdna` 21 Sep 2026 |
| Device wheels `rocm-sdk-device-gfx1030` + `amd-torch-device-gfx1030` | **Solid** | `Dockerfile.base` `INSTALL_ROCM_SDK_DEVICE=1` |
| Missing device wheels → `hipErrorInvalidImage` on index torch | **Fork-source** | `Dockerfile.base` comment |
| `[device-gfx1030]` is only an extra on multi-arch index wheels | **Solid** | TheRock device extras; source wheels have no such extra |
| From-source gfx1030 wheels: install the `.whl`, no device extra | **Community** | `uv` warning on `~/wheels/rdna2/` (`torch==2.12.0+git6bbd260`) |
| FA allowlist drops gfx1030; Triton FA at runtime | **Solid** | `Dockerfile.base` |
| causal-conv1d pin `4f6ae4e` + `HIP_ARCHITECTURES` | **Solid** | `Dockerfile.vllm` |
| Do not pass `--torch-backend=rocm7.14` | **Community** | That flag selects the PyTorch index that failed on 21 Sep |
| `_rocm_sdk_*` `LD_LIBRARY_PATH` | **Community** | `#vllm-rdna` serve blocks; same dirs as the image `ldconfig` |

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
| Long-context quant sweep (Q8 wins) | **Community** | `#benchmarks` Aug 30 — single host |
| ROCm 7.1 vs 10.0 Ice Lake sweep | **Community** | Close numbers; prefer 7.14 for RCCL day-to-day |
| Flash-Next ~28 t/s llama.cpp vs ~60 t/s vLLM | **Community** | `#llamacpp` 4× V620 comparison |
| Flash-Next APEX GGUF ~370 PP / ~26 t/s (2× V620) | **Community** | `#llamacpp` Sep 11 — `mudler/Qwen3.8-Flash-Next-APEX-GGUF` |
| Flash-Next Q4 ~6 t/s on 2× V620 (LocalAI) | **Community** | `#llamacpp` Sep 10 — n-gram on NVMe |
| Flash-Next IQ4_XS ~150–200 PP / ~25 t/s (2× V620) | **Community** | `#llamacpp` / `#vllm-rdna` Sep 14–15 — layer split + `--n-cpu-moe` |
| Flash-Next llama.cpp ~300 PP / ~25 t/s “normal” (2×) | **Community** | `#llamacpp` Sep 16 — drops ~200/15 on long ctx |
| Flash-Next MTP needs PR / Unsloth desktop | **Community** | `#llamacpp` Sep 17 — stock / Studio bundled llama.cpp refused MTP |
| Unsloth Studio bundled llama.cpp loads Flash-Next MTP | **Needs verify** | `#llamacpp` Sep 20 — one host; called a same-day bundle fix |
| Flash-Next 4× patched MTP ~400 PP / ~27–45 t/s | **Needs verify** | `#llamacpp` Sep 17 — others call 400 PP low |
| Flash-Next 3× + 32 GB RAM does not load | **Community** | `#llamacpp` Sep 17 — n-gram / host RAM bound |
| Flash-Next 3× + MTP needs ~64 GB RAM | **Community** | `#llamacpp` Sep 17 — later report; SSD offload untested |
| Flash-Next MTP: stock llama.cpp can load when forks refuse | **Needs verify** | `#llamacpp` Sep 18 — single-thread |
| Flash-Next MTP on `markldn` / `okigan` trees, no TP | **Needs verify** | `#llamacpp` Sep 19 — MTP loaded; tensor parallel did not |
| DeepSeek-V4 llama.cpp TP4 ~22 t/s | **Needs verify** | `#llamacpp` Sep 17 — kernel unpublished |
| Flash-Next UD-IQ3_XXS ~530 PP / ~30 t/s (2× V620, stock ROCm) | **Community** | `#general` / `#forum` Sep 15 — official llama.cpp ROCm; ~10 t/s at 80k+; Vulkan ~25% slower |
| Flash-Next n-gram table ~50 GB RAM | **Community** | `#llamacpp` Sep 10 — 4× V620 to avoid storage offload |
| PR #10 / #12 status | **Needs verify** | Re-check fork PRs |

### `rdna2-speculative.md`

| Statement | Status | Verify how |
|---|---|---|
| DFlash2 draft Q4_K_M not Q8_0 | **Community** | Discord consensus |
| DFlash2 vs MTP bench (~30 vs ~40) | **Community** | Synthetic bench caveat documented |
| DFlash hurts PP more than MTP | **Community** | `#llamacpp` Aug 28 |
| MTP n=3 often beats n=4 (27B Q8 TP4) | **Community** | Real-prompt A/B; acceptance dropped at n=4 |
| Flash-Next TP experimental / deferred | **Community** | Fork update + forum benches; layer-split only |
| Flash-Next llama.cpp << vLLM | **Community** | `#llamacpp` / forum Sep 2026 — except 2-card / RAM-offload |
| 2× V620 IQ4_XS layer-split + ngram-mod | **Community** | `#llamacpp` / `#vllm-rdna` Sep 14–15 — host-specific `-t` |
| `SPEC_SIDECAR=1` MTP path | **Community** | Fork maintainer tip; pull latest |
| Sidecar GGUF identity / Unsloth vs other Q8 | **Community** | `#llamacpp` Sep 2026 — match publisher families |
| DFlash2 aperture violation crash | **Community** | `#llamacpp` Sep 2026 — pull latest / A/B MTP |
| `--spec-draft-p-min` ≠ 0 disarms MTP | **Community** | `#benchmarks` Sep 2026 tip |
| Full DFlash2 TP4 command | **Community** | Author production recipe |
| MTP + LCP prompt-cache position desync | **Community** | `#llamacpp` Sep 10 — HTTP 200 / no tokens; `--ctx-checkpoints 0` does not fix |
| `markldn` / `okigan` MTP trees, no tensor parallel | **Needs verify** | `#llamacpp` Sep 19 — one host |
| Unsloth Studio bundled llama.cpp loads Flash-Next MTP | **Needs verify** | `#llamacpp` Sep 20 — one host; re-try bundled binary first |

### `rdna2-serving.md`

| Statement | Status | Verify how |
|---|---|---|
| Short env stack preferred | **Community** | `#llamacpp` — long `GFX1030_*` lists can clash with RCCL |
| `GGML_HIP_SAFE_STATE_IO=1` FA workaround | **Community** | Fork maintainer note |
| ubatch ~1024 per GPU | **Community** | `#llamacpp` PP tuning tip |
| Docker compose TP2 MTP4 | **Community** | `#llamacpp` example |
| KV checkpoint workaround | **Community** | `--ctx-checkpoints 0` |
| Multi-socket hurts TP | **Community** | Topology advice |
| 160 W / 140 W PSU workaround | **Community** | Transients on TP prefill; miner PSU / P620 cables |
| TP3 driver crash | **Community** | Prefer 2 or 4 GPUs |
| `hipcub-devel` for GPU sampling | **Community** | Forum / `#llamacpp` build note |
| Single V620 Q4_0 ~39–49 t/s (27B) | **Community** | `#llamacpp` Sep 2026 |
| Embedder idle +~40 W/GPU | **Community** | `#llamacpp` — nomic/etc. alongside chat |
| Fork day-to-day on ROCm 7.14 | **Community** | `#llamacpp` Sep 2026 — not mid-7.2.x |
| Concurrent `--parallel` crush (~5–9 t/s) | **Community** | `#llamacpp` Sep 2026 V620 multi-agent |
| Upstream PR #22466 fast tensor loads | **Needs verify** | Community <35 s on 122B-Q4; watch merge |
| Long-ctx prefer Q6+/Q8 | **Opinion** | `#llamacpp` Sep 2026 quality reports |

---

## Reference & troubleshooting

| Area | Status | Notes |
|---|---|---|
| `reference/env-vars.md` tables | **Community** | Cheat-sheet; not exhaustive upstream API |
| `troubleshooting/vllm.md` graph fix | **Fork-source** | Same image-lag caveat |
| `troubleshooting/vllm.md` AMDSMI / missing `render` | **Community** | `#vllm-rdna` Docker compose |
| `tuning/power.md` Fedora path | **Solid** | From `v620_toolbox` powertuning |
| `tuning/power.md` Ubuntu 26.04 path | **Community** | `ubuntu_powertuning/` — validated kernel `7.0.0-30-generic`; re-verify after kernel upgrades |
| `tuning/power.md` Fedora Server 44 / kernel 6.19 | **Community** | Ice Lake 4× V620 host — power floor + 180 W + ~7 W idle |
| `tuning/power.md` V620 slot-power / `setperflevel` | **Community** | `#llamacpp` — TDP from slot, not 8-pin |
| `tuning/power.md` soft unlock (`amd-v620-soft-unlock`) | **Community** | Upstream README + `#general` passthrough reports; wiki-unverified TFLOPS / 232–275 W range |
| `setup/hardware.md` prefer soft unlock over W6800 flash | **Community** | Soft unlock keeps 72 CUs; W6800 flash → 54 CU |
| `tuning/ecc.md` two-reboot `ras_enable=0` | **Community** | lunnova on W6800; Discord reports V620 ECC-on (~30 GB). Confirm `rocm-smi` after two reboots |
| `tuning/ecc.md` Vulkan memtest before ECC-off | **Community** | `#general` Sep 23 — quiet dmesg ≠ clean card; leave ECC on if errors |
| `tuning/p2p.md` validation | **Solid** | From `v620_toolbox` on Fedora + AMD CPU |
| `tuning/p2p.md` Ubuntu 26 / 7.0 kernel gap | **Community** | `#vllm-rdna` Sep 24 — validated Pre-7 ≤6.19 or Post-7 ≥7.1; 7.0 not the Fedora recipe |
| `tuning/p2p.md` ~25 GB/s bandwidth | **Community** | Bandwidth ≠ inference speed |
| `tuning/p2p.md` host topology table | **Community** | Dual-socket, gen3 x4, PLX, TP3 — `#general` / `#llamacpp` |
| `tuning/p2p.md` Ice Lake P2P no-op / ~4% regression | **Community** | Ice Lake 4× V620 host — llama.cpp + vLLM |
| `tuning/p2p.md` Intel IOMMU-off breaks P2P | **Community** | Ice Lake host; opposite of some generic docs |
| `tuning/p2p.md` PLX daisy-chain / heatsink fan | **Community** | `#general` PLX 88096 |
| `tuning/p2p.md` PEX88096 8-riser / H12D-8D x8x8 | **Community** | `#general` Sep 19 — 6 GPUs still awkward; 2/4/8 vs any even count |
| `tuning/p2p.md` X570 x8x8 gen4 + single x16 PEX88096 uplink | **Community** | `#general` Sep 21 — cards talk on the switch bus |
| `troubleshooting/llama-cpp.md` RADV crash / AMDVLK slow | **Community** | `#llamacpp` — prefer ROCm for TP |
| `troubleshooting/llama-cpp.md` FA `max_blocks_per_sm` abort | **Community** | head-256 occupancy 0 on gfx1030; q8 KV needs FA |
| `troubleshooting/llama-cpp.md` DAX mmap SVM oops | **Community** | `--no-mmap` mandatory on `dax=always` |
| `troubleshooting/general.md` CPU governor / unsupported AMDGPU punt | **Community** | Flash-Next PP; Polaris/WX4100-in-box ROCm skip; unbind > ROCR_VISIBLE alone |
| `troubleshooting/vllm.md` leftover EngineCore / PleOffloadWorker | **Community** | `#vllm-rdna` Sep 18 |
| `troubleshooting/vllm.md` TunableOp rocBLAS hash lock | **Fork-source** | `#vllm-rdna` Sep 23–24 + `tunableop/` README |
| `troubleshooting/vllm.md` Flash-Next vision pixel-cap / runtime OOM | **Fork-source / Community** | Launcher `max_pixels`; `#vllm-rdna` Sep 21 ~15–20 s/image |
| `troubleshooting/vllm.md` FP8 KV rejected on QSA | **Community** | `#vllm-rdna` Sep 22 — unquantized-only backends + AITER/CDNA |
| `troubleshooting/vllm.md` no-P2P PYNCCL / MTP vs PLE | **Community** | `#vllm-rdna` Sep 18 |
| `troubleshooting/vllm.md` MTP concurrency / PLE stall | **Community** | `#vllm-rdna` — recipe PRs + P2P A/B |
| `troubleshooting/vllm.md` ROCR idle CPU spin | **Community** | TheRock 7.14 / ROCR 1.21 — Flash-Next fork patch |
| `troubleshooting/vllm.md` Flash-Next V2=0 long prompts | **Community** | `#vllm-rdna` Sep 2026; Sep 15 100%/~40 W stall |
| `troubleshooting/vllm.md` Flash-Next PP3 `rdna_extras` corruption | **Needs verify** | `#vllm-rdna` Sep 15–16 — leapdragon OK; fresh pull still bad |
| `troubleshooting/vllm.md` PP3 `b33f9b6` graph KeyError | **Needs verify** | `#vllm-rdna` Sep 19 — `[1,2,4,8]` capture; 1.5e9 KV cap; 0% hit-rate log unreliable |
| `troubleshooting/vllm.md` Flash-Next FULL-graph corruption | **Needs verify** | `#vllm-rdna` Sep 16–17 — PIECEWISE workaround |
| `troubleshooting/vllm.md` wvSplitK / `VLLM_RDNA_DENSE_GEMV` | **Community** | `#vllm-rdna` Sep 17 |
| `troubleshooting/vllm.md` hybrid KV overstated / livelock | **Needs verify** | `#vllm-rdna` Sep 17 |
| `troubleshooting/vllm.md` prefix cache 0% / `e45dd5cb` | **Fork-source** | `#vllm-rdna` Sep 17 + org extras commit |
| `vllm/recipes.md` 3× MoE HIP overlay | **Community** | `#vllm-rdna` Sep 17–18 — public temp repo |
| `vllm/recipes.md` DeepSeek-V4 Flash | **Needs verify** | `#vllm-rdna` Sep 17 — not Hub |
| `setup/hardware.md` no NVIDIA+V620 TP | **Community** | `#general` Sep 17 — mixed prefill collapsed |
| `vllm/recipes.md` 4× PIECEWISE Flash-Next serve | **Community** | `#vllm-rdna` Sep 17 — host venv, not Hub |
| `troubleshooting/vllm.md` upstream KV offload ~1 t/s | **Community** | `#vllm-rdna` Sep 15 — Mamba/SSM worse |
| `troubleshooting/vllm.md` `vllm#57160` pinned CPU KV | **Needs verify** | `#vllm-rdna` Sep 20 — mainline; not on `rdna_extras` yet |
| `vllm/recipes.md` 3× PP3 + MTP VRAM squeeze | **Needs verify** | `#vllm-rdna` Sep 15 — local patches |
| `setup/hardware.md` V620 vs 32 GB MI50 more PP less TG | **Community** | `#general` Sep 15 — first-pass A/B |
| `setup/host-firmware.md` GpuMMIOFix every-boot remap | **Needs verify** | `#motherboard` Sep 15–16 — suggested for WS-Z270 ReBAR |
| `troubleshooting/vllm.md` 128k PP cliff / batched-tokens 4096 | **Community** | `#vllm-rdna` Sep 11 — Intel AutoRound draft |
| `troubleshooting/vllm.md` shm_broadcast / Triton vs RCCL | **Community** | `#vllm-rdna` Sep 7 — wait + cache; stay on 7.14 |
| `troubleshooting/llama-cpp.md` DFlash2 / sidecar / concurrent | **Community** | `#llamacpp` Sep 2026 |
| `troubleshooting/llama-cpp.md` MTP LCP position desync | **Community** | `#llamacpp` Sep 10 — `ctx_dft` / M-RoPE `X < Y` |
| `troubleshooting/general.md` V620 thermals / graphene pads | **Community** | `#general` Sep 2026 — mixed repaste reports |
| `vllm/overview.md` MoE sweet spot / Flash-Next needs 4 cards | **Opinion** | `#general` Sep 2026 workload consensus |
| `tuning/power.md` 8× @ 180 W ≈ 1440 W + HELA 2050 | **Community** | `#forum` Sep 2026 build notes |
| `tuning/power.md` 1600 W for 5× stock 250 W; 1200 W tight for 4× | **Community** | `#general` Sep 22–23 — 0.7 A blowers OK at 180 W; ~86 °C at 250 W |
| `vllm/recipes.md` 3× overlay unmaintained / KV dump loops | **Community** | `#general` Sep 23 — cap ~3 agents on long ctx; ~45 t/s / ~1100 PP on one 3-card host |
| `tuning/power.md` 180 W token-cost economics | **Community** | `#llamacpp` vs stock 250 W |
| `tuning/p2p.md` SlimSAS / passive riser notes | **Community** | `#general` cabling |
| `tuning/p2p.md` external Xpander / narrow uplink | **Community** | `#general` Sep 2026 — long-ctx collapse |
| `tuning/p2p.md` layer vs tensor split explainer | **Community** | `#benchmarks` Sep 2026 |
| `setup/installing-rocm.md` avoid mid-7.2.x (e.g. 7.2.4) | **Community** | Same RCCL pin as 7.2.1+; prefer 7.2.0 or 7.14.0 |
| `setup/hardware.md` W6800 BIOS on V620 → 54 CU | **Community** | `#general` Sep 2026 PSA — stay on stock V620 VBIOS |
| `setup/host-firmware.md` Dell Precision MMIO High / `mmiohsize` | **Community** | `#motherboard` / `#general` Sep 12–14 2026 — dump your own UEFI; 0x03 vs 0x04 |
| `setup/host-firmware.md` T7820 4× V620 + PEX880xx POSTed after SR-IOV off | **Needs verify** | Single-host `#motherboard` Sep 14 2026 — board-specific hidden vars |
| `setup/host-firmware.md` CMOS loss re-enables SR-IOV / disables ReBAR | **Community** | `#general` Sep 23 — MMIO / missing GPUs until BIOS restored |
| `setup/host-firmware.md` 256 GB MMIO window for 4× V620 | **Community** | `#motherboard` — 64 GB too small; no known 128 GB option |
| `tuning/p2p.md` 4-slot vs 5-slot PLX card width | **Community** | `#general` Sep 12 2026 |
| `tuning/p2p.md` 5 GPUs → TP=4 + 1 standalone | **Community** | `#general` Sep 11 2026 |
| `troubleshooting/llama-cpp.md` lemonade-sdk M-RoPE A/B | **Needs verify** | `#llamacpp` Sep 13 — one host; not the RDNA2 fork |
| `troubleshooting/general.md` blower vs unducted 120 mm | **Community** | `#general` Sep 13 2026 |

### `setup/host-firmware.md`

| Statement | Status | Verify how |
|---|---|---|
| Dell Precision MMIO High / `mmiohsize` 0x03 or 0x04 | **Community** | `#motherboard` / `#general` Sep 12–14 — dump your own UEFI |
| T7820 4× V620 + PEX880xx POSTed after SR-IOV off | **Needs verify** | Single-host `#motherboard` Sep 14 — board-specific |
| CMOS / BIOS loss re-enables SR-IOV / disables ReBAR | **Community** | `#general` Sep 23 — fewer GPUs / MMIO errors until settings restored |
| 256 GB MMIO window for 4× V620 | **Community** | `#motherboard` — 64 GB too small |
| GpuMMIOFix OS BAR remap | **Needs verify** | `#motherboard` Sep 15–16 — typically every boot |
| HP Z4 G4 32-bit MIMO + ReBAR + `pci=realloc=off` | **Needs verify** | `#motherboard` Sep 16 — Xeon W-2133; Linux reallocates 64 GB VRAM |

---

## Checklist before production

1. `docker pull` latest `-extras` image; confirm fork commit in build metadata.
2. Grep logs for `Using RDNA2W4A16LinearKernel`.
3. Try CUDA graphs before `--enforce-eager`.
4. One matched A/B on your hardware.
5. Update this page when you confirm or refute a claim.

See [Contributing](../meta/contributing.md) and [Wiki structure](../meta/structure.md).
