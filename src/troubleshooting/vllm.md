# vLLM troubleshooting

## CUDA graph capture crashes {#cuda-graph-capture-crashes}

Symptom: crash at `GDN _output_projection all-reduce` / `_SimpleCData.__new__`, or OOM during graph capture.

**Root cause (fixed in current extras):** TP comm wrappers were not `allow_in_graph`. Fork fix:
`fix(distributed): allow TP comm ops in torch.compile graph capture`.

**First, pull the latest image:**

```bash
docker pull blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0
```

Then try CUDA graphs (fast path):

```bash
--compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "compile_ranges_endpoints": []}'
```

Mount cache volumes — see [Configuration](../vllm/configuration.md#cache-volumes-first-boot-is-slow).

**Fallback:** `--enforce-eager`

On multi-GPU AOT cache issues: `VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1`

Low throughput (~4–5 t/s on 27B)? Check for **older image** where AWQ still used Triton — see
[Quantization](../vllm/quantization.md).

## First boot is extremely slow

Triton and torch.compile JIT on first run. Typical cache: `~/.triton/cache` (~3 GB),
`~/.cache/vllm/torch_compile_cache` (~700 MB).

## GPTQ/AWQ not using RDNA2 kernels

Check logs for `Using RDNA2W4A16LinearKernel`. If you see Triton/Exllama instead:

```bash
export VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel
```

Confirm `-extras` image from current extras. See [Fork kernel dispatch](../vllm/fork.md#kernel-dispatch-on-gfx1030).

## TunableOp aborts or is ~20% slow after a ROCm / rocBLAS bump {#tunableop-rocblas-mismatch}

`#vllm-rdna` (Sep 23–24) + in-tree
[`tunableop/`](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/tunableop): the committed
FP16 rows are **locked to one rocBLAS library hash** (qualified example: `rocblas-c27e2252cc7a`).
Solution IDs must **not** be reused across a different rocBLAS build, even when the version string
matches.

Symptoms:

- First GEMM **aborts** / TunableOp turns itself off.
- Prefill sits **~25% below** the `#17` published table (community: **~1496 vs ~1958** tok/s at 16k)
  until a matching table is loaded. **ROCm 10 vs 7.14** was **not** the gap on that host.

Fix: stop the serve, then regenerate and qualify for **this** library — see
[Regenerate for another rocBLAS build](https://github.com/opengfx1030/vllm-rdna/tree/rdna_extras/tunableop#regenerate-for-another-rocblas-build).
Serve with `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_TUNING=0`, and
`PYTORCH_TUNABLEOP_FILENAME` pointing at `tunableop/rocblas-<your-hash>/…`. The `#17` launcher
refuses incompatible solver IDs rather than loading them.

Do not commit a host-specific CSV into the wiki. Do not copy another machine’s library hash.

## vLLM picks the wrong platform / doesn't see my Radeon

Use published [`blivioniag/vllm-rdna`](../vllm/images.md) images with `patches/*rocm-platform*` fixes rather
than stock upstream builds.

## `Failed to infer device type` / `AMDSMI_STATUS_NOT_INIT`

Symptom: vLLM logs `ROCm platform is not available because no GPU is found` and
`AMDSMI_STATUS_NOT_INIT - Device not initialized` (often right after CUDA/NVML is also missing — that
part is expected on AMD).

Check the Docker device/group block **exactly** — `#vllm-rdna` hits this when `render` is missing:

```yaml
devices: [/dev/kfd, /dev/dri]
group_add: [video, render]    # both — video alone is not enough on many hosts
ipc: host
security_opt: [label=disable]  # or seccomp=unconfined on docker run
```

Host user still needs `render`/`video` as in [General troubleshooting](./general.md). Debug with
`VLLM_LOGGING_LEVEL=DEBUG`. If the same compose worked on an older tag, `docker pull` a known-good
image — a bad rebuild can also fail AMDSMI init.

## `ROCM_ATTN` hangs for hours (Triton compile)

AMD Triton flash-attention compile on gfx1030 can sit there for **hours** (RCCL vs Triton). On `-extras`,
use `--attention-backend RDNA_ATTN` and/or `VLLM_USE_RDNA2_FA=1` instead of `ROCM_ATTN`. See
[Configuration](../vllm/configuration.md).

`FA_RDNA2` / `RDNA_ATTN` may not appear in the backend list on older `-extras` images or some GPTQ
models (logs only show Triton / ROCM / TurboQuant). Pull the latest `-extras` tag and confirm
`Using RDNA2W4A16LinearKernel` / native FA in startup logs. Qwen3.8-27B AWQ needs **head size 256**
on the fork — see [Quantization](../vllm/quantization.md#int4-on-gfx1030-no-native-int4-alus).

### `shm_broadcast` / one GPU + one CPU pegged {#shm-broadcast-triton-vs-rccl}

Symptom (`#vllm-rdna` Sep 2026, often Flash-Next / multi-GPU): serve looks wedged; logs repeat
something like:

```text
No available shared memory broadcast block found in 60 seconds.
This typically happens when some processes are hanging or doing some time-consuming work
(e.g. compilation, weight/kv cache quantization).
```

Community diagnosis: classic **Triton JIT compile fighting RCCL** — not necessarily a dead process.
Guidance:

1. **Wait** — first boots can sit like this a long time; keep the Triton / compile cache volumes
   mounted ([Configuration](../vllm/configuration.md#cache-volumes-first-boot-is-slow)).
2. Stay on **ROCm 7.2.0 or 7.14.x** — avoid mid-7.2.x (same pin as
   [multi-GPU RCCL](#multi-gpu-rccl-hangs-or-cards-drop-offline)).
3. Prefer HIP / `RDNA_ATTN` paths over AMD Triton FA where the fork offers them — less Triton means
   fewer of these stalls.
4. Next cold start with a warm cache should be much shorter; if it never recovers after hours, A/B
   `VLLM_USE_V2_MODEL_RUNNER=0` and the [long-prompt](./vllm-flash-next.md#flash-next-long-prompt-stalls) notes.

## Multi-GPU RCCL hangs or cards drop offline

If TP works on one image and dies after a host ROCm bump, check the **ROCm version** before the
model. **7.2.1 through ~7.13** are reported to have a multi-card RCCL bug. Stay on **7.2.0** or
**7.14.0** — see [Installing ROCm](../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

## Prefill blocks decode / MTP stalls under concurrency

Symptom: with speculative decode (MTP) and multiple in-flight requests, generation stalls while
prefill runs; or graph + MTP3 reaches "Application startup complete" then hangs on PLE lookup /
`sample_tokens` timeout.

Community notes (`#vllm-rdna`):

- Prefer **GPTQ + RDNA2 W4A16** (or AWQ HIP) paths over GGUF-in-vLLM for these cards.
  `#vllm-rdna` (Sep 15): `RDNA2W4A16MoEExperts` was reported to raise **usable concurrency** vs the
  older MoE path — confirm the kernel name in logs; A/B if your image is older.
- **One prefill stream can fully block other decode streams** (community Sep 15). When you bench,
  alternate prefill+decode in flight — concurrency-only or decode-only numbers miss this.
- Concurrent MTP / prefill-vs-decode fixes land in community recipes first — see open PRs on
  [`leapdragon/vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe).
- MTP=0 vs MTP=3 are different bug surfaces; a commit that "works" at MTP=3 can still emit spurious
  tokens at MTP=0. A/B and pin a known-good recipe commit.
- Slow or broken P2P + custom all-reduce can look like MTP latency bugs — A/B the
  [disable vs PIX custom AR](../vllm/configuration.md#custom-all-reduce--p2p-two-community-stacks)
  stacks.

## Upstream KV offload tanks decode {#upstream-kv-offload-tanks-decode}

`#vllm-rdna` (Sep 15): **upstream vLLM** CPU → SSD **KV cache offload** (not LMCache) failed to
bring blocks back to VRAM usefully. Community: decode fell to the **~1 t/s** class. **Mamba / SSM**
state models are called out as especially unreliable on this path (including on Hopper-class hosts
in-channel).

`#vllm-rdna` (Sep 16): public RAM-offload Flash-Next packs (example:
[`Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`](https://huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound))
are interesting for long context, but recipes that need **`--no-enable-prefix-caching`** are a
**non-starter** on gfx1030 vLLM. Qwen4exp KV is already relatively efficient; parking **non-linear**
state in host RAM fights prefix cache. **LMCache** is still the intended overflow path.

Do not plan production multi-chat overflow on native vLLM KV offload. `#lmcache` is still the
intended gfx1030 path (standalone LMCache server + vLLM connector) but has **no published recipe**
yet. See [fork landscape](../vllm/fork.md#consolidation-status).

`#vllm-rdna` (Sep 20): upstream [`vllm#57160`](https://github.com/vllm-project/vllm/pull/57160)
(mainline ROCm, **not** v0.29 and **not** confirmed on `rdna_extras`) changes CPU KV offload to
private pinned tensors after `cudaHostRegister` failures on large TP. Treat as a tracking note —
do not assume it restores usable decode on gfx1030.

`#vllm-rdna` / `#general` (Sep 18–19): official **`rocm/pytorch`** images are the **wrong** place to
compile an LMCache connector (missing HIP / developer packages). Intended shape:

1. Run [`lmcache/standalone`](https://hub.docker.com/r/lmcache/standalone) in **CPU** mode — see the
   [standalone starter](https://docs.lmcache.ai/getting_started/quickstart/standalone_starter.html).
2. Add the LMCache connector to a **bare-metal / venv** `rdna_extras` serve, or patch it onto
   [`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) / [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna).
3. Do **not** use `lmcache/vllm-openai` on gfx1030 — that image is the CUDA path.

NVMe-as-KV is the usual motive (low host RAM). Until someone posts a working gfx1030 compose, treat
this as **Needs verify**.

`#vllm-rdna` (Sep 22) + `#lmcache` (Sep 23–24): still **no working gfx1030 compose**. Extra facts:

- Do **not** remake standalone — use [`lmcache/standalone`](https://hub.docker.com/r/lmcache/standalone).
- A community **0.5.6.dev** wheel built against gfx1030; that is the **library**, not a proven
  connector. Official LMCache kernels are **CDNA-only**; generic fallback still has **Mamba align**
  problems (last known real blocker on `0.5.4`).
- Hybrid Qwen (MambaSpec + QSA FullAttentionSpec) creates **multiple KV groups** and needs **HMA**.
  `LMCacheConnectorV1` is **not HMA-capable** (`SupportsHMA` missing; vendored copy still asserts
  a single group) → startup `ValueError: Failed to promote local KV cache specs to one unified type`.
- A community LDS **~143 KB vs 64 KB** claim is still **Needs verify** (LLM-sourced, no compile log).

Keep the standalone-CPU + connector shape; expect **fork patches** before this is a recipe.

## Leftover vLLM / PLE workers after a restart {#leftover-vllm-workers}

`#vllm-rdna` (Sep 18): killing the OpenAI API process is **not** enough. `VLLM::Worker`,
`VLLM::EngineCore`, and `PleOffloadWorker` can stay up and hold VRAM / the PLE sidecar. Before
relaunch, stop those leftover processes (match the **process names**, not a broad pattern) and wait
a few seconds. The [4× Flash-Next recipe](../vllm/flash-next-serve.md#flash-next-4x-piecewise) already
`cd`s to `/tmp` so GPU coredumps do not land in a checkout.

## ROCR idle CPU spin (TheRock 7.14) {#rocr-idle-cpu-spin-therock-714}

Symptom: after starting a multi-GPU vLLM serve on **TheRock / ROCm 7.14** (ROCR **1.21**), the host
holds several CPU cores at high utilization even when the GPUs are idle (~one core per HIP process,
plus more once RCCL initializes).

Root cause: `AsyncEventsLoop` / signal-wait paths busy-spin without backoff
([ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051),
[ROCm/ROCm#6522](https://github.com/ROCm/ROCm/issues/6522)). Stock env knobs
(`HSA_ENABLE_INTERRUPT`, etc.) do not fix multi-GPU cases.

**Fix:** rebuild only `libhsa-runtime64.so` with the poll-backoff patch and `LD_PRELOAD` it. Step-by-step
for host builds (and note that recipe containers already bake the patch):
[`docs/rdna2/ROCR-CPU-FIX.md`](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/docs/rdna2/ROCR-CPU-FIX.md)
in [`leapdragon/vllm-rdna2-qwen`](https://github.com/leapdragon/vllm-rdna2-qwen/tree/rdna2/qwen38-flash-next).
Pull latest recipe / container before re-debugging idle CPU.

Flash-Next stalls, graph corruption, KV accounting, PP3, and PLE issues are on
[Flash-Next troubleshooting](./vllm-flash-next.md).

