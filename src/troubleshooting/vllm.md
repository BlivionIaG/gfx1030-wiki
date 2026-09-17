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

Mount cache volumes — see [Configuration](../../vllm/configuration.md#cache-volumes-first-boot-is-slow).

**Fallback:** `--enforce-eager`

On multi-GPU AOT cache issues: `VLLM_USE_AOT_COMPILE=0 VLLM_DISABLE_COMPILE_CACHE=1`

Low throughput (~4–5 t/s on 27B)? Check for **older image** where AWQ still used Triton — see
[Quantization](../../vllm/quantization.md).

## First boot is extremely slow

Triton and torch.compile JIT on first run. Typical cache: `~/.triton/cache` (~3 GB),
`~/.cache/vllm/torch_compile_cache` (~700 MB).

## GPTQ/AWQ not using RDNA2 kernels

Check logs for `Using RDNA2W4A16LinearKernel`. If you see Triton/Exllama instead:

```bash
export VLLM_DISABLED_KERNELS=ExllamaLinearKernel,TritonW4A16LinearKernel
```

Confirm `-extras` image from current extras. See [Fork kernel dispatch](../../vllm/fork.md#kernel-dispatch-on-gfx1030).

## vLLM picks the wrong platform / doesn't see my Radeon

Use published [`blivioniag/vllm-rdna`](../../vllm/images.md) images with `patches/*rocm-platform*` fixes rather
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
[Configuration](../../vllm/configuration.md).

`FA_RDNA2` / `RDNA_ATTN` may not appear in the backend list on older `-extras` images or some GPTQ
models (logs only show Triton / ROCM / TurboQuant). Pull the latest `-extras` tag and confirm
`Using RDNA2W4A16LinearKernel` / native FA in startup logs. Qwen3.8-27B AWQ needs **head size 256**
on the fork — see [Quantization](../../vllm/quantization.md#int4-on-gfx1030-no-native-int4-alus).

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
   mounted ([Configuration](../../vllm/configuration.md#cache-volumes-first-boot-is-slow)).
2. Stay on **ROCm 7.2.0 or 7.14.x** — avoid mid-7.2.x (same pin as
   [multi-GPU RCCL](#multi-gpu-rccl-hangs-or-cards-drop-offline)).
3. Prefer HIP / `RDNA_ATTN` paths over AMD Triton FA where the fork offers them — less Triton means
   fewer of these stalls.
4. Next cold start with a warm cache should be much shorter; if it never recovers after hours, A/B
   `VLLM_USE_V2_MODEL_RUNNER=0` and the [long-prompt](#flash-next-long-prompt-stalls) notes.

## Multi-GPU RCCL hangs or cards drop offline

If TP works on one image and dies after a host ROCm bump, check the **ROCm version** before the
model. **7.2.1 through ~7.13** are reported to have a multi-card RCCL bug. Stay on **7.2.0** or
**7.14.0** — see [Installing ROCm](../../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

## Flash-Next long-prompt stalls {#flash-next-long-prompt-stalls}

Symptom (Flash-Next fork / recipe containers, `#vllm-rdna` Sep 2026): short prompts decode fine, but
**large prompts** (tens of k tokens — agentic coding, session resume) take many minutes, timeout, or
appear wedged. Temps and power caps look healthy. Sep 15 community add-on: **intermittent** TTFT
from a few seconds to **minutes** (one host: up to ~300 s on a tiny prompt), with one or more GPUs
at **100% util but ~40 W** — not a PCIe drop; looks like an RCCL / runner stall.

**Community fix that unblocked 4× V620 hosts:**

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

One host then saw stable **~68 tok/s** with dense INT8 + custom all-reduce; another (leapdragon
container, TP4) reported the stall **gone** after the same switch. `#vllm-rdna` Sep 15: **MoE on
the 0.28 Flash-Next line wants V1** until upstream **0.29** (V2 becomes the default). The Flash-Next
docs already call `V2=0` out. Official-extras authors note separate Dense-on-V2 fixes on the org
rebase — A/B both values on your image.

Also rule out thermal / power first ([Power tuning](../../tuning/power.md)), and measure expected
prefill time (~1k tok/s class ⇒ ~40 s for 40k tokens, not minutes). Prefill campaign numbers:
[vLLM overview](../../vllm/overview.md#qwen38-flash-next-on-vllm).

## Flash-Next 128k prefill cliff {#flash-next-128k-prefill-cliff}

Symptom (`#vllm-rdna` Sep 2026, Intel AutoRound / draft
[`opengfx1030/vllm-rdna#5`](https://github.com/opengfx1030/vllm-rdna/pull/5)): prefill holds
**~950 tok/s** through 64k, then falls to **~375 tok/s** at **128k** while decode stays flat
(~48–55 tok/s). Looks like a scheduler / chunking misconfig more than a kernel cliff.

**Community fix on that host:** raise the batch cap:

```bash
--max-num-batched-tokens 4096
```

The suspected combination was `enable_chunked_prefill=True` with **2048** scheduled / batched
tokens. After 4096, 128k prose/code PP returned to the **~950 tok/s** class (same 4× V620, FP16
KV, CPU PLE offload). See [overview](../../vllm/overview.md#intel-autoround-flash-next).

If output is **wrong** rather than just slow, stay on **2048** and switch graphs to **PIECEWISE**
first — [FULL-graph corruption](#flash-next-full-graph-corruption).

If 128k is still slow after 4096, A/B [V2 runner](#flash-next-long-prompt-stalls) and confirm you
are not on quantized PLE (known-good is the embedded BF16 n-gram table).

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
  [disable vs PIX custom AR](../../vllm/configuration.md#custom-all-reduce--p2p-two-community-stacks)
  stacks.

## Flash-Next PP3 output corruption on `rdna_extras` {#flash-next-pp3-output-corruption}

`#vllm-rdna` (Sep 15): **Qwen Next / Flash-Next** on **3× V620**, `PP=3` / `TP=1`, TheRock **10.0**,
`--enforce-eager`. A **leapdragon** tree (with the same local PP patches) decoded correctly. The same
prompt on **`rdna_extras` HEAD** (community pin `f663686`) **corrupted after ~3 chunks** — first
request of a fresh server, ~4k-token prompt, only a fraction of tokens correct.

Disabling `VLLM_USE_V2_MODEL_RUNNER`, `VLLM_USE_RDNA2_FA`, `VLLM_GDN_HIP_PREFILL`, prefix cache, and
forcing single-chunk prefill (`--max-num-batched-tokens 8192`) **did not** fix it. Suspected area:
HIP GDN decode / prefix-cache interaction — **not confirmed**.

`#vllm-rdna` (Sep 16): a **fresh `rdna_extras` pull still corrupted** on PP3. One host still decoded
on a **leapdragon** tree. Community suspected
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (`_forward_core_decode_non_spec` /
`gdn_decode_rdna2`) — **Needs verify**. `--max-num-seqs 4` (and later **6** with CUDA-graph capture
sizes `[1,2,4,8]`) reduced some TP corruption while debugging; it is **not** a PP3 fix.

**What to do:** if you need 3-card Flash-Next today, stay on a **known-good leapdragon image**. Do
not treat current org HEAD as a drop-in for PP3. Report a matched A/B (same prompt, both trees) on
`#vllm-rdna`.

## Flash-Next FULL-graph decode corruption {#flash-next-full-graph-corruption}

Symptom (`#vllm-rdna` Sep 16–17, `rdna_extras` Flash-Next, **TP**): serve starts, then output
**corrupts** after a few chunks or after turning features back on. Prefix cache + graphs can look
healthy, then fail again. Community isolated **FULL** CUDA graphs (decode) as the bad path;
**`PIECEWISE` held**.

**Community workarounds that unblocked a 4× V620 host:**

```bash
--compilation-config '{"cudagraph_mode":"PIECEWISE","compile_ranges_endpoints":[]}'
--max-num-seqs 6
--max-num-batched-tokens 2048
```

Also used while hunting: `HSA_NO_SCRATCH_RECLAIM=1`, `--max-num-seqs 4` (stricter), and leaving
CUDA-graph capture sizes **unpinned** so vLLM chose `[1,2,4,8]`. A 16k/1k × 8 concurrency
snapshot was called a **stable base** before performance claw-back.

This does **not** replace [PP3 corruption](#flash-next-pp3-output-corruption). Hub `-extras` **27B**
can still use `FULL_AND_PIECEWISE` — see [Configuration](../../vllm/configuration.md#cuda-graphs-preferred-over---enforce-eager).
Sanitized serve line: [Flash-Next PIECEWISE recipe](../../vllm/recipes.md#flash-next-4x-piecewise).

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
yet. See [fork landscape](../../vllm/fork.md#consolidation-status).

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
