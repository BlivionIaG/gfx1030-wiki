# llama.cpp troubleshooting

## KV checkpoint crash on tensor split

Symptom: fatal error in `ggml-backend-meta.cpp` during warmup with tensor split.

Fix: `--ctx-checkpoints 0`. Known on stock llama.cpp and the RDNA2 fork. See
[RDNA2 serving limits](../../llama-cpp/rdna2-serving.md#notable-limits).

## FlashAttention abort: `max_blocks_per_sm > 0`

Symptom: server aborts with something like:

```text
fattn-common.cuh:…: GGML_ASSERT(max_blocks_per_sm > 0) failed
  launch_fattn<256, …>
```

`#llamacpp` / community reports this on **gfx1030** when the HIP occupancy query returns **0** for
the **head-size 256** FA tile kernel (common on Qwen3.8 dense and some MoE paths).

Constraints that make this painful:

| Goal | Constraint |
|---|---|
| Quantized **V** cache (`q8` KV, etc.) | Requires `--flash-attn on` — no bypass |
| `--flash-attn off` | Forces **f16** KV |
| `--split-mode tensor` | Effectively needs FA on many workloads |

Mitigations to try (in order):

1. Prefer the RDNA2 fork build script (`./scripts/build-rdna2-portable.sh`) so FA paths match the
   fork's gfx1030 profile.
2. Keep the **simplified** env stack — especially `GGML_HIP_SAFE_STATE_IO=1` (known ROCm FA crash
   workaround). Do **not** pile on every `GGML_HIP_GFX1030_*` flag; that set can clash with the RCCL
   autotune path. See [Serving](../../llama-cpp/rdna2-serving.md#recommended-env-stack).
3. If FA still aborts on head-256 models under TP: fall back to **f16 KV + FA on** only after a
   fork update / local occupancy patch, or temporarily use **layer split** for that model until FA
   occupancy is fixed upstream/fork-side.

## DAX-backed mmap oopses amdgpu SVM

Symptom: loading a GGUF from an Optane / pmem **`dax=always`** mount with default mmap → kernel fault
in `svm_range_dma_map_dev`, process becomes unkillable, VRAM leaked on all cards, reboot required.

**Fix:** `--no-mmap` and/or `--load-mode none` whenever the model file lives on a DAX mount. Storage
only affects **load time** (community: Optane DAX ~4 GB/s vs NVMe hundreds of MB/s for a ~30 GB
GGUF); once weights are in VRAM, inference is unchanged. Great for swap-testing models; useless for a
single long-lived production load.

## RCCL all-reduce fails (HIP "operation cannot be performed")

Symptom: `ggml_backend_cuda_comm_allreduce_nccl` crash, `NCCL WARN HIP failure`.

Try in order:

1. Confirm [PCIe P2P](../../tuning/p2p.md) is working.
2. Set `NCCL_P2P_LEVEL=PHB` or `NCCL_P2P_DISABLE=1`.
3. On the RDNA2 fork: `GGML_HIP_GFX1030_P2P_ALLREDUCE=off` or `GGML_CUDA_ALLREDUCE=none`.
4. Check ACS — CPU root-port ACS can block GPU-direct P2P.

## P2P enabled but slower inference

Bandwidth tests can pass while inference regresses on gen3 x4 or ACS-blocked topologies. A/B with
`NCCL_P2P_DISABLE=1`. See [When P2P hurts](../../tuning/p2p.md#when-p2p-is-enabled-but-inference-is-slower).

## PSU dies the moment tensor-split prefill starts

Symptom: layer split is stable; `--split-mode tensor` kills power (no HIP error in logs). `#llamacpp`
traced this to **PSU transients**, not the kernels — especially old miner PSUs and Lenovo P620
proprietary GPU cables (that chassis PSU often only feeds **two** cards).

1. Cap at **160 W** or **140 W** ([Power Tuning](../../tuning/power.md)).
2. A/B card pairs — one slot pair can trip protection while others do not.
3. Prefer **2 or 4** GPUs; TP3 has caused driver crashes after a "successful" run.
4. Split GPU power off the motherboard PSU if the board only has two GPU power ports.

## Dual-socket / NUMA is slow

Pin llama.cpp to one socket (`numactl --cpunodebind=0 --membind=0`) and keep all TP GPUs on that
socket. Crossing NUMA for tensor split is a known prefill killer — see
[Host topology](../../tuning/p2p.md#host-topology).

## Vulkan RADV hard-crashes; AMDVLK is slow

`#llamacpp`: Mesa **RADV** can hard-reboot or crash the host on V620 llama.cpp; switching the ICD to
**AMDVLK** (`VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json`) can get inference running but is
**much slower**. For multi-GPU **`--split-mode tensor`**, the community path is **ROCm / HIP**, not
Vulkan — tensor parallel needs RCCL. See [Building llama.cpp](../../llama-cpp/building.md#vulkan-alternative).

If a **new V620** hard-reboots a box that was stable with a 3080, read
[Slot power and PSU transients](../../tuning/power.md#slot-power-and-psu-transients) before chasing
Vulkan ICDs.
