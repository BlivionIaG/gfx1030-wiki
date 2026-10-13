# Onboard vLLM on V620

> **WIP:** Happy path below follows the `#announcements` **v0.31.0** release (11 Oct 2026) and
> day-to-day practice in `#vllm-rdna`. See [Verification status](../reference/verification.md#onboardingmd).

Get from a bare V620 host to an OpenAI-compatible `vllm serve` with the community
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) fork (`rdna_extras` = upstream
**vLLM 0.31.0** + RDNA HIP kernels). Prefer this page over the deeper vLLM docs until the first
request succeeds.

Not sure vLLM is the right stack? Read [llama.cpp or vLLM](../choose-a-stack.md) first.

## What you need

| Item | Notes |
|---|---|
| **GPU** | One or more **Radeon PRO V620** (gfx1030). Cousin cards need [`HSA_OVERRIDE`](../setup/hsa-override.md). |
| **Host** | Linux with working `amdgpu` + ROCm userspace. Multi-GPU: pin **ROCm 7.2.0 or 7.14.0** — [Installing ROCm](../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140). |
| **Groups** | Your user in `video` and `render` (reboot after `usermod`). Missing `render` → [device infer failure](../troubleshooting/vllm.md#failed-to-infer-device-type--amdsmi_status_not_init). |
| **Path** | **Docker** (Path A) or a host **Python 3.12** venv + `hipcc` (Path B). |

Optional but worth it on V620 before long serves: [power cap](../tuning/power.md),
[ECC](../tuning/ecc.md), [PCIe P2P](../tuning/p2p.md) for multi-GPU. Full host walkthrough:
[Getting started](../setup/getting-started.md).

## Pick a path

| Goal | Use |
|---|---|
| First serve today, day-to-day **27B AWQ** | **Path A** — Hub image `blivioniag/vllm-rdna:v0.31.0-extras` |
| **Flash-Next**, HEAD kernels, or the in-tree launcher | **Path B** — clone the fork + [host venv](./host-venv.md) + [`serve_rdna.sh`](./serve-scripts.md) |

`#announcements` (11 Oct 2026): Hub **`v0.31.0-extras`** is the published 0.31 bake (ROCm **7.14**,
torch **2.12**, Triton **3.7.1**). Older `v0.27.1-extras` / `v0.28.0-extras` tags remain on Hub for
comparison only.

---

## Path A — Docker Hub `-extras` (fastest)

```sh
docker pull docker.io/blivioniag/vllm-rdna:v0.31.0-extras

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

- Always `docker pull` before debugging — tags refresh in place.
- Prefer **`--dtype float16`**. RDNA2 BF16 is weak/emulated.
- Multi-GPU: add `--tensor-parallel-size N`. Prefer power-of-2 TP. Enable [P2P](../tuning/p2p.md) when
  your host supports it; if P2P is broken, set `NCCL_P2P_DISABLE=1` (and see Path B `P2P=0`).
- First boot is slow (Triton / torch.compile). Mount caches for restarts —
  [Configuration](./configuration.md#cache-volumes-first-boot-is-slow).

### Smoke test

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"cyankiwi/Qwen3.8-27B-AWQ-INT4","messages":[{"role":"user","content":"Hi from V620"}]}'
```

### Flash-Next in Docker (4× V620)

`#announcements` (11 Oct): the same **`v0.31.0-extras`** image can serve Flash-Next. You need local
weights plus the PLE table ([`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
and [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant)).
Mount the INT4 PLE directory and pass the fork env:

```sh
# Paths are examples — use your local checkpoints.
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc host \
  -p 8000:8000 \
  -v /path/to/Qwen3.8-Flash-Next-AWQ:/model \
  -v /path/to/Qwen3.8-Flash-Next-PLE-quant/ples_int4:/ples_int4 \
  -e VLLM_PLE_QUANT_DIR=/ples_int4 \
  docker.io/blivioniag/vllm-rdna:v0.31.0-extras \
  vllm serve /model --dtype float16 --tensor-parallel-size 4
```

For **disk** PLE (lower pinned RAM), also set `-e VLLM_PLE_CPU_OFFLOAD=1`. Prefer Path B’s
`RECIPE=flashnext-mtp2` once you want the full tuned argv (graphs, MTP, features). Details:
[Flash-Next](./flash-next.md), [Serve scripts](./serve-scripts.md).

---

## Path B — Fork checkout + host venv

Use this when you want HEAD of
[`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) (`rdna_extras`) or the in-tree
launcher. Build steps live on [Host venv](./host-venv.md) (ROCm **7.14.0** multi-arch wheels are the
usual match for the 0.31 Docker bake).

```sh
git clone -b rdna_extras https://github.com/opengfx1030/vllm-rdna.git
cd vllm-rdna
# Build torch + FlashAttention + causal-conv1d + editable vLLM — see host-venv.md
```

After `import vllm` works in the venv, serve from that checkout:

```sh
# Day-to-day dense 27B (local checkpoint or set HF_HUB_OFFLINE=0)
MODEL=/path/to/Qwen3.8-27B-AWQ VENV=/path/to/venv-7.14.0 \
  bash tools/rdna/serve_rdna.sh RECIPE=27b-awq

# Flash-Next MTP=2 on 4× V620 (local weights + INT4 PLE)
MODEL=/path/to/Qwen3.8-Flash-Next-AWQ VENV=/path/to/venv-7.14.0 \
  bash tools/rdna/serve_rdna.sh RECIPE=flashnext-mtp2 \
  PLE=int4 PLE_QUANT_DIR=/path/to/ples_int4 \
  HOST=0.0.0.0 PORT=8888
```

- `MODEL` and `VENV` are required. Recipes do **not** bake in home directories or IPs.
- Dry-run: `PRINT=1 … RECIPE=27b-awq` prints env + argv without launching.
- Hosts without working PCIe P2P: add `P2P=0` (`#vllm-rdna` Oct 11 — disables `rdna_ar` and sets
  `NCCL_P2P_DISABLE=1`).
- Other recipes: `flashnext`, `flashnext-mtp0`, `27b-exl3`, `dsv4-flash`, `full` —
  [Serve scripts](./serve-scripts.md).

---

## Which model for your card count

| Cards | Start here |
|---|---|
| **1×** | Hub Path A: **Qwen3.8-27B AWQ** or a MoE (e.g. Qwen3.6 **35B-A3B**). Flash-Next is **not** a 1-card path. |
| **2×** | Path A/B with `--tensor-parallel-size 2` / recipe TP=2. Flash-Next usually wants more cards or [llama.cpp](../llama-cpp/rdna2-speculative.md#flash-next-2x-iq4). |
| **4×** | Best Flash-Next layout: **TP=4**, Path B `RECIPE=flashnext-mtp2` (or Hub Path A with PLE mounts). |
| **8×** | Not the default Flash-Next recipe. Community **GLM-5.3-Flash** notes — [Recipes](./recipes.md#glm-53-flash). |

Full matrix: [Recipes](./recipes.md). Workload fit: [What fits on V620](../choose-a-stack.md#what-fits-well-on-v620).

## If something breaks

| Symptom | Where |
|---|---|
| `Failed to infer device type` / amdsmi | [Troubleshooting](../troubleshooting/vllm.md#failed-to-infer-device-type--amdsmi_status_not_init) — add `--group-add render` |
| First start takes forever | [Cache volumes](./configuration.md#cache-volumes-first-boot-is-slow) |
| TunableOp / rocBLAS mismatch | [TunableOp](../troubleshooting/vllm.md#tunableop-rocblas-mismatch) |
| Flash-Next PP3 / graphs / prefix cache | [Flash-Next troubleshooting](../troubleshooting/vllm-flash-next.md) |
| Multi-GPU RCCL weirdness on ROCm 7.2.1–7.13 | [ROCm pin](../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140) |

Ask in Discord `#vllm-rdna` — [invite](https://discord.gg/mESex2aBp).

## Next steps

- [Recipes](./recipes.md) — Hub vs recipe container vs Flash-Next by card count
- [Configuration](./configuration.md) — env vars, CUDA graphs, Compose
- [Quantization](./quantization.md) — AWQ / GPTQ / MTP / KV
- [vLLM forks](./fork.md) — kernel landscape and PR status
- [Building images](./images.md) — bake your own from [`vllm-rdna-docker`](https://github.com/opengfx1030/vllm-rdna-docker)
