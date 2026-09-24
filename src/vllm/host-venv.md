# Host venv (no Docker)

> **WIP:** Commands below follow
> [`opengfx1030/vllm-rdna-docker`](https://github.com/opengfx1030/vllm-rdna-docker)
> (`Dockerfile.base`, `Dockerfile.vllm`, `docker-bake.hcl`) and the
> [`opengfx1030/vllm-rdna`](https://github.com/opengfx1030/vllm-rdna) `rdna_extras` tree.
> See [Verification status](../../reference/verification.md#host-venvmd).

Day-to-day serving should stay on the prebuilt images — [Running (Docker)](../running.md).
Use this page when you want the same stack in a host Python 3.12 `uv` venv (kernel work,
no image rebuild). You need a ROCm toolchain with `hipcc`. For more than one GPU, pin
**ROCm 7.2.0 or 7.14.0** — [Installing ROCm](../../setup/installing-rocm.md#multi-gpu-pin-rocm-720-or-7140).

The image bake lists seven gfx targets. A gfx1030-only venv can set `PYTORCH_ROCM_ARCH=gfx1030`.

## ROCm 7.2.0

`#general` (1 Aug 2026) plus bake target `base-rocm720`. Wheels come from the PyTorch ROCm 7.2
index. That index does **not** publish `rocm-sdk-device-*` or `amd-torch-device-*` — do not add them.

```sh
uv venv --python 3.12 venv-7.2.0
source venv-7.2.0/bin/activate

uv pip install --pre --force-reinstall \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  "torch==2.12.0" "triton==3.5.1" torchvision torchaudio
```

The image pins `triton==3.5.1`. The same `#general` thread also had `triton-rocm==3.7.1` installed
beside it. Triton is a wheel either way — do not build it from source for this path.

Then [build FlashAttention, causal-conv1d, and vLLM](#build-into-the-venv). On 7.2.0 only, add
`--torch-backend=rocm7.2` to the vLLM `uv pip install`.

## ROCm 7.14.0

Do **not** swap the 7.2 index for `https://download.pytorch.org/whl/rocm7.14`. On Python 3.12
x86_64 that index has no matching `torchaudio` wheel (`#vllm-rdna`, 21 Sep 2026).

Bake target `base-rocm714` uses the multi-arch index and then installs the per-GPU device
packages. `triton==3.7.1` on that index is a wheel (`triton-3.7.1+git….rocm7.14.0`), not a
source build. gfx103X still has no aotriton, so PyTorch's SDPA flash path stays off; that is
separate from the Triton package.

`docker-bake.hcl` currently pins torch `2.12.0+rocm7.14.0`. The repo README and the
[image matrix](../running.md#image-matrix) still describe the published `7.14.0` tag as PyTorch
2.13.0. Recreate the bake with the pins below; if you are matching an image you already pulled,
use whatever `torch.__version__` that container prints (2.13 pairs with `torchvision==0.28.0+rocm7.14.0`
and `amd-torch-device-gfx1030==2.13.0+rocm7.14.0` on the same index).

```sh
uv venv --python 3.12 venv-7.14.0
source venv-7.14.0/bin/activate

uv pip install --pre --force-reinstall \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "torch==2.12.0+rocm7.14.0" \
  "torchvision==0.27.0+rocm7.14.0" \
  "torchaudio==2.11.0+rocm7.14.0" \
  "triton==3.7.1"

uv pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "rocm-sdk-device-gfx1030==7.14.0" \
  "amd-torch-device-gfx1030==2.12.0+rocm7.14.0"
```

### What `[device-gfx1030]` is

On a **multi-arch index** package, `torch[device-gfx1030]` is only a shortcut for the two wheels
above (`amd-torch-device-gfx1030` and `rocm-sdk-device-gfx1030`). The Docker file installs those
packages by name. It does not pass the extra.

Without them, an index torch looks for code objects under the venv's `_rocm_sdk_*` tree and
kernel launch fails with `hipErrorInvalidImage` (`Dockerfile.base`). `/opt/rocm/lib/rocblas/library`
is not the tree that wheel consults.

### Wheels you built yourself

[`leapdragon/vllm-rdna2-qwen` `tools/rdna2/build-torch-rocm714.sh`](https://github.com/leapdragon/vllm-rdna2-qwen/blob/rdna2/qwen38-flash-next/tools/rdna2/build-torch-rocm714.sh)
compiles PyTorch, Triton, and torchvision with `PYTORCH_ROCM_ARCH=gfx1030`. Those `.whl` files
already are gfx1030 builds. They have **no** `device-gfx1030` extra. Passing it makes `uv` warn
and does nothing — for example `torch==2.12.0+git6bbd260` and `torchvision==0.27.1+…` from
`~/wheels/rdna2/`.

Install the files. Do not add `amd-torch-device-*` on top:

```sh
uv pip install ~/wheels/rdna2/torch-*.whl \
  ~/wheels/rdna2/triton-*.whl \
  ~/wheels/rdna2/torchvision-*.whl
```

### Library path (index / rocm-sdk venv)

The 7.14 image writes these into `ldconfig`. A host venv needs them in the environment before
`vllm serve` (`#vllm-rdna` serve blocks). Skip this if you installed a from-source gfx1030 wheel
and have no `_rocm_sdk_*` directories.

```sh
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export LD_LIBRARY_PATH="$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_core/lib/host-math/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_core/lib/core/lib:$SP/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export HIP_PATH="$ROCM_HOME"
```

Point `ROCM_HOME` at the tree that contains `bin/hipcc`. The image base is
`rocm/dev-ubuntu-22.04:7.14.0-full` (`/opt/rocm`). A TheRock layout is often
`/opt/rocm/core-7.14`.

Do **not** pass `--torch-backend=rocm7.14` when installing vLLM. That flag selects the PyTorch
`rocm7.14` index, which is the one missing the 3.12 x86_64 `torchaudio` wheel.

## Build into the venv

Same three steps on 7.2.0 and 7.14.0, after torch is importable. `Dockerfile.vllm` is the
source for the flags and the causal-conv1d pin. The 1 Aug `#general` paste left the
FlashAttention clone as a placeholder and did not pin causal-conv1d; the image build fills both
in as below.

### FlashAttention

gfx1030 is not on the Dao-AILab allowlist
(`native`, `gfx90a`, `gfx942`, `gfx950`, `gfx1100`–`gfx1201`). Leave it out of `GPU_ARCHS`.
gfx1030 uses Triton FA at runtime (`FLASH_ATTENTION_TRITON_AMD_ENABLE`).

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init
export ENABLE_CK=0
export PREBUILD_KERNELS=0
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
GPU_ARCHS="gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201" \
  uv pip install --no-build-isolation .
cd ..
```

### causal-conv1d

`HIP_ARCHITECTURES` is the variable this commit reads (not `GPU_ARCHS`). The image pins
`4f6ae4e26ae5fe8af9372f8d312ab25cc4595223`.

```sh
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
HIP_ARCHITECTURES=gfx1030 \
ROCM_PATH="${ROCM_HOME:-/opt/rocm}" \
  uv pip install --no-build-isolation \
  "git+https://github.com/Dao-AILab/causal-conv1d.git@4f6ae4e26ae5fe8af9372f8d312ab25cc4595223"
```

### vLLM (`rdna_extras`)

`uv pip install -e . --no-build-isolation` is what `Dockerfile.vllm` runs. It does not install
`requirements/rocm.txt` or `requirements/dev.txt` (`dev.txt` pulls the CUDA test set).

On **7.2.0**, append `--torch-backend=rocm7.2`. On **7.14.0**, leave that flag off.

```sh
git clone -b rdna_extras https://github.com/opengfx1030/vllm-rdna.git
cd vllm-rdna
VLLM_USE_PRECOMPILED=0 \
ENABLE_CK=0 \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
ROCM_HOME="${ROCM_HOME:-/opt/rocm}" \
PYTORCH_ROCM_ARCH=gfx1030 \
  uv pip install -e . --no-build-isolation
```

Smoke test, from [Getting started](../../setup/getting-started.md#5-smoke-test):

```sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import vllm, vllm._rocm_C; print('vllm', vllm.__version__)"
```

## Related

- [Running (Docker)](../running.md) — prefer this unless you need a host venv.
- [Building images](../images.md) — the bake graph this page mirrors.
- [vLLM forks](../fork.md) — kernel behavior after the install.
