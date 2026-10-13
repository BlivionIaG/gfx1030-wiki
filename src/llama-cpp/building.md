# llama.cpp on gfx1030

> **WIP:** Tips marked community-validated in this page come from `#llamacpp` — see
> [Verification status](../reference/verification.md#llama-cpp).

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a fast, low-dependency way to run GGUF LLMs on
gfx1030. This page has two stock HIP recipes from `#llamacpp`: **Fedora + ROCm 7.2.0**, and
**Ubuntu + ROCm 7.14** (`/opt/rocm/core-7.14`). A **Vulkan** path is included as an alternative that
doesn't require ROCm.

> Adjust package names and ROCm prefixes for your distro. The Fedora block uses `/opt/rocm` +
> `hipconfig`; the Ubuntu block uses TheRock-style `/opt/rocm/core-7.14`.

> Looking for heavy multi-GPU tuning? See the [RDNA2-optimized fork](./rdna2-overview.md)
> (`edwinbrowwn/llama.cpp-rdna2`) with RDNA2/V620 tensor-parallel and MMQ optimizations.

## Dependencies (Fedora)

```bash
sudo dnf install @development-tools glm-devel cmake libpng-devel wayland-devel libpciaccess-devel \
  libX11-devel libXpresent libxcb xcb-util libxcb-devel libXrandr-devel xcb-util-keysyms-devel \
  xcb-util-wm-devel python3 git lz4-devel libzstd-devel python3-distutils-extra qt gcc-g++ \
  wayland-protocols-devel ninja-build python3-jsonschema qt5-qtbase-devel qt6-qtbase-devel \
  libcurl-devel xinput libXinerama xcb-util-cursor
```

## ROCm (recommended)

### Install ROCm

Example: Fedora with **ROCm 7.2.0**. Create `/etc/yum.repos.d/rocm.repo`:

```ini
[rocm720]
name=ROCm 7.2.0 repository
baseurl=https://repo.radeon.com/rocm/el10/7.2/main
enabled=1
gpgcheck=1
priority=50
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
```

Then install ROCm and add yourself to the GPU access groups:

```bash
sudo dnf clean all
sudo dnf makecache
sudo rpm --import https://repo.radeon.com/rocm/rocm.gpg.key
sudo dnf install rocm rocm-hip-runtime-devel
sudo usermod -a -G render,video $LOGNAME
# log out / back in (or reboot) so the group change takes effect
```

See [Installing ROCm](../setup/installing-rocm.md) for more detail and for non-Fedora distros.

### Build llama.cpp with ROCm (HIP)

```bash
git clone https://github.com/ggml-org/llama.cpp.git
export MAX_JOBS=8            # adjust to your CPU cores / available RAM
export ROCM_HOME=/opt/rocm
export PATH=${ROCM_HOME}/bin:${PATH}

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S llama.cpp -B build \
    -DLLAMA_CURL=ON -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1030 && \
  cmake --build build --config Release -- -j ${MAX_JOBS}
```

`-DGPU_TARGETS=gfx1030` targets Navi 21. For a non-Navi-21 RDNA2 card, build for its real target (e.g.
`gfx1031`/`gfx1032`) or add it to the list; see [HSA_OVERRIDE for RDNA2 Cousins](../setup/hsa-override.md).

### Ubuntu + ROCm 7.14 (community, Oct 2026) {#ubuntu-rocm-714-stock-hip}

`#llamacpp` (13 Oct) and `#general`: stock `ggml-org/llama.cpp` on **Ubuntu** with **ROCm 7.14**.
Community: llama.cpp's default HIP cmake flags are **not enough** on gfx1030 — this extra set was
reported **~3–6%** faster than a default mainline build on **2× V620**. The
[RDNA2 fork](./rdna2-overview.md) is still faster than even this tuned stock build.

Community host class: **Dell Precision 5820**, Xeon W-2155, 128 GB quad-channel, each GPU
**PCIe 3.0 x16**, Proxmox passthrough into a dedicated Ubuntu VM.

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_PATH=/opt/rocm/core-7.14
export HIPCXX=/opt/rocm/core-7.14/lib/llvm/bin/clang
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-7.14/lib/llvm/bin/clang \
  -DCMAKE_HIP_PLATFORM=amd \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-7.14 \
  -DAMDGPU_TARGETS=gfx1030 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DGGML_HIP=ON \
  -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DGGML_NATIVE=ON \
  -DGGML_AVX2=OFF \
  -DGGML_AVX512=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build --config Release -- -j$(nproc)
```

Serve with the same [short env stack](./rdna2-serving.md#recommended-env-stack) the fork uses (this
report applied it to stock HIP):

```sh
HSA_OVERRIDE_GFX_VERSION=10.3.0 \
HSA_NO_SCRATCH_RECLAIM=1 \
GGML_HIP_RDNA2_AUTO=1 \
GGML_HIP_SAFE_STATE_IO=1 \
GGML_TP_SHARDED_OUTPUT=1 \
HIP_VISIBLE_DEVICES=0,1 \
./build/bin/llama-server \
  -m /path/to/model.gguf
```

`GGML_TP_SHARDED_OUTPUT=1` is TP2+ only. `GGML_AVX2=OFF` / `GGML_AVX512=OFF` were part of this
Skylake-W recipe — A/B if your CPU is newer. If your 7.14 install is not under
`/opt/rocm/core-7.14`, see [Installing ROCm](../setup/installing-rocm.md).

## Vulkan (alternative)

The Vulkan backend works without ROCm and runs across many GPUs/drivers. For **multi-GPU tensor
split**, prefer the [ROCm build](#rocm-recommended) (or the [RDNA2 fork](./rdna2-overview.md)) —
`#llamacpp` finds HIP/`-sm tensor` faster, and RCCL TP is a ROCm path.

Mesa **RADV** has been reported to **hard-crash** V620 llama.cpp; **AMDVLK** can stay up but is
severely slower. If you must use Vulkan, pin the ICD explicitly:

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json   # AMDVLK, not RADV
export GGML_VULKAN_DEVICE=0
```

### Vulkan SDK from your distro

```bash
sudo dnf install mesa-vulkan-drivers vulkan-devel glslc spirv-headers-devel
```

### Vulkan SDK from source

Example with version `1.4.350.1`, assuming you keep things in `~/Apps/llama.cpp`:

```bash
export VULKAN_VERSION=1.4.350.1
wget https://sdk.lunarg.com/sdk/download/${VULKAN_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_VERSION}.tar.xz
mkdir vulkan
cd vulkan
tar xf ../vulkansdk-linux-x86_64-${VULKAN_VERSION}.tar.xz
export VULKAN_SDK=~/Apps/llama.cpp/vulkan/${VULKAN_VERSION}/x86_64
export PATH=${VULKAN_SDK}/bin:${PATH}
export LD_LIBRARY_PATH=$VULKAN_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VK_LAYER_PATH=${VULKAN_SDK}/share/vulkan/explicit_layer.d
export VK_ADD_LAYER_PATH=${VULKAN_SDK}/share/vulkan/explicit_layer.d
export PKG_CONFIG_PATH=$VULKAN_SDK/share/pkgconfig:$VULKAN_SDK/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
export CMAKE_PREFIX_PATH=${VULKAN_SDK}:${VULKAN_SDK}/lib/VulkanLoader
```

### Build with Vulkan

```bash
git clone https://github.com/ggml-org/llama.cpp.git
export MAX_JOBS=8            # adjust to your CPU cores / available RAM

cmake -S llama.cpp -B build \
  -DLLAMA_CURL=ON -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build --config Release -- -j ${MAX_JOBS}
```

## Usage

Multi-GPU `llama-server` examples with speculative decoding (MTP) and tensor-split across four cards.

ROCm:

```bash
llama-server -hf unsloth/Qwen3.5-122B-A10B-MTP-GGUF:UD-Q4_K_XL \
  --no-mmap -dio -fa on -ngl 999 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --device ROCm0,ROCm1,ROCm2,ROCm3 --split-mode tensor --host 0.0.0.0
```

Vulkan:

```bash
llama-server -hf unsloth/Qwen3.5-122B-A10B-MTP-GGUF:UD-Q4_K_XL \
  --no-mmap -dio -fa on -ngl 999 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 6 \
  --device Vulkan0,Vulkan1,Vulkan2,Vulkan3 --split-mode tensor --host 0.0.0.0
```

Flag notes (tune to your setup):

- `-ngl 999` — offload all layers to the GPU(s).
- `-fa on` — flash attention.
- `--device ROCm0,ROCm1,…` / `Vulkan0,Vulkan1,…` — select the backend devices to use.
- `--split-mode tensor` — split each tensor across the selected GPUs (needs good inter-GPU bandwidth; see
  [Multi-GPU PCIe P2P](../tuning/p2p.md)).
- `--spec-type draft-mtp --spec-draft-n-max N` — Multi-Token-Prediction speculative decoding; the Vulkan
  example above uses a larger `N` (6) than the ROCm one (2).
- `--spec-draft-device ROCm0` — run the MTP draft model on a single GPU while the main model is
  tensor-split across multiple cards (community-validated on TP2).
- `--no-mmap`, `-dio` — memory/IO tuning; `-np 1` sets the number of parallel sequences.

For RDNA2-tuned multi-GPU serving (DFlash2, RCCL all-reduce, higher throughput), see the
[RDNA2-optimized fork](./rdna2-overview.md) — that's where most `#llamacpp` performance work
happens.

Adjust the model, quant, device list, and speculative-decoding settings for your hardware.
