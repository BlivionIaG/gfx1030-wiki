# llama.cpp on gfx1030

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a fast, low-dependency way to run GGUF LLMs on
gfx1030. This page is a build-from-source recipe validated by the gfx1030 community (the `#llama.cpp`
channel), on **Fedora** with **ROCm 7.2.0** targeting `gfx1030`. A **Vulkan** path is included as an
alternative that doesn't require ROCm.

> Commands are shown as used on Fedora. Adjust package names for your distro and tweak versions/paths as
> needed.

> Looking for heavy multi-GPU tuning? See the [RDNA2-optimized fork](./llama_cpp_rdna2_fork.md)
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

See [Installing ROCm](./installing_rocm.md) for more detail and for non-Fedora distros.

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
`gfx1031`/`gfx1032`) or add it to the list; see [HSA_OVERRIDE for RDNA2 Cousins](./hsa_override.md).

## Vulkan (alternative)

The Vulkan backend works without ROCm and runs across many GPUs/drivers.

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
  [Multi-GPU PCIe P2P](./tuning_p2p.md)).
- `--spec-type draft-mtp --spec-draft-n-max N` — Multi-Token-Prediction speculative decoding; the Vulkan
  example above uses a larger `N` (6) than the ROCm one (2).
- `--no-mmap`, `-dio` — memory/IO tuning; `-np 1` sets the number of parallel sequences.

Adjust the model, quant, device list, and speculative-decoding settings for your hardware.
