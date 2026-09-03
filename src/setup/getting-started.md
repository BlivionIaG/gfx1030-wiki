# Getting Started

This page gets you from a bare Linux install to a working ROCm + PyTorch stack on a gfx1030 card, and
explains how to preview this wiki locally.

## 1. Confirm your hardware

Make sure your card is a Navi 21 gfx1030 (or a [related RDNA2 card](./hsa-override.md)):

```sh
lspci | grep -i vga
# After ROCm is installed:
rocminfo | grep -m1 -o 'gfx[0-9]*'
```

See [Supported Hardware](./hardware.md) for the full card list.

## 2. Install ROCm

Follow [Installing ROCm](./installing-rocm.md). At a high level:

```sh
# Add the amdgpu repo, then:
sudo apt install rocm
sudo usermod -aG render,video "$LOGNAME"
# Reboot, then verify:
rocminfo
clinfo | grep -i 'gfx\|Board'
```

## 3. (Optional) Tune the card

If you run one or more **Radeon PRO V620** (or **PRO W6800**) cards, these tweaks are worth doing
before you load models (Fedora or Ubuntu 26.04 for power cap; P2P is Fedora + AMD-validated — A/B
on Intel, see [When P2P helps](../tuning/p2p.md#when-p2p-helps--and-when-it-does-not)):

- [Power Tuning](../tuning/power.md) — drop the VBIOS-locked 250 W floor to 120 W and boot-cap at 180 W.
- [Disabling ECC](../tuning/ecc.md) — Pro cards hide ~2 GB behind ECC; optional extra VRAM.
- [Multi-GPU PCIe P2P](../tuning/p2p.md) — enable GPU↔GPU peer-to-peer for multi-card setups.

Power and P2P come from the [`v620_toolbox`](https://github.com/blivioniag/v620_toolbox) repo.

## 4. Run an inference stack (Docker)

The fastest path is the prebuilt images — no local ROCm/PyTorch/vLLM build required:

- [Running vLLM (Docker)](../vllm/overview.md) — [`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna)
  on a [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) PyTorch base.

Prefer GGUF and building from source? See [Building & Running llama.cpp](../llama-cpp/building.md) (ROCm or Vulkan).

Want to build the images yourself, or use the RDNA-tuned kernels? See
[Building the Images](../vllm/images.md) and [The rdna_extras fork](../vllm/fork.md).

## 5. Smoke test

```sh
# Inside a rocm-rdna / vllm-rdna container, or a local ROCm PyTorch env:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and your Radeon card's name. If not, see [Troubleshooting](../troubleshooting/index.md).

## 6. Join the community Discord

Questions, live benches, and fork updates land first on **gfx1030 club**:
[https://discord.gg/mESex2aBp](https://discord.gg/mESex2aBp) (`#vllm-rdna`, `#llamacpp`, `#general`,
`#benchmarks`). See [Useful resources](../meta/resources.md#community-discord).

---

## Previewing this wiki locally

This site is built with [mdBook](https://rust-lang.github.io/mdBook/). You do **not** need a GPU or
ROCm to work on the docs.

### Prerequisites

- **Git** — to clone the repository.
- **mdBook** — a single static binary (no runtime dependencies).

### Install mdBook

The quickest way is to grab a prebuilt binary from the
[mdBook releases page](https://github.com/rust-lang/mdBook/releases):

```sh
mkdir -p "$HOME/.local/bin"
MDBOOK_VERSION=v0.5.4
curl -sL "https://github.com/rust-lang/mdBook/releases/download/${MDBOOK_VERSION}/mdbook-${MDBOOK_VERSION}-x86_64-unknown-linux-gnu.tar.gz" \
  | tar -xz -C "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
mdbook --version
```

If you have a Rust toolchain, `cargo install mdbook` also works.

### Build and preview

```sh
git clone https://github.com/blivioniag/gfx1030-wiki.git
cd gfx1030-wiki

mdbook build      # outputs static HTML to ./book
mdbook serve      # live-reloading preview at http://localhost:3000
```

### Add content

1. Create a new markdown file in `src/`, e.g. `src/my_page.md`.
2. Add an entry for it in `src/SUMMARY.md`.
3. Re-run `mdbook serve` to preview.
4. Open a pull request against `master`.

See [Contributing](../meta/contributing.md) for the full guidelines.
