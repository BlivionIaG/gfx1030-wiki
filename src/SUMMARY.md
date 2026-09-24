# Summary

[Introduction](./intro.md)
- [llama.cpp or vLLM](./choose-a-stack.md)

# Setup

- [Getting Started](./setup/getting-started.md)
- [Supported Hardware](./setup/hardware.md)
- [Installing ROCm](./setup/installing-rocm.md)
- [HSA_OVERRIDE for RDNA2 Cousins](./setup/hsa-override.md)
- [Host firmware (MMIO High / SR-IOV)](./setup/host-firmware.md)

# Tuning (V620 / gfx1030)

- [Power Tuning (120 W floor + soft unlock)](./tuning/power.md)
- [Disabling ECC (Pro VRAM)](./tuning/ecc.md)
- [Multi-GPU PCIe P2P](./tuning/p2p.md)

# vLLM on RDNA

- [Overview](./vllm/overview.md)
- [Running (Docker)](./vllm/running.md)
- [Recipes (Hub and presets)](./vllm/recipes.md)
- [Flash-Next](./vllm/flash-next.md)
    - [Serve lines](./vllm/flash-next-serve.md)
- [Configuration](./vllm/configuration.md)
- [Quantization](./vllm/quantization.md)
- [vLLM forks (rdna_extras + landscape)](./vllm/fork.md)
- [Host venv (no Docker)](./vllm/host-venv.md)
- [Building images](./vllm/images.md)

# llama.cpp

- [Overview](./llama-cpp/overview.md)
- [Building & running (stock)](./llama-cpp/building.md)
- [RDNA2 fork](./llama-cpp/rdna2-overview.md)
    - [Benchmarks](./llama-cpp/rdna2-benchmarks.md)
    - [Speculative decoding](./llama-cpp/rdna2-speculative.md)
    - [Serving](./llama-cpp/rdna2-serving.md)

# Troubleshooting

- [Overview](./troubleshooting/index.md)
- [General](./troubleshooting/general.md)
- [vLLM](./troubleshooting/vllm.md)
- [vLLM Flash-Next](./troubleshooting/vllm-flash-next.md)
- [llama.cpp](./troubleshooting/llama-cpp.md)

# Reference

- [Environment variables](./reference/env-vars.md)
- [Verification status (WIP)](./reference/verification.md)

# About this wiki

- [Contributing](./meta/contributing.md)
- [Useful resources](./meta/resources.md)
- [Wiki structure](./meta/structure.md)
