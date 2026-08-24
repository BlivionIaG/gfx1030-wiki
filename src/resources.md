# Useful Resources

## This project's repos & images

- [`v620_toolbox`](https://github.com/blivioniag/v620_toolbox) — V620/gfx1030 tuning: power cap + PCIe
  P2P recipes and scripts.
- [`vllm-rdna-docker`](https://github.com/blivioniag/vllm-rdna-docker) — the build system for the RDNA
  base and vLLM images.
- [`blivioniag/vllm` @ `rdna2_extras`](https://github.com/blivioniag/vllm/tree/rdna2_extras) — vLLM fork
  with custom RDNA2 HIP kernels.
- [`leapdragon/vllm-rdna2-recipe`](https://github.com/leapdragon/vllm-rdna2-recipe) — community vLLM
  serving recipes for RDNA2.
- [vLLM upstream PR #52391](https://github.com/vllm-project/vllm/pull/52391) — RDNA gfx1030 platform
  detection (aims to make gfx1030 work from upstream vLLM CI).

Docker Hub:

- [`blivioniag/rocm-rdna`](https://hub.docker.com/r/blivioniag/rocm-rdna) — ROCm + PyTorch base for RDNA.
- [`blivioniag/vllm-rdna`](https://hub.docker.com/r/blivioniag/vllm-rdna) — vLLM images (upstream +
  `-extras`).

## Official AMD / ROCm

- [ROCm documentation](https://rocm.docs.amd.com/)
- [ROCm install on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [System requirements & supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [GPU architecture specs (Navi 21 = gfx1030)](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)

## Upstream projects

- [vLLM](https://github.com/vllm-project/vllm) — the upstream serving engine this fork tracks.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF LLM inference (HIP/ROCm and Vulkan backends).
- [`edwinbrowwn/llama.cpp-rdna2`](https://github.com/edwinbrowwn/llama.cpp-rdna2) — RDNA2/V620-optimized
  llama.cpp fork (tensor-parallel + MMQ); see [the fork page](./llama_cpp_rdna2_fork.md).
- [TheRock: adding support for RDNA2 (gfx103X) cards](https://github.com/ROCm/TheRock/pull/1629)
- [ROCm Device Support Wishlist (community-tracked support matrix)](https://github.com/ROCm/ROCm/discussions/4276)

## Related community wikis

- [wiki-gfx906](https://github.com/skyne98/wiki-gfx906) — the sibling wiki for gfx906 (Vega 20 / MI50)
  that inspired this one.

## Tools

- `rocminfo`, `rocm-smi`, `amd-smi`, `clinfo` — ship with ROCm; your first stop for diagnosing GPU
  visibility and P2P topology.

> Know a great gfx1030 resource that isn't listed? Please [contribute](./contributing.md) a link.
