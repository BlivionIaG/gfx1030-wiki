# llama.cpp troubleshooting

## KV checkpoint crash on tensor split

Symptom: fatal error in `ggml-backend-meta.cpp` during warmup with tensor split.

Fix: `--ctx-checkpoints 0`. Known on stock llama.cpp and the RDNA2 fork. See
[RDNA2 serving limits](../../llama-cpp/rdna2-serving.md#notable-limits).

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
