# Experimental CPU/GPU expert alignment

When a routed expert runs on the CPU, its activation quantization and dot-product order can differ from the GPU expert kernels. This experiment brings the supported CPU paths closer to the GPU arithmetic for Flash-Next experts with a 2560-wide input and a 640-wide hidden layer. It does not change the weights, expert placement, or cache configuration.

Build with `-DSTRATA_BUILD_CPU_EXPERT_ALIGNMENT=ON` on Linux x86-64 with GCC and the native Strata CPU quant layer. The runtime paths require AVX2, FMA and F16C. The build option and the runtime settings are off by default.

Set these values in the engine's `env` for the combined experiment:

| Setting | Value | Effect |
| --- | --- | --- |
| `STRATA_CPU_Q81_GU_ALIGN` | `1` | Use GPU-style Q8_1 activation blocks and accumulation order for supported IQ gate/up rows. |
| `STRATA_CPU_GPU_SILU_EXP2` | `1` | Use an exp2-based CPU SwiGLU approximation in the aligned gate/up path. |
| `STRATA_CPU_Q81_Q2_DOWN_SCALE` | `1` | Round the activation scales of the 640-wide Q2_0 down path to FP16. |
| `STRATA_CPU_IQ4NL_HIDDEN_ROUND` | `1` | Match GPU rounding when packing the 640 hidden activations for IQ4_NL down rows. |
| `STRATA_CPU_Q81_DOWN_ORDER` | `1` | Use GPU-style accumulation order for supported IQ4_NL down rows. |
| `STRATA_CPU_DOWN_REUSE` | `1` | Reuse decoded values in supported IQ4_NL down batches. The GPU-order path takes precedence when eligible. |
| `STRATA_CPU_DOWN_BATCH_REUSE` | `1` | Prepare down activations once per expert batch and publish them to the pool workers. |

This is a numerical comparison with a GPU reference. The exp2 approximation can still differ from the GPU intrinsic, and closer expert outputs do not guarantee better answers. Recheck numerical accuracy, serving speed, and answer quality on the revision and model being deployed.
