# LMS Model Fit Report

- Run directory: `/home/scott/git/lms/runs/xwing-npu`

| Model | Params B | Quant | Est. GiB | Fit | Notes |
|---|---:|---|---:|---|---|
| `xwing-npu-qwen2.5-0.5b` | 0.5 | unknown_assume_q4 | 0.31 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |

## Notes

- These estimates are heuristic and based on model naming conventions.
- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.
- Benchmark load success and runtime stability remain the source of truth.
