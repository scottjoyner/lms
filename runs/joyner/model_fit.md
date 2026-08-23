# LMS Model Fit Report

- Run directory: `/home/scott/git/lms/runs/joyner`

| Model | Params B | Quant | Est. GiB | Fit | Notes |
|---|---:|---|---:|---|---|
| `google/gemma-4-12b-qat` | 12.0 | unknown_assume_q4 | 7.54 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `ornith-1.0-9b` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `refinedtoolcallv5-3b` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `text-embedding-nomic-embed-text-v1.5` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |

## Notes

- These estimates are heuristic and based on model naming conventions.
- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.
- Benchmark load success and runtime stability remain the source of truth.
