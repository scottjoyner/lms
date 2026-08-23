# LMS Model Fit Report

- Run directory: `/home/scott/git/lms/runs/scott-optiplex-9030-aio`

| Model | Params B | Quant | Est. GiB | Fit | Notes |
|---|---:|---|---:|---|---|
| `google/gemma-3-1b` | 1.0 | unknown_assume_q4 | 0.63 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `ibm/granite-4-h-tiny` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `lfm2.5-8b-a1b` | 8.0 | unknown_assume_q4 | 5.03 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `liquid/lfm2.5-1.2b` | 1.2 | unknown_assume_q4 | 0.75 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8 | unknown_assume_q4 | 0.5 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-2b` | 2.0 | unknown_assume_q4 | 1.26 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 2.0 | unknown_assume_q4 | 1.26 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-4b` | 4.0 | unknown_assume_q4 | 2.51 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `refinedtoolcallv5-3b` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `text-embedding-nomic-embed-text-v1.5` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `vibethinker-3b-i1` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |

## Notes

- These estimates are heuristic and based on model naming conventions.
- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.
- Benchmark load success and runtime stability remain the source of truth.
