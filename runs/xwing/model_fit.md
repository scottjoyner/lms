# LMS Model Fit Report

- Run directory: `/home/scott/git/lms/runs/xwing`

| Model | Params B | Quant | Est. GiB | Fit | Notes |
|---|---:|---|---:|---|---|
| `orinth-1.0-9b` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `ornith-1.0-35b` | 35.0 | unknown_assume_q4 | 25.66 | borderline | Estimated model may fit, but expect pressure from KV cache, context length, and other processes. |
| `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8 | unknown_assume_q4 | 0.5 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.6-14b-a3b-vibeforged-v2` | 14.0 | unknown_assume_q4 | 8.8 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `text-embedding-nomic-embed-text-v1.5` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `vibethinker-3b-hermes` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |

## Notes

- These estimates are heuristic and based on model naming conventions.
- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.
- Benchmark load success and runtime stability remain the source of truth.
