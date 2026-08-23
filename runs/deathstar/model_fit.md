# LMS Model Fit Report

- Run directory: `/home/scott/git/lms/runs/deathstar`

| Model | Params B | Quant | Est. GiB | Fit | Notes |
|---|---:|---|---:|---|---|
| `gemma-4-e4b-uncensored-hauhaucs-aggressive` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `google/gemma-4-12b-qat` | 12.0 | unknown_assume_q4 | 7.54 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `lfm2-1.2b-tool` | 1.2 | unknown_assume_q4 | 0.75 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `lfm2.5-8b-a1b` | 8.0 | unknown_assume_q4 | 5.03 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `lfm2.5-8b-a1b-terminal-toolbench-full-sft-1epoch` | 8.0 | unknown_assume_q4 | 5.03 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `openai/gpt-oss-20b` | 20.0 | unknown_assume_q4 | 13.09 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `ornith-1.0-35b` | 35.0 | unknown_assume_q4 | 25.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `ornith-1.0-9b` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8 | unknown_assume_q4 | 0.5 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-4b-claude-4.6-opus-reasoning-distilled-v2` | 4.0 | unknown_assume_q4 | 2.51 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-4b-neo-heretic-i1` | 4.0 | unknown_assume_q4 | 2.51 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-4b-uncensored-hauhaucs-aggressive` | 4.0 | unknown_assume_q4 | 2.51 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-9b-neo` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-9b-neo-heretic-i1` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.6-12b-iq-ultra-heretic-uncensored-thinking-v2-hightop` | 12.0 | unknown_assume_q4 | 7.54 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex` | 35.0 | unknown_assume_q4 | 25.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `refinedtoolcallv5-3b` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `text-embedding-nomic-embed-text-v1.5` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `vibethinker-3b-hermes` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `vibethinker-3b-i1` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |

## Notes

- These estimates are heuristic and based on model naming conventions.
- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.
- Benchmark load success and runtime stability remain the source of truth.
