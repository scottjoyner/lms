# LMS Model Fit Report

- Run directory: `/home/scott/git/lms/runs/x1-370`

| Model | Params B | Quant | Est. GiB | Fit | Notes |
|---|---:|---|---:|---|---|
| `google/gemma-4-12b-qat` | 12.0 | unknown_assume_q4 | 7.54 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `hermes-qwen3.5-0.8b-sft-v7-fresh` | 0.8 | unknown_assume_q4 | 0.5 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `laguna-s-2.1` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `laguna-xs-2.1` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `lfm2.5-1.2b-instruct` | 1.2 | unknown_assume_q4 | 0.75 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `lfm2.5-8b-a1b` | 8.0 | unknown_assume_q4 | 5.03 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `liquid/lfm2-24b-a2b` | 24.0 | unknown_assume_q4 | 16.21 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `lmstudio-community/laguna-xs-2.1` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `minicpm5-1b-agentic-tooluse` | 1.0 | unknown_assume_q4 | 0.63 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `mradermacher/vibethinker-3b-hermes` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `openai/gpt-oss-20b` | 20.0 | unknown_assume_q4 | 13.09 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `orinth-1.0-9b` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `ornith-1.0-35b-mtp-apex` | 35.0 | unknown_assume_q4 | 25.66 | borderline | Estimated model may fit, but expect pressure from KV cache, context length, and other processes. |
| `ornith-1.0-35b-rocmfpx` | 35.0 | unknown_assume_q4 | 25.66 | borderline | Estimated model may fit, but expect pressure from KV cache, context length, and other processes. |
| `poolside-laguna-s-2.1-dflash@?` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `poolside-laguna-s-2.1-dflash@q4_k_m` |  | Q4_K_M |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `poolside-laguna-s-2.1-dflash@q8_0` |  | Q8_0 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `poolside/laguna-xs-2.1` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `prism-ml/bonsai-27b` | 27.0 | unknown_assume_q4 | 18.66 | borderline | Estimated model may fit, but expect pressure from KV cache, context length, and other processes. |
| `qwen3.5-0.8b` | 0.8 | unknown_assume_q4 | 0.5 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8 | unknown_assume_q4 | 0.5 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-14b-a3b-claude-4.6-opus-reasoning-distilled-reap` | 14.0 | unknown_assume_q4 | 8.8 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored` | 9.0 | unknown_assume_q4 | 5.66 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.6-14b-a3b-vibeforged-v2` | 14.0 | unknown_assume_q4 | 8.8 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex` | 35.0 | unknown_assume_q4 | 25.66 | borderline | Estimated model may fit, but expect pressure from KV cache, context length, and other processes. |
| `qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp` | 35.0 | unknown_assume_q4 | 25.66 | borderline | Estimated model may fit, but expect pressure from KV cache, context length, and other processes. |
| `refinedneuro/vibethinker-3b-hermes` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `refinedtoolcallv5-3b` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `refinedtoolcallv5-3b@f16` |  | F16 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `refinedtoolcallv5-3b@q6_k` |  | Q6_K |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `refinedtoolcallv5-3b@q8_0` |  | Q8_0 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `ternary-bonsai-27b@?` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `ternary-bonsai-27b@q4_1` |  | Q4_1 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `text-embedding-nomic-embed-text-v1.5` |  | unknown_assume_q4 |  | unknown | Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory. |
| `vibethinker-3b` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |
| `vibethinker-3b-i1` | 3.0 | unknown_assume_q4 | 1.89 | good | Estimated model memory fits comfortably in currently available RAM/VRAM. |

## Notes

- These estimates are heuristic and based on model naming conventions.
- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.
- Benchmark load success and runtime stability remain the source of truth.
