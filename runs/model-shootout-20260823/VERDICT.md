# Daily-Driver Shootout — Qwen3.8-27B-NVFP4 vs Ornith-1.5-35B-A3B (RX 9070 XT 32 GB)

Date: 2026-08-23 · Host: x1-370 · LM Studio ROCm backend 2.29.1

## Speedups tested ("dflash" and friends)

The available engine-level speedups for these two models are MTP draft
(`--speculative-draft-mtp`, model-native MTP weights) and simple draft
(`--speculative-draft-simple --speculative-draft-model <small-qwen3.5>`).
(The `dflash` label on poolside laguna variants is a model-format speedup; no
dflash build exists for qwen3.8/ornith.)

| Model | Config | c1 tps | c4 tps | c8 tps |
|---|---|---:|---:|---:|
| qwen3.8-27b | plain, FA on, KV q8_0, parallel 4/8 | **26.3** | **77.2** | ~68–77 |
| qwen3.8-27b | + MTP draft | 23.7 | 36.7 | 36.2 |
| qwen3.8-27b | + draft-simple (Qwen3.5-0.8B draft, n=4) | 18.9 | 28.8 | 33.1 |
| ornith-1.5-35b | plain, FA on, KV q8_0, parallel 16 | **66** | **136–141** | **155–180** |
| ornith-1.5-35b | + MTP n-max 2 | 44.1 | 85.8 | 88.5 |
| ornith-1.5-35b | + MTP n-max 4, p-min 0.5 | 43.2 | — | 79.4 |

**Every speculative mode is a net loss on this GPU** — the Navi 48 is already
compute-saturated, so draft compute directly steals from the target. Plain
load wins for both models.

## Quality (agent_skill_suite.v1, 8k ctx)

| task_family | ornith-1.5-35b | qwen3.8-27b |
|---|---:|---:|
| agent_planning | **0.400** | 0.000 |
| coding | 0.500 | 0.500 |
| debugging | **1.000** | 0.000 |
| long_context | 0.750 | **1.000** |
| operational_health | 0.500 | 0.500 |
| repo_work | **1.000** | 0.167 |
| safety | 0.583 | **0.875** |
| structured_output | **1.000** | 0.000 |
| **overall eval_score_avg** | **0.7106** | 0.5379 |
| eval tps_med / ttft_med s | **28.5 / 3.7** | 3.3 / 13.2 |

## Verdict

**Ornith-1.5-35B-A3B-APEX-MTP (MTP off) is the daily driver.**
- 2.5× faster single-stream, 2.3× faster aggregate
- Higher overall quality (0.71 vs 0.54), dramatically better at agentic work
  (debugging/repo_work/structured_output all perfect)
- 65k context vs 32k practical
- Costs more VRAM (22.1 vs 17.2 GiB loaded) — acceptable with 10 GiB headroom

qwen3.8 keeps two niches: safety-sensitive routing and max-fidelity long
context — both at ⅑ the token rate.

Production config left in place: `lms load ornith-1.5-35b-a3b-apex-mtp --gpu max`
(per-model defaults: ctx 65536, parallel 16, FA on, KV q8_0/q8_0, threads 12,
MTP off).
