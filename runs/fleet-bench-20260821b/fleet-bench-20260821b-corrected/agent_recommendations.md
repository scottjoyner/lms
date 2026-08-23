# LMS Agent Recommendations

- Generated UTC: `2026-08-22T07:58:25.030992+00:00`
- Run directory: `runs/fleet-bench-20260821b/fleet-bench-20260821b-corrected`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 5 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.9077 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.636; tps=15.384; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.9023 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.779; tps=13.949; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.8674 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.562; tps=4.628; max_ctx= |
| `debugging` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8668 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.943; tps=4.475; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.8642 | B | 8192 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=7.187; tps=3.799; max_ctx=8192 |
| `structured_output` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8635 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.023; tps=3.600; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8589 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.239; tps=2.377; max_ctx= |
| `debugging` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8561 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.251; tps=1.636; max_ctx= |
| `structured_output` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8555 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.149; tps=1.476; max_ctx= |
| `long_context` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8553 | B | 8192 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.381; tps=1.422; max_ctx=8192 |
| `operational_health` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8544 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.240; tps=1.162; max_ctx= |
| `operational_health` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8526 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.656; tps=0.692; max_ctx= |
| `safety` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8359 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.9271; ttft=2.987; tps=4.989; max_ctx= |
| `debugging` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8300 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=21.225; tps=8.011; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.8136 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=0.702; tps=10.305; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8100 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=11.684; tps=2.664; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.8091 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=0.600; tps=9.107; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8081 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=8.446; tps=2.171; max_ctx= |
| `repo_work` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8065 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=11.976; tps=1.732; max_ctx= |
| `debugging` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8057 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=9.243; tps=1.530; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.8044 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=1.006; tps=11.576; max_ctx= |
| `structured_output` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8035 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=15.113; tps=0.930; max_ctx= |
| `operational_health` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8011 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=8.240; tps=0.284; max_ctx= |
| `long_context` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8006 | B | 8192 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=271.245; tps=0.169; max_ctx=8192 |
| `agent_planning` | `x1-370` | `qwen3.5-0.8b-mlx` | 0.7997 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.675; tps=10.575; max_ctx= |
| `repo_work` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.7960 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=3.133; tps=5.611; max_ctx= |
| `agent_planning` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.7821 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=2.924; tps=5.881; max_ctx= |
| `agent_planning` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.7669 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=8.000; tps=1.848; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.7536 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=41.556; tps=7.620; max_ctx= |
| `safety` | `x1-370` | `ternary-bonsai-27b@?` | 0.7437 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8750; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `ternary-bonsai-27b@?` | 0.7437 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8750; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.7271 | C | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=0.67; eval_score=0.8333; ttft=78.854; tps=0.557; max_ctx=4096 |
| `safety` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.7174 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=8.943; tps=1.728; max_ctx= |
| `safety` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.7171 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=8.686; tps=1.648; max_ctx= |
| `agent_planning` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.7158 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=10.378; tps=1.535; max_ctx= |
| `repo_work` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.6560 | C |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6667; ttft=12.359; tps=1.598; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.6425 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=3.162; tps=4.665; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.6217 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5417; ttft=25.732; tps=7.454; max_ctx= |
| `operational_health` | `x1-370` | `ternary-bonsai-27b@?` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `ternary-bonsai-27b@?` | 0.5750 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
