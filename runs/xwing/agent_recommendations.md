# LMS Agent Recommendations

- Generated UTC: `2026-07-16T02:53:36.630542+00:00`
- Run directory: `/home/scott/git/lms/runs/xwing`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 1.0000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.124; tps=63.243; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 1.0000 | A |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=2.716; tps=69.377; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9818 | A |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=5.065; tps=35.149; max_ctx= |
| `repo_work` | `x1-370` | `vibethinker-3b-hermes` | 0.9433 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.559; tps=24.881; max_ctx= |
| `debugging` | `x1-370` | `vibethinker-3b-hermes` | 0.9364 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.387; tps=23.039; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9250 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=5.237; tps=76.594; max_ctx= |
| `debugging` | `x1-370` | `orinth-1.0-9b` | 0.8925 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=6.161; tps=11.328; max_ctx= |
| `agent_planning` | `x1-370` | `vibethinker-3b-hermes` | 0.8490 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.550; tps=23.739; max_ctx= |
| `safety` | `x1-370` | `orinth-1.0-9b` | 0.8384 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=14.884; tps=10.243; max_ctx= |
| `repo_work` | `x1-370` | `orinth-1.0-9b` | 0.8349 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=23.784; tps=9.309; max_ctx= |
| `agent_planning` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8200 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6000; ttft=2.718; tps=142.411; max_ctx= |
| `safety` | `x1-370` | `vibethinker-3b-hermes` | 0.8010 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.7084; ttft=0.426; tps=21.927; max_ctx= |
| `agent_planning` | `x1-370` | `orinth-1.0-9b` | 0.7536 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=14.773; tps=11.636; max_ctx= |
| `operational_health` | `x1-370` | `vibethinker-3b-hermes` | 0.7045 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.272; tps=21.190; max_ctx= |
| `coding` | `x1-370` | `vibethinker-3b-hermes` | 0.6985 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.456; tps=19.595; max_ctx= |
| `long_context` | `x1-370` | `orinth-1.0-9b` | 0.6930 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.7500; ttft=23.736; tps=1.460; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.6905 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=2.979; tps=17.479; max_ctx= |
| `long_context` | `x1-370` | `vibethinker-3b-hermes` | 0.6725 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=3.718; tps=12.661; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `orinth-1.0-9b` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `orinth-1.0-9b` | 0.5750 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `vibethinker-3b-hermes` | 0.4852 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=0.384; tps=22.724; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.4279 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=2.447; tps=7.433; max_ctx= |
| `structured_output` | `x1-370` | `orinth-1.0-9b` | 0.3823 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=8.133; tps=8.614; max_ctx= |
| `operational_health` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `ornith-1.0-35b` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
