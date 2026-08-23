# LMS Agent Recommendations

- Generated UTC: `2026-07-15T14:30:41.227221+00:00`
- Run directory: `/home/scott/git/lms/runs/scott-optiplex-9030-aio`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8405 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=21.336; tps=10.805; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8110 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=16.923; tps=2.929; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.7675 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=27.353; tps=11.331; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.7252 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.7604; ttft=26.014; tps=8.803; max_ctx= |
| `agent_planning` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.6625 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6000; ttft=13.684; tps=11.324; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.6537 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6667; ttft=63.269; tps=0.981; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.3507 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=82.367; tps=0.193; max_ctx= |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `vibethinker-3b-i1` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-2b-claude-4.6-opus-reasoning-distilled` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `lfm2.5-8b-a1b` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
