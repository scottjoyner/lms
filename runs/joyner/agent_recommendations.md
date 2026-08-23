# LMS Agent Recommendations

- Generated UTC: `2026-07-15T16:07:05.155327+00:00`
- Run directory: `/home/scott/git/lms/runs/joyner`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b` | 0.8729 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.881; tps=6.105; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b` | 0.8722 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.767; tps=5.918; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b` | 0.8661 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.811; tps=4.285; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b` | 0.7835 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=1.938; tps=6.263; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b` | 0.7631 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.7604; ttft=2.409; tps=5.589; max_ctx= |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b` | 0.6483 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.209; tps=6.221; max_ctx= |
| `long_context` | `x1-370` | `refinedtoolcallv5-3b` | 0.6474 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.399; tps=5.965; max_ctx= |
| `operational_health` | `x1-370` | `ornith-1.0-9b` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `refinedtoolcallv5-3b` | 0.4249 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=0.247; tps=6.652; max_ctx= |
| `structured_output` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `google/gemma-4-12b-qat` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
