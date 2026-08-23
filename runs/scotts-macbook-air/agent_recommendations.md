# LMS Agent Recommendations

- Generated UTC: `2026-07-15T15:50:47.322404+00:00`
- Run directory: `/home/scott/git/lms/runs/scotts-macbook-air`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `structured_output` | `x1-370` | `liquid/lfm2.5-1.2b` | 1.0000 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.363; tps=46.748; max_ctx= |
| `debugging` | `x1-370` | `liquid/lfm2.5-1.2b` | 1.0000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.310; tps=58.029; max_ctx= |
| `repo_work` | `x1-370` | `liquid/lfm2.5-1.2b` | 1.0000 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.328; tps=60.438; max_ctx= |
| `structured_output` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 1.0000 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.538; tps=44.767; max_ctx= |
| `debugging` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 1.0000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.372; tps=57.361; max_ctx= |
| `repo_work` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 1.0000 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.333; tps=61.360; max_ctx= |
| `debugging` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9800 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=5.306; tps=34.655; max_ctx= |
| `safety` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.9109 | A |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=0.290; tps=59.863; max_ctx= |
| `safety` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.9109 | A |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=0.293; tps=59.972; max_ctx= |
| `agent_planning` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.9100 | A |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.279; tps=71.334; max_ctx= |
| `agent_planning` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.9100 | A |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.270; tps=71.019; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9038 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.140; tps=14.339; max_ctx= |
| `operational_health` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8901 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.280; tps=10.697; max_ctx= |
| `long_context` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8879 | B | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=4.542; tps=10.107; max_ctx=4096 |
| `long_context` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8733 | B | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=7.358; tps=6.212; max_ctx=4096 |
| `operational_health` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8518 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=6.213; tps=0.483; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b` | 0.8305 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=42.527; tps=8.143; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8199 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=8.534; tps=29.047; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.7750 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.351; tps=58.078; max_ctx= |
| `coding` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.7750 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.410; tps=57.640; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.7564 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=6.913; tps=35.041; max_ctx= |
| `agent_planning` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.7300 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.4000; ttft=3.021; tps=45.658; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.6632 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6667; ttft=17.498; tps=3.519; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b` | 0.6245 | C |  | task=debugging; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6000; ttft=37.754; tps=1.205; max_ctx= |
| `long_context` | `x1-370` | `refinedtoolcallv5-3b` | 0.5813 | C | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.5000; ttft=18.298; tps=1.685; max_ctx=4096 |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b` | 0.5750 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.4236 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=6.419; tps=6.303; max_ctx= |
| `structured_output` | `x1-370` | `refinedtoolcallv5-3b` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b` | 0.3500 | D |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b` | 0.3500 | D |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
