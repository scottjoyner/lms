# LMS Agent Recommendations

- Generated UTC: `2026-07-15T16:25:12.533810+00:00`
- Run directory: `/home/scott/git/lms/runs/destroyer`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `destroyer` | `lfm2.5-1.2b-instruct` | 0.8802 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.083; tps=8.057; max_ctx= |
| `structured_output` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8717 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.911; tps=5.781; max_ctx= |
| `debugging` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8696 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.712; tps=5.226; max_ctx= |
| `coding` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8692 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.590; tps=5.129; max_ctx= |
| `operational_health` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8570 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.617; tps=1.855; max_ctx= |
| `operational_health` | `destroyer` | `lfm2.5-1.2b-instruct` | 0.8559 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.906; tps=1.573; max_ctx= |
| `structured_output` | `destroyer` | `lfm2.5-1.2b-instruct` | 0.8530 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.447; tps=0.799; max_ctx= |
| `repo_work` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8176 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=14.528; tps=4.705; max_ctx= |
| `safety` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8170 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=11.243; tps=4.529; max_ctx= |
| `long_context` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.8011 | B | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=155.863; tps=0.290; max_ctx=4096 |
| `agent_planning` | `destroyer` | `lfm2.5-1.2b-instruct` | 0.7997 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=2.967; tps=10.594; max_ctx= |
| `agent_planning` | `destroyer` | `liquid/lfm2-24b-a2b` | 0.7294 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=10.158; tps=5.176; max_ctx= |
| `coding` | `destroyer` | `lfm2.5-1.2b-instruct` | 0.6561 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=3.660; tps=8.280; max_ctx= |
| `coding` | `destroyer` | `lfm2.5-8b-a1b` | 0.5809 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=98.616; tps=1.578; max_ctx= |
| `operational_health` | `destroyer` | `lfm2.5-8b-a1b` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `structured_output` | `destroyer` | `lfm2.5-8b-a1b` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `destroyer` | `lfm2.5-8b-a1b` | 0.3500 | D |  | task=debugging; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `destroyer` | `mradermacher/vibethinker-3b-hermes` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `destroyer` | `refinedneuro/vibethinker-3b-hermes` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `destroyer` | `vibethinker-3b-i1` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
