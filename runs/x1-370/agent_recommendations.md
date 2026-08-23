# LMS Agent Recommendations

- Generated UTC: `2026-08-03T21:40:25.562999+00:00`
- Run directory: `/home/scott/git/lms/runs/x1-370`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9480 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.376; tps=26.121; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9374 | A |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=7.188; tps=23.314; max_ctx= |
| `repo_work` | `x1-370` | `lfm2.5-1.2b-instruct` | 0.8961 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.669; tps=12.305; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.8890 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.415; tps=10.409; max_ctx= |
| `debugging` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8857 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=8.564; tps=22.855; max_ctx= |
| `debugging` | `x1-370` | `mradermacher/vibethinker-3b-hermes` | 0.8798 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=4.063; tps=7.938; max_ctx= |
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.8745 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.674; tps=6.541; max_ctx= |
| `debugging` | `x1-370` | `refinedneuro/vibethinker-3b-hermes` | 0.8729 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=6.032; tps=6.110; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b` | 0.8720 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=4.541; tps=5.854; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.8717 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=5.030; tps=5.778; max_ctx= |
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b` | 0.8698 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=6.753; tps=5.286; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b` | 0.8693 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=5.057; tps=5.134; max_ctx= |
| `safety` | `x1-370` | `lfm2.5-1.2b-instruct` | 0.8630 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.9271; ttft=2.025; tps=12.216; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8600 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=7.557; tps=2.655; max_ctx= |
| `operational_health` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8515 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=4.708; tps=0.409; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8360 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=7.337; tps=16.268; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8209 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=8.299; tps=25.565; max_ctx= |
| `coding` | `x1-370` | `openai/gpt-oss-20b` | 0.8128 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=15.010; tps=3.420; max_ctx= |
| `structured_output` | `x1-370` | `laguna-s-2.1` | 0.8081 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=18.314; tps=2.160; max_ctx= |
| `structured_output` | `x1-370` | `liquid/lfm2-24b-a2b` | 0.8070 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=10.108; tps=1.879; max_ctx= |
| `coding` | `x1-370` | `laguna-s-2.1` | 0.8060 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=21.632; tps=1.594; max_ctx= |
| `coding` | `x1-370` | `mradermacher/vibethinker-3b-hermes` | 0.8038 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=4.193; tps=7.692; max_ctx= |
| `long_context` | `x1-370` | `lfm2.5-1.2b-instruct` | 0.8031 | B | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=60.188; tps=0.831; max_ctx=4096 |
| `operational_health` | `x1-370` | `openai/gpt-oss-20b` | 0.8010 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=9.927; tps=0.278; max_ctx= |
| `operational_health` | `x1-370` | `laguna-s-2.1` | 0.8008 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=12.301; tps=0.207; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.7910 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.384; tps=8.275; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b` | 0.7783 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=4.833; tps=4.889; max_ctx= |
| `safety` | `x1-370` | `ornith-1.0-35b-mtp-apex` | 0.7471 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8750; ttft=63.594; tps=0.890; max_ctx= |
| `agent_planning` | `x1-370` | `mradermacher/vibethinker-3b-hermes` | 0.6959 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6000; ttft=4.047; tps=6.909; max_ctx= |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.6605 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.328; tps=9.471; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.6591 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.538; tps=9.096; max_ctx= |
| `coding` | `x1-370` | `refinedneuro/vibethinker-3b-hermes` | 0.6439 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=7.039; tps=5.035; max_ctx= |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b` | 0.6436 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=1.669; tps=4.949; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b` | 0.6420 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=5.446; tps=4.546; max_ctx= |
| `agent_planning` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.6400 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.2000; ttft=3.931; tps=42.568; max_ctx= |
| `long_context` | `x1-370` | `refinedtoolcallv5-3b@f16` | 0.5941 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=9.959; tps=5.101; max_ctx= |
| `long_context` | `x1-370` | `refinedtoolcallv5-3b` | 0.5790 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=191.733; tps=1.066; max_ctx= |
| `operational_health` | `x1-370` | `ornith-1.0-35b-mtp-apex` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `ornith-1.0-35b-mtp-apex` | 0.5750 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
