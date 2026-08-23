# LMS Agent Recommendations

- Generated UTC: `2026-07-15T20:31:28.469828+00:00`
- Run directory: `/home/scott/git/lms/runs/beelink-ryzen-7-mini-pc`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.9195 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.730; tps=18.534; max_ctx= |
| `structured_output` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.9066 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.575; tps=15.093; max_ctx= |
| `repo_work` | `x1-370` | `vibethinker-3b-heretic_decensored` | 0.9056 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.654; tps=14.836; max_ctx= |
| `debugging` | `x1-370` | `vibethinker-3b-heretic_decensored` | 0.9000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.119; tps=13.345; max_ctx= |
| `debugging` | `x1-370` | `vibethinker-3b-hermes` | 0.8909 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.895; tps=10.909; max_ctx= |
| `safety` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8904 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.9271; ttft=0.681; tps=19.508; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b` | 0.8859 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=2.730; tps=9.563; max_ctx= |
| `repo_work` | `x1-370` | `vibethinker-3b-hermes` | 0.8851 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.101; tps=9.351; max_ctx= |
| `safety` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8842 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=1.00; eval_score=0.9688; ttft=2.676; tps=9.120; max_ctx= |
| `repo_work` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8828 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.286; tps=8.756; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b` | 0.8823 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.636; tps=8.611; max_ctx= |
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b` | 0.8819 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.302; tps=8.506; max_ctx= |
| `debugging` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8812 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.570; tps=8.324; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b` | 0.8755 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.855; tps=6.803; max_ctx= |
| `coding` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8755 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.896; tps=6.798; max_ctx= |
| `operational_health` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8659 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.357; tps=4.252; max_ctx= |
| `repo_work` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8556 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=1.174; tps=21.487; max_ctx= |
| `agent_planning` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8470 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.649; tps=23.210; max_ctx= |
| `safety` | `x1-370` | `vibethinker-3b-heretic_decensored` | 0.8274 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8334; ttft=1.948; tps=13.962; max_ctx= |
| `agent_planning` | `x1-370` | `vibethinker-3b-heretic_decensored` | 0.8199 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=1.594; tps=15.975; max_ctx= |
| `long_context` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8166 | B | 4096 | task=long_context; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=9.616; tps=4.420; max_ctx=4096 |
| `coding` | `x1-370` | `vibethinker-3b-hermes` | 0.8127 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=1.885; tps=10.057; max_ctx= |
| `repo_work` | `x1-370` | `orinth-1.0-9b` | 0.8088 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=88.871; tps=2.348; max_ctx= |
| `debugging` | `x1-370` | `orinth-1.0-9b` | 0.8056 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=91.099; tps=1.496; max_ctx= |
| `debugging` | `x1-370` | `google/gemma-4-12b-qat` | 0.8025 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=283.106; tps=0.672; max_ctx= |
| `agent_planning` | `x1-370` | `vibethinker-3b-hermes` | 0.8016 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=1.445; tps=11.097; max_ctx= |
| `safety` | `x1-370` | `vibethinker-3b-hermes` | 0.8012 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=1.715; tps=10.739; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b` | 0.7936 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=2.459; tps=8.950; max_ctx= |
| `agent_planning` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.7905 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=2.358; tps=8.128; max_ctx= |
| `safety` | `x1-370` | `qwen3.6-28b-reap20-a3b` | 0.7438 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8750; ttft=490.981; tps=0.016; max_ctx= |
| `safety` | `x1-370` | `google/gemma-4-12b-qat` | 0.7144 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=312.086; tps=0.934; max_ctx= |
| `agent_planning` | `x1-370` | `google/gemma-4-12b-qat` | 0.7136 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=344.961; tps=0.967; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.6951 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.647; tps=18.699; max_ctx= |
| `long_context` | `x1-370` | `orinth-1.0-9b` | 0.6884 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.7500; ttft=138.779; tps=0.249; max_ctx= |
| `coding` | `x1-370` | `vibethinker-3b-heretic_decensored` | 0.6694 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=1.847; tps=11.847; max_ctx= |
| `operational_health` | `x1-370` | `vibethinker-3b-hermes` | 0.6615 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=1.131; tps=9.745; max_ctx= |
| `operational_health` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.6577 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.314; tps=8.713; max_ctx= |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b` | 0.6575 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.315; tps=8.677; max_ctx= |
| `safety` | `x1-370` | `orinth-1.0-9b` | 0.6561 | C |  | task=safety; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6771; ttft=476.489; tps=0.362; max_ctx= |
| `long_context` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.6552 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.549; tps=8.046; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
