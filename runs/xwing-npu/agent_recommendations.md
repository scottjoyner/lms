# LMS Agent Recommendations

- Generated UTC: `2026-08-27T15:02:51.079644+00:00`
- Run directory: `/home/scott/git/lms/runs/xwing-npu`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- Benchmark runner: `x1-370`; endpoint target: `xwing` via Tailscale (`100.108.99.47:1237`). Machine hardware profile below describes the runner, not the endpoint target.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 1.0000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=4.653; tps=68.321; max_ctx= |
| `repo_work` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.9500 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=23.227; tps=100.618; max_ctx= |
| `structured_output` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.9250 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=1.095; tps=40.179; max_ctx= |
| `agent_planning` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.9100 | A |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=7.037; tps=87.760; max_ctx= |
| `operational_health` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.8938 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.257; tps=11.691; max_ctx= |
| `safety` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.8359 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.6354; ttft=3.201; tps=74.433; max_ctx= |
| `long_context` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.7942 | B | 2048 | task=long_context; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.7500; ttft=2.393; tps=15.128; max_ctx=2048 |
| `coding` | `x1-370` | `xwing-npu-qwen2.5-0.5b` | 0.7750 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=4.935; tps=68.474; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
