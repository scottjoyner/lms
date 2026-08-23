# LMS Agent Recommendations

- Generated UTC: `2026-07-15T21:56:01.445574+00:00`
- Run directory: `/home/scott/git/lms/runs/lenovo-ideapad-330s-15ikb`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `structured_output` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8589 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=7.765; tps=2.367; max_ctx= |
| `operational_health` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8541 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=2.735; tps=1.097; max_ctx= |
| `debugging` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8522 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=6.508; tps=0.580; max_ctx= |
| `debugging` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.8521 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=6.808; tps=0.573; max_ctx= |
| `repo_work` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8116 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=15.224; tps=3.085; max_ctx= |
| `structured_output` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8077 | B |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=11.112; tps=2.059; max_ctx= |
| `repo_work` | `x1-370` | `google/gemma-4-e2b` | 0.8048 | B |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=303.374; tps=1.284; max_ctx= |
| `debugging` | `x1-370` | `google/gemma-4-e2b` | 0.8038 | B |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=186.007; tps=1.005; max_ctx= |
| `coding` | `x1-370` | `lfm2.5-8b-a1b-terminal-toolbench-full-sft-1epoch` | 0.8025 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=213.267; tps=0.659; max_ctx= |
| `operational_health` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.8002 | B |  | task=operational_health; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=56.170; tps=0.053; max_ctx= |
| `agent_planning` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.7740 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=5.625; tps=3.725; max_ctx= |
| `agent_planning` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.7697 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=5.797; tps=2.577; max_ctx= |
| `safety` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.7532 | B |  | task=safety; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.7604; ttft=6.189; tps=2.946; max_ctx= |
| `repo_work` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.7267 | C |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=602.371; tps=0.445; max_ctx= |
| `coding` | `x1-370` | `google/gemma-4-e2b` | 0.7259 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=427.208; tps=0.255; max_ctx= |
| `agent_planning` | `x1-370` | `google/gemma-4-e2b` | 0.7134 | C |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=290.913; tps=0.908; max_ctx= |
| `coding` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.6529 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6667; ttft=13.658; tps=0.758; max_ctx= |
| `coding` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.6357 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=7.242; tps=2.859; max_ctx= |
| `debugging` | `x1-370` | `lfm2.5-8b-a1b-terminal-toolbench-full-sft-1epoch` | 0.6211 | C |  | task=debugging; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6000; ttft=309.005; tps=0.289; max_ctx= |
| `coding` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.5857 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=11.501; tps=2.862; max_ctx= |
| `operational_health` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.5750 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `lfm2.5-8b-a1b-terminal-toolbench-full-sft-1epoch` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.5311 | D |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.4000; ttft=824.569; tps=0.286; max_ctx= |
| `long_context` | `x1-370` | `liquid/lfm2.5-1.2b` | 0.4631 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.2500; ttft=36.555; tps=0.167; max_ctx= |
| `operational_health` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.4052 | D |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=4.312; tps=1.382; max_ctx= |
| `safety` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.4042 | D |  | task=safety; ok_rate=0.50; eval_ok_rate=0.50; eval_score=0.5000; ttft=29.850; tps=1.125; max_ctx= |
| `safety` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.4000 | D |  | task=safety; ok_rate=0.50; eval_ok_rate=0.50; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `lfm2.5-8b-a1b-terminal-toolbench-full-sft-1epoch` | 0.4000 | D |  | task=safety; ok_rate=0.50; eval_ok_rate=0.50; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.3549 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=14.127; tps=1.310; max_ctx= |
| `debugging` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.3515 | D |  | task=debugging; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=125.265; tps=0.398; max_ctx= |
| `long_context` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.3508 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=316.025; tps=0.209; max_ctx= |
| `repo_work` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.3506 | D |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=569.413; tps=0.151; max_ctx= |
| `agent_planning` | `x1-370` | `vibethinker-3b-heretic-i1` | 0.3500 | D |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `google/gemma-4-e2b` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `google/gemma-4-e2b` | 0.3500 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.3500 | D |  | task=debugging; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `qwopus3.5-4b-coder-mtp` | 0.3500 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
