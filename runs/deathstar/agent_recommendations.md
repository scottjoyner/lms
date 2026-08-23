# LMS Agent Recommendations

- Generated UTC: `2026-07-15T15:50:54.941720+00:00`
- Run directory: `/home/scott/git/lms/runs/deathstar`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

| Task | Host | Model | Score | Grade | Max reliable context | Evidence |
|---|---|---|---:|---|---:|---|
| `debugging` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 1.0000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=3.840; tps=46.565; max_ctx= |
| `debugging` | `x1-370` | `refinedtoolcallv5-3b` | 1.0000 | A |  | task=debugging; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.592; tps=42.297; max_ctx= |
| `repo_work` | `x1-370` | `refinedtoolcallv5-3b` | 1.0000 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.728; tps=42.045; max_ctx= |
| `repo_work` | `x1-370` | `vibethinker-3b-i1` | 1.0000 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.948; tps=40.588; max_ctx= |
| `coding` | `x1-370` | `refinedtoolcallv5-3b` | 0.9864 | A |  | task=coding; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=0.580; tps=36.371; max_ctx= |
| `repo_work` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9250 | A |  | task=repo_work; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=1.985; tps=62.077; max_ctx= |
| `safety` | `x1-370` | `refinedtoolcallv5-3b` | 0.9250 | A |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8334; ttft=0.614; tps=42.968; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9241 | A |  | task=structured_output; ok_rate=1.00; eval_ok_rate=1.00; eval_score=1.0000; ttft=1.757; tps=19.747; max_ctx= |
| `safety` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.9109 | A |  | task=safety; ok_rate=1.00; eval_ok_rate=0.50; eval_score=0.8021; ttft=4.181; tps=48.391; max_ctx= |
| `agent_planning` | `x1-370` | `refinedtoolcallv5-3b` | 0.9100 | A |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8000; ttft=0.572; tps=42.871; max_ctx= |
| `agent_planning` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8200 | B |  | task=agent_planning; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.6000; ttft=2.068; tps=69.090; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.8075 | B |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.8333; ttft=7.606; tps=8.671; max_ctx= |
| `operational_health` | `x1-370` | `refinedtoolcallv5-3b` | 0.7461 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=0.380; tps=32.299; max_ctx= |
| `long_context` | `x1-370` | `vibethinker-3b-i1` | 0.6361 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=12.847; tps=16.301; max_ctx= |
| `long_context` | `x1-370` | `refinedtoolcallv5-3b` | 0.6345 | C |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=11.655; tps=15.857; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.6-12b-iq-ultra-heretic-uncensored-thinking-v2-hightop` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `qwen3.5-9b-neo` | 0.5750 | C |  | task=operational_health; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `qwen3.5-9b-neo` | 0.5750 | C |  | task=coding; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.5000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `refinedtoolcallv5-3b` | 0.5500 | C |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=0.651; tps=40.906; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled` | 0.5403 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.2500; ttft=5.056; tps=7.422; max_ctx= |
| `safety` | `x1-370` | `lfm2-1.2b-tool` | 0.5395 | D |  | task=safety; ok_rate=0.50; eval_ok_rate=0.50; eval_score=0.4688; ttft=3.747; tps=23.874; max_ctx= |
| `long_context` | `x1-370` | `qwen3.5-9b-neo-heretic-i1` | 0.5239 | D |  | task=long_context; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.3750; ttft=25.105; tps=1.371; max_ctx= |
| `debugging` | `x1-370` | `google/gemma-4-12b-qat` | 0.3500 | D |  | task=debugging; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.6-12b-iq-ultra-heretic-uncensored-thinking-v2-hightop` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `qwen3.5-9b-neo` | 0.3500 | D |  | task=structured_output; ok_rate=1.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `openai/gpt-oss-20b` | 0.1750 | D |  | task=long_context; ok_rate=0.50; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `long_context` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=long_context; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `repo_work` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=repo_work; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `safety` | `x1-370` | `text-embedding-nomic-embed-text-v1.5` | 0.0000 | F |  | task=safety; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `operational_health` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=operational_health; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `structured_output` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=structured_output; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `coding` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=coding; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `debugging` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=debugging; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |
| `agent_planning` | `x1-370` | `ornith-1.0-9b` | 0.0000 | F |  | task=agent_planning; ok_rate=0.00; eval_ok_rate=0.00; eval_score=0.0000; ttft=; tps=; max_ctx= |

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
