# Model Report: `qwen3.5-2b-claude-4.6-opus-reasoning-distilled`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://scott-optiplex-9030-aio.tailcb8954.ts.net:1234/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 0.004 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.004 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.007 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.004 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.006 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.004 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.007 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.007 |  |  | `` | `HTTPConnectionPool(host='scott-optiplex-9030-aio.tailcb8954.ts.net', port=1234): Max retries exceeded with url: /v1/chat` |