# Model Report: `lmstudio-beelink-ryzen-7-mini-pc.refinedtoolcallv5-3b`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://127.0.0.1:8088/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 0.196 |  |  | `` | `HTTP 500: Internal Server Error` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.201 |  |  | `` | `HTTP 500: Internal Server Error` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.194 |  |  | `` | `HTTP 500: Internal Server Error` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.204 |  |  | `` | `HTTP 500: Internal Server Error` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.250 |  |  | `` | `HTTP 500: Internal Server Error` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.261 |  |  | `` | `HTTP 500: Internal Server Error` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.359 |  |  | `` | `HTTP 500: Internal Server Error` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.236 |  |  | `` | `HTTP 500: Internal Server Error` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.168 |  |  | `` | `HTTP 500: Internal Server Error` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.209 |  |  | `` | `HTTP 500: Internal Server Error` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.215 |  |  | `` | `HTTP 500: Internal Server Error` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.199 |  |  | `` | `HTTP 500: Internal Server Error` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.201 |  |  | `` | `HTTP 500: Internal Server Error` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.254 |  |  | `` | `HTTP 500: Internal Server Error` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.205 |  |  | `` | `HTTP 500: Internal Server Error` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 1.662 |  |  | `` | `HTTP 500: Internal Server Error` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.252 |  |  | `` | `HTTP 500: Internal Server Error` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.254 |  |  | `` | `HTTP 500: Internal Server Error` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.198 |  |  | `` | `HTTP 500: Internal Server Error` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.263 |  |  | `` | `HTTP 500: Internal Server Error` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.213 |  |  | `` | `HTTP 500: Internal Server Error` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.197 |  |  | `` | `HTTP 500: Internal Server Error` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.186 |  |  | `` | `HTTP 500: Internal Server Error` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.238 |  |  | `` | `HTTP 500: Internal Server Error` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.201 |  |  | `` | `HTTP 500: Internal Server Error` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.199 |  |  | `` | `HTTP 500: Internal Server Error` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.226 |  |  | `` | `HTTP 500: Internal Server Error` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.456 |  |  | `` | `HTTP 500: Internal Server Error` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.167 |  |  | `` | `HTTP 500: Internal Server Error` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.408 |  |  | `` | `HTTP 500: Internal Server Error` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 1.263 |  |  | `` | `HTTP 500: Internal Server Error` |