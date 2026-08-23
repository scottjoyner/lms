# Model Report: `prism-ml/bonsai-27b`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://127.0.0.1:1234/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 0.018 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.013 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.020 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.006 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.020 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.020 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.008 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.009 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.006 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.006 |  |  | `` | `HTTP 500: {
    "error": {
        "message": "Failed to resolve model metadata for prism-ml/bonsai-27b.",
        "type` |