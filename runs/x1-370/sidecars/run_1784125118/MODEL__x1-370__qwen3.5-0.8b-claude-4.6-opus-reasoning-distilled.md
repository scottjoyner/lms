# Model Report: `qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://127.0.0.1:1234/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 1.126 |  |  | `` | `HTTP 400: {"error":"Model is unloaded."}` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 1.138 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.959 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.905 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.706 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.710 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.829 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.732 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.619 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.664 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.737 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled\"` |