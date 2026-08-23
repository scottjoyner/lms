# Model Report: `text-embedding-nomic-embed-text-v1.5`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://destroyer.tailcb8954.ts.net:1234/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 0.024 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.008 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.009 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.010 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.010 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.011 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.010 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.013 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.012 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.008 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.006 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Invalid model identifier \"text-embedding-nomic-embed-text-v1.5\". Please` |