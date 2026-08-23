# Model Report: `text-embedding-nomic-embed-text-v1.5`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://100.78.106.121:1234/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 0.013 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.008 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 0.008 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.013 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.007 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.009 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.010 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.007 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.005 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "No models loaded. Please load a model in the developer page or use the 'l` |