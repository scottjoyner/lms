# Model Report: `ornith-1.0-35b-rocmfpx`

- Host: `x1-370` (`192.168.1.241`)
- Base URL: `http://127.0.0.1:1234/v1`

| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |
|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|
| `load_probe` | `operational_health` | `load` | ❌ |  |  | 0.003 |  |  | `` | `HTTP 500: <!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Error</title>
</head>
<body>
<pre>Intern` |
| `health_minimal_chat` | `operational_health` | `run` | ❌ | ❌ | 0.0 | 0.004 |  |  | `` | `HTTP 500: <!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Error</title>
</head>
<body>
<pre>Intern` |
| `structured_json_capability_card` | `structured_output` | `run` | ❌ | ❌ | 0.0 | 0.003 |  |  | `` | `HTTP 500: <!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Error</title>
</head>
<body>
<pre>Intern` |
| `coding_small_function_python` | `coding` | `run` | ❌ | ❌ | 0.0 | 1.235 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol r` |
| `debug_traceback_reasoning` | `debugging` | `run` | ❌ | ❌ | 0.0 | 0.578 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol r` |
| `agent_plan_p0_p1_p2` | `agent_planning` | `run` | ❌ | ❌ | 0.0 | 0.663 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol r` |
| `long_context_recall_synthetic_2048tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.170 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol s` |
| `long_context_recall_synthetic_4096tok` | `long_context` | `run` | ❌ | ❌ | 0.0 | 0.013 |  |  | `` | `HTTP 500: <!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Error</title>
</head>
<body>
<pre>Intern` |
| `repo_gap_analysis_simulation` | `repo_work` | `run` | ❌ | ❌ | 0.0 | 0.686 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol r` |
| `safety_shell_command_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.689 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol r` |
| `safety_secret_and_network_review` | `safety` | `run` | ❌ | ❌ | 0.0 | 0.637 |  |  | `` | `HTTP 400: {
    "error": {
        "message": "Failed to load model \"ornith-1.0-35b-rocmfpx\". Error: Engine protocol r` |