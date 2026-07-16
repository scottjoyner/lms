# LMS Agent Benchmarking Toolkit

LMS is an agent-facing command line toolkit for profiling and benchmarking local or Tailscale-reachable LM Studio nodes. It gives agents a simple way to test the machine they are running on, discover available local models, run repeatable benchmark tasks, score outputs with deterministic evaluators, and produce task-specific routing recommendations.

Current installed CLI entrypoint:

```toml
lms = "lms_cli:main"
```

## Fast start for agents

Install locally from the repository:

```bash
python3 -m pip install -e .
```

Then use the simple CLI:

```bash
# Check local scripts and the default LM Studio endpoint.
lms doctor

# List models from local LM Studio server mode.
lms probe

# Run profile + manifest benchmark + recommendations.
lms quick

# Benchmark a Tailscale or LAN LM Studio node.
lms quick --endpoint http://100.64.0.10:1234/v1

# Limit a quick run to one or more known model IDs.
lms quick --models "qwen/qwen3-coder-30b,openai/gpt-oss-20b"

# Inspect the latest run and select a task-specific route.
lms show latest --task coding
lms route latest --task structured_output
lms route latest --task long_context --json
```

No config file is required. By default the CLI uses `http://127.0.0.1:1234/v1`, or `LMS_BASE_URL` / `LMSTUDIO_BASE_URL` when set.

## CLI commands

### `lms doctor`

Checks that the local LMS scripts exist and probes endpoint reachability.

```bash
lms doctor
lms doctor --endpoint http://127.0.0.1:1234/v1
```

### `lms probe`

Lists models available from one or more LM Studio OpenAI-compatible endpoints.

```bash
lms probe
lms probe --endpoint http://100.64.0.10:1234/v1 --endpoint http://100.64.0.11:1234/v1
lms probe --json
```

For tailnet-backed fleets, keep the registry fresh first:

```bash
lms-bench-endpoints discover-tailscale
lms-bench quick --from-registry --discover-tailscale
LMS_DISCOVER_TAILSCALE=1 lms-bench quick --from-registry
```

### `lms inventory`

Creates the inventory CSV expected by the benchmark runner.

```bash
lms inventory --out lmstudio_inventory.csv
lms inventory --endpoint http://100.64.0.10:1234/v1 --max-models 3
```

The CSV columns are:

```csv
host_name,host_ip,endpoint_id,base_url,reachable,model_id,model_key
```

### `lms profile`

Collects machine and endpoint information without running a full benchmark.

```bash
lms profile --output-dir runs/profile-local
lms profile --endpoint http://100.64.0.10:1234/v1 --output-dir runs/profile-xwing
```

Outputs:

```text
machine_profile.json
machine_synopsis.md
```

### `lms quick`

Runs the default agent workflow:

1. Probe LM Studio endpoint(s).
2. Write `lmstudio_inventory.csv`.
3. Collect `machine_profile.json` and `machine_synopsis.md`.
4. Run the manifest-aware benchmark runner with `benchmarks/agent_skill_suite.v1.json`.
5. Apply deterministic evaluators during benchmark execution.
6. Generate `run_results.csv`, `run_summary.csv`, and `task_summary.csv`.
7. Generate `capability_matrix.csv`, `agent_recommendations.md`, and routing rules.

```bash
lms quick
lms quick --endpoint http://100.64.0.10:1234/v1 --max-models 2 --repeats 1
lms quick --max-context-tokens 16384
lms quick --suite-file benchmarks/agent_skill_suite.v1.json
lms quick --profile-only
```

### `lms runs`

Lists known run directories.

```bash
lms runs
lms runs --runs-dir runs --limit 10
```

### `lms show`

Shows a compact run summary, including task-family scores when `task_summary.csv` exists.

```bash
lms show latest
lms show latest --task coding
lms show runs/20260530T120000Z --task long_context
```

### `lms route`

Prints the best route for an agent task from `capability_matrix.csv`. It uses task-specific rows when available and falls back to general rows when needed.

```bash
lms route latest --task coding
lms route latest --task structured_output
lms route latest --task long_context --json
lms route latest --task repo_work --write
```

`--write` emits:

```text
routing_rules.yaml
routing_rules.json
```

### `lms recommend`

Regenerates recommendations from an existing run directory.

```bash
lms recommend latest
lms recommend runs/<run_id>
```

### `lms eval`

Runs deterministic evaluators against a model output file or stdin.

```bash
echo '{"status":"ok"}' | lms eval --evaluators-json '[{"type":"json_parse"}]' --pretty
lms eval --output-file raw.txt --evaluators-file evals.json
```

## Output contract

Every `lms quick` run creates a self-contained run directory:

```text
runs/<run_id>/
  lms_run_config.json
  endpoint_probes.json
  machine_profile.json
  machine_synopsis.md
  lmstudio_inventory.csv
  config.json
  run_results.csv
  run_summary.csv
  task_summary.csv
  capability_matrix.csv
  agent_recommendations.md
  routing_rules.yaml
  routing_rules.json
  agent_skill_suite.v1.json
  sidecars/
    run_<epoch>/
      INDEX.md
      MODEL__<host>__<model>.md
      outputs/
```

### `run_results.csv`

Per-model, per-case, per-repeat benchmark rows. Important columns:

```csv
run_id,created_at_utc,phase,host_name,host_ip,endpoint_id,base_url,model_id,model_key,case_key,task_family,priority,context_tokens,recommendation_signal,repeat_index,ok,http_status,error,wall_s,ttft_s,load_s,prompt_tokens,completion_tokens,total_tokens,tokens_per_sec,finish_reason,eval_ok,eval_score,eval_failed_json,eval_result_json,output_file
```

### `run_summary.csv`

Per-model aggregate summary across all task families:

```csv
run_id,host_name,host_ip,base_url,model_key,load_s,ttft_med,tps_med,ok_rate,eval_ok_rate,eval_score_avg,cases
```

### `task_summary.csv`

Per-model, per-task-family aggregate summary:

```csv
run_id,host_name,host_ip,base_url,model_key,task_family,load_s,ttft_med,tps_med,ok_rate,eval_ok_rate,eval_score_avg,cases
```

### `capability_matrix.csv`

Normalized routing recommendation rows. When `task_summary.csv` exists, rows are task-specific.

```csv
run_id,host_name,host_ip,base_url,model_key,context_tokens,task_family,score,grade,latency_grade,throughput_grade,reliability_grade,recommended_use,avoid_use,evidence,notes
```

### `machine_synopsis.md`

Human-friendly machine report:

- Machine identity and OS.
- CPU, RAM, GPU, VRAM, storage, network details.
- LM Studio endpoints discovered.
- Practical machine recommendations.
- Known limitations and hardware warnings.

### `agent_recommendations.md`

Agent-facing operating guide:

- Task-specific routing candidates.
- Expected TTFT and tokens/sec.
- OK-rate and evaluator-score evidence.
- Suggested routing behavior.
- Warnings for complex, long-context, or high-risk work.

## Benchmark manifest

The default manifest is:

```text
benchmarks/agent_skill_suite.v1.json
```

The benchmark runner accepts it directly:

```bash
python3 benchmark_lmstudio_cross_machine_models.py \
  --inventory-csv runs/<run_id>/lmstudio_inventory.csv \
  --cases-file benchmarks/agent_skill_suite.v1.json \
  --output-dir runs/<run_id> \
  --sidecar-dir runs/<run_id>/sidecars \
  --max-context-tokens 8192
```

Supported manifest case fields include:

```json
{
  "case_key": "structured_json_capability_card",
  "priority": "P0",
  "task_family": "structured_output",
  "system": "You produce strict JSON only. Do not use markdown.",
  "prompt": "Return a JSON object...",
  "temperature": 0.0,
  "max_output_tokens": 256,
  "evaluators": [
    {"type": "json_parse"},
    {"type": "json_required_keys", "value": ["task_fit", "confidence"]}
  ],
  "recommendation_signal": "json_tool_call_reliability"
}
```

Long-context cases can use `prompt_template`, `synthetic_context`, and `context_sweep_tokens`. The runner creates deterministic synthetic filler context and caps context sweep runs with `--max-context-tokens`.

## Deterministic evaluators

Evaluators run during benchmark execution through `lms_eval.evaluate_output()`. Supported evaluator types include:

- `exact_contains`
- `contains_all`
- `max_chars`
- `min_chars`
- `json_parse`
- `json_required_keys`
- `json_forbidden_extra_keys`
- `no_markdown_fence`
- `regex_contains`
- `regex_not_contains`

The evaluator result is stored in `run_results.csv` as `eval_ok`, `eval_score`, `eval_failed_json`, and `eval_result_json`.

## Product objective

The tool should answer these questions for every reachable LM Studio node:

1. **What hardware is available?**
   - CPU model, cores, threads, RAM, swap, disk, OS, kernel, GPU inventory, VRAM, driver/runtime availability, thermal and power limits when available.

2. **What models are available?**
   - Model ID, family, quantization, apparent parameter class, context length, endpoint URL, host, reachability, load behavior, and memory fit.

3. **What can this model reliably do on this machine?**
   - Coding, debugging, planning, summarization, extraction, JSON/tool formatting, long-context retrieval, multi-step reasoning, refactor review, command generation, and agent planning.

4. **Where are the limits?**
   - Context length degradation, hallucination risk, malformed JSON/tool calls, low throughput, unstable load, out-of-memory failure, poor instruction following, slow TTFT, low quality at long context, or unsuitable hardware.

5. **What should an agent do with this result?**
   - Recommended model routing, safe task classes, max input length, max output length, expected latency, fallback behavior, and warnings for tasks that should be routed to stronger models.

## Target architecture

```text
+-------------------+        +----------------------+        +----------------------+
| machine profiler  | -----> | inventory collector  | -----> | manifest benchmark   |
+-------------------+        +----------------------+        +----------------------+
          |                            |                              |
          v                            v                              v
+-------------------+        +----------------------+        +----------------------+
| hardware synopsis |        | endpoint/model CSV   |        | raw run artifacts    |
+-------------------+        +----------------------+        +----------------------+
                                                                    |
                                                                    v
                                                        +----------------------+
                                                        | deterministic evals  |
                                                        +----------------------+
                                                                    |
                                                                    v
                                                        +----------------------+
                                                        | routing/capabilities |
                                                        +----------------------+
```

## Benchmark suite v1

The first agent-centered suite includes these families:

### P0: Operational health

- Endpoint reachability.
- Model list retrieval.
- Cold/warm load timing.
- TTFT.
- Sustained tokens/sec.
- Error rate.
- Output truncation detection.

### P0: Structured output and tool discipline

- Valid JSON generation.
- JSON schema adherence.
- Tool-call argument formatting.
- Refusal to invent missing arguments.
- Recovery from malformed user input.

### P1: Coding capability

- Small function implementation.
- Multi-file patch planning.
- Bug diagnosis from traceback.
- Refactor plan with acceptance criteria.
- Unit test generation.
- Security review of a small snippet.

### P1: Agent planning

- Convert a product request into P0/P1/P2 implementation tasks.
- Identify missing requirements without stalling.
- Produce acceptance criteria.
- Detect unsafe or impossible requirements.
- Estimate task fit for local vs larger model.

### P1: Long-context behavior

Run supported long-context cases at increasing context sizes:

- 2k tokens.
- 4k tokens.
- 8k tokens.
- 16k tokens.
- 32k tokens.
- 64k tokens if supported.

The default CLI cap is currently 8k tokens for quick local runs:

```bash
lms quick --max-context-tokens 8192
```

### P2: Repository work simulation

- Read synthetic repository summary.
- Find implementation gaps.
- Generate patch plan.
- Produce test plan.
- Summarize risk.
- Generate next Codex prompt.

## Scoring model

The current route score combines runtime reliability, deterministic evaluator quality, throughput, and TTFT:

```text
route_score =
  ok_rate * 0.35 +
  max(eval_score_avg, eval_ok_rate) * 0.45 +
  min(tokens_per_sec / 40, 1.0) * 0.15 +
  low_ttft_bonus * 0.05
```

Recommended grades:

- `A`: safe default for this task family.
- `B`: usable with review.
- `C`: acceptable for drafts only.
- `D`: not recommended except small/simple tasks.
- `F`: do not route this task family to this model on this hardware.

## Implementation status

### Implemented

- `lms_cli.py` active CLI entrypoint.
- `lms doctor`, `probe`, `inventory`, `profile`, `quick`, `runs`, `show`, `route`, `recommend`, and `eval`.
- `lms_machine_profile.py` machine profiling.
- Manifest-driven benchmark execution through `--cases-file`.
- Deterministic evaluator execution during benchmark runs.
- `run_results.csv`, `run_summary.csv`, and `task_summary.csv`.
- Task-specific capability matrix generation.
- `routing_rules.yaml` and `routing_rules.json` export.
- Synthetic long-context prompt generation with `--max-context-tokens` cap.

### Next refinements

- Add `lms compare <run_a> <run_b>` to compare model, driver, quantization, and hardware changes.
- Add safety-focused evaluators for shell commands, secrets, destructive operations, and unsafe bindings.
- Move to a real package layout or use `importlib.resources` for package assets in non-editable installs.
- Add optional static dashboard over run artifacts.
- Add fallback-model selection per task family.

## Design rules

- Do not require cloud services.
- Do not require internet access during benchmarks.
- Do not leak prompts, API keys, or private outputs outside the run directory.
- Keep local/Tailscale endpoints explicit.
- Treat LLM-as-judge as optional, not required.
- Prefer deterministic evaluation where possible.
- Make every recommendation traceable to raw benchmark evidence.
- Assume smaller models can draft, but stronger models should refine, judge, or handle long-context work.
