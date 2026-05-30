# LMS Agent Benchmarking Toolkit

LMS is an agent-facing command line toolkit for profiling and benchmarking local or Tailscale-reachable LM Studio nodes. The goal is to let an agent test the machine it is running on, discover available models, run repeatable benchmark tasks, and produce a practical synopsis of what each model should or should not be trusted to do.

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

# Run the default profile + quick benchmark workflow.
lms quick

# Benchmark a Tailscale or LAN LM Studio node.
lms quick --endpoint http://100.64.0.10:1234/v1

# Limit a quick run to one or more known model IDs.
lms quick --models "qwen/qwen3-coder-30b,openai/gpt-oss-20b"

# Generate or regenerate recommendations from an existing run directory.
lms recommend runs/<run_id>
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
4. Run the current benchmark runner.
5. Generate `capability_matrix.csv` and `agent_recommendations.md`.

```bash
lms quick
lms quick --endpoint http://100.64.0.10:1234/v1 --max-models 2 --repeats 1
lms quick --profile-only
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
  machine_profile.json
  machine_synopsis.md
  lmstudio_inventory.csv
  run_results.csv
  run_summary.csv
  capability_matrix.csv
  agent_recommendations.md
  agent_skill_suite.v1.json
  sidecars/
```

### `machine_synopsis.md`

Human-friendly report:

- Machine identity and OS.
- CPU, RAM, GPU, VRAM, storage, network details.
- LM Studio endpoints discovered.
- Practical model recommendations.
- Known limitations and hardware warnings.

### `agent_recommendations.md`

Agent-facing operating guide:

- Default model candidate.
- Expected TTFT and tokens/sec.
- OK-rate evidence.
- Suggested routing behavior.
- Warnings for complex, long-context, or high-risk work.

### `capability_matrix.csv`

Normalized model recommendation rows:

```csv
run_id,host_name,host_ip,base_url,model_key,context_tokens,task_family,score,grade,latency_grade,throughput_grade,reliability_grade,recommended_use,avoid_use,evidence,notes
```

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
| machine profiler  | -----> | inventory collector  | -----> | benchmark scheduler  |
+-------------------+        +----------------------+        +----------------------+
          |                            |                              |
          v                            v                              v
+-------------------+        +----------------------+        +----------------------+
| hardware synopsis |        | endpoint/model CSV   |        | raw run artifacts    |
+-------------------+        +----------------------+        +----------------------+
                                                                    |
                                                                    v
                                                        +----------------------+
                                                        | evaluator/judge      |
                                                        +----------------------+
                                                                    |
                                                                    v
                                                        +----------------------+
                                                        | capability report    |
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

Run each core benchmark at increasing context sizes:

- 2k tokens.
- 4k tokens.
- 8k tokens.
- 16k tokens.
- 32k tokens.
- 64k tokens if supported.
- 128k tokens if supported.

Track quality degradation, latency, missed instructions, and recall accuracy.

### P2: Repository work simulation

- Read synthetic repository summary.
- Find implementation gaps.
- Generate patch plan.
- Produce test plan.
- Summarize risk.
- Generate next Codex prompt.

## Scoring model

Use layered scoring instead of a single score:

```text
overall_score = weighted mean(
  correctness,
  instruction_following,
  structured_output_validity,
  completeness,
  latency_fit,
  throughput_fit,
  stability,
  context_retention
)
```

Recommended grades:

- `A`: safe default for this task family.
- `B`: usable with review.
- `C`: acceptable for drafts only.
- `D`: not recommended except small/simple tasks.
- `F`: do not route this task family to this model on this hardware.

## Implementation plan

### P0.1 Normalize the artifact model

Create a `runs/<run_id>/` layout and write all benchmark outputs under that directory. Keep CSV as the primary interchange format, but add machine and recommendation Markdown reports.

### P0.2 Add machine profiling

Implemented in `lms_machine_profile.py` and exposed through `lms profile` / `lms quick`.

### P0.3 Add benchmark case manifest support

Benchmark cases are defined in `benchmarks/agent_skill_suite.v1.json`. The next implementation step is wiring this manifest directly into `benchmark_lmstudio_cross_machine_models.py`; the CLI already copies the manifest into each run directory.

### P0.4 Add deterministic auto-evaluators

Implemented in `lms_eval.py`. The next implementation step is applying these evaluators during benchmark execution and writing task-family scores into `capability_matrix.csv`.

### P1.1 Add context sweep mode

Each benchmark case should be runnable at multiple context lengths. The report should show the largest context where the model remains reliable.

### P1.2 Add agent recommendation synthesis

Initial implementation is exposed through `lms recommend` and automatically called by `lms quick`.

### P1.3 Add comparison reports

Add `compare_benchmark_runs.py` to compare two run directories and show routing changes, regressions, and improvements.

### P2 Add lightweight dashboard

A small static dashboard or FastAPI page can read the run artifacts and show:

- Model throughput rankings.
- Quality by task family.
- Context degradation curves.
- Hardware fit warnings.
- Agent routing recommendations.

## Design rules

- Do not require cloud services.
- Do not require internet access during benchmarks.
- Do not leak prompts, API keys, or private outputs outside the run directory.
- Keep local/Tailscale endpoints explicit.
- Treat LLM-as-judge as optional, not required.
- Prefer deterministic evaluation where possible.
- Make every recommendation traceable to raw benchmark evidence.
- Assume smaller models can draft, but stronger models should refine, judge, or handle long-context work.
