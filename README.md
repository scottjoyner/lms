# LMS Agent Benchmarking Toolkit

This repository is being refined from a local LM Studio benchmarking script collection into an **agent self-evaluation skill**. The goal is to let an agent inspect a machine, discover local or Tailscale-reachable LM Studio nodes, run repeatable benchmark tasks, evaluate output quality, and produce a practical synopsis of what the machine and model should or should not be trusted to do.

## Current baseline

The current repo direction already includes:

- LM Studio OpenAI-compatible endpoint benchmarking.
- Inventory-driven runs using CSV rows with host, endpoint, and model metadata.
- Cross-machine model comparison.
- Benchmark output as CSV, JSON config, Markdown sidecars, and captured full outputs.
- Basic operational metrics such as load time, TTFT, tokens/sec, OK rate, and optional quality scoring.

This README defines the next implementation cycle to turn those primitives into a full agent benchmarking product.

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

## Output contract

Every run should produce a stable run directory:

```text
runs/<run_id>/
  config.json
  machine_profile.json
  lmstudio_inventory.csv
  benchmark_results.csv
  benchmark_summary.csv
  capability_matrix.csv
  agent_recommendations.md
  machine_synopsis.md
  raw_outputs/
  sidecars/
```

### `machine_synopsis.md`

Human-friendly report:

- Machine identity and OS.
- CPU, RAM, GPU, VRAM, storage, network details.
- LM Studio endpoints discovered.
- Model loadability and throughput summary.
- Practical model recommendations.
- Known limitations and unsafe task classes.

### `agent_recommendations.md`

Agent-facing operating guide:

- Default model for coding.
- Default model for planning.
- Default model for summarization.
- Maximum recommended context window by model.
- Expected tokens/sec ranges.
- When to ask for clarification vs proceed.
- When to split tasks.
- When to route to another endpoint.
- When to refuse/flag low confidence.

### `capability_matrix.csv`

Suggested columns:

```csv
run_id,host_name,base_url,model_key,context_tokens,task_family,score,latency_grade,throughput_grade,reliability_grade,recommended_use,avoid_use,notes
```

## Benchmark suite v1

The first agent-centered suite should include these families:

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

### P2: Machine-specific routing recommendations

Convert benchmark results into routing rules:

```yaml
routing:
  code_review:
    preferred_model: <model_key>
    max_context_tokens: 16000
    fallback_model: <model_key>
  quick_shell_help:
    preferred_model: <small_fast_model>
    max_context_tokens: 4000
  long_repo_planning:
    preferred_model: <large_context_model>
    min_quality_score: 0.80
```

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

Deliverables:

- `machine_profile.json`
- `benchmark_results.csv`
- `benchmark_summary.csv`
- `capability_matrix.csv`
- `machine_synopsis.md`
- `agent_recommendations.md`

### P0.2 Add machine profiling

Add a standalone profiler script that runs before benchmarks.

Linux commands/data sources:

- `/etc/os-release`
- `uname -a`
- `lscpu --json`
- `free -b`
- `lsblk --json`
- `df -B1`
- `lspci`
- `nvidia-smi --query-gpu=... --format=csv,json` when available
- `rocm-smi --showall --json` when available
- `vainfo` or Vulkan info when useful

The profiler must degrade gracefully if a command is unavailable.

### P0.3 Add benchmark case manifest support

Move hard-coded cases into a versioned JSON/YAML manifest. The script should support:

```bash
python3 benchmark_lmstudio_cross_machine_models.py \
  --inventory-csv lmstudio_inventory.csv \
  --cases-file benchmarks/agent_skill_suite.v1.json \
  --output-dir runs
```

### P0.4 Add deterministic auto-evaluators

Before using LLM-as-judge, implement deterministic checks:

- JSON parse success.
- Schema validation.
- Required keyword/section presence.
- Code block detection.
- Command safety flags.
- Output length/truncation checks.
- Exact-answer checks for small retrieval tasks.

### P1.1 Add context sweep mode

Each benchmark case should be runnable at multiple context lengths. The report should show the largest context where the model remains reliable.

### P1.2 Add agent recommendation synthesis

Generate practical routing recommendations from measured results:

- Fast local draft model.
- Best coding model.
- Best long-context model.
- Best JSON/tool model.
- Models to avoid.
- Hardware bottlenecks.

### P1.3 Add comparison reports

Support compare runs:

```bash
python3 compare_benchmark_runs.py runs/2026-05-30T... runs/2026-05-31T...
```

Outputs:

- Regression/improvement table.
- Hardware differences.
- Model upgrades/downgrades.
- Routing rule changes.

### P2 Add lightweight dashboard

A small static dashboard or FastAPI page can read the run artifacts and show:

- Model throughput rankings.
- Quality by task family.
- Context degradation curves.
- Hardware fit warnings.
- Agent routing recommendations.

## Recommended next files to add

```text
benchmarks/agent_skill_suite.v1.json
schemas/capability_matrix.schema.json
schemas/machine_profile.schema.json
lms_machine_profile.py
lms_agent_benchmark.py
lms_report_writer.py
compare_benchmark_runs.py
```

## Design rules

- Do not require cloud services.
- Do not require internet access during benchmarks.
- Do not leak prompts, API keys, or private outputs outside the run directory.
- Keep local/Tailscale endpoints explicit.
- Treat LLM-as-judge as optional, not required.
- Prefer deterministic evaluation where possible.
- Make every recommendation traceable to raw benchmark evidence.
- Assume smaller models can draft, but stronger models should refine, judge, or handle long-context work.
