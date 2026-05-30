# LMS Design Review

This document reviews the current LMS design after adding the agent-facing CLI. It focuses on whether the repo is becoming a usable agent skill rather than a loose collection of benchmark scripts.

## Current design state

LMS now has these layers:

```text
lms CLI
  ├── doctor/probe/inventory/profile
  ├── quick workflow
  ├── runs/show/recommend/route
  └── eval wrapper

machine profiler
  ├── OS/CPU/RAM/storage/GPU probes
  ├── LM Studio endpoint probes
  └── machine_profile.json + machine_synopsis.md

benchmark runner
  ├── inventory CSV input
  ├── OpenAI-compatible chat completions
  ├── run_results.csv
  ├── run_summary.csv
  └── sidecar Markdown/output files

evaluator layer
  ├── deterministic checks
  └── future benchmark-run integration

recommendation layer
  ├── capability_matrix.csv
  ├── agent_recommendations.md
  └── routing_rules.yaml / routing_rules.json
```

## What is working well

### 1. Agent usability is improving

The CLI exposes simple commands:

```bash
lms doctor
lms probe
lms quick
lms runs
lms show latest
lms route latest --task general
```

This is the right product shape for agents because it avoids config-file setup for common paths.

### 2. Artifacts are becoming self-contained

`lms quick` now creates a run directory containing probe output, profile output, inventory CSV, benchmark output, recommendation output, and copied suite manifests.

### 3. Routing is becoming explicit

The `route` command gives agents a concrete routing rule instead of requiring them to parse a Markdown report manually.

### 4. Evaluators are local and deterministic

`lms_eval.py` keeps first-pass scoring cheap, explainable, and reproducible. This is important because LLM-as-judge should remain optional.

## Main design gaps

### Gap 1: The benchmark runner still owns hard-coded cases

The CLI copies `benchmarks/agent_skill_suite.v1.json` into the run directory, but `benchmark_lmstudio_cross_machine_models.py` still runs its internal default cases.

Impact:

- The manifest is not yet the source of truth.
- Deterministic evaluator specs in the manifest are not applied during benchmark execution.
- `capability_matrix.csv` only has general performance routing, not task-family capability scoring.

Required refinement:

- Add `--cases-file` to the benchmark runner.
- Convert manifest cases into `BenchCase` objects.
- Apply `lms_eval.evaluate_output()` to each run result.
- Add evaluator fields to `run_results.csv`.

Target fields:

```csv
eval_ok,eval_score,eval_failed_json,task_family,priority,recommendation_signal,context_tokens
```

### Gap 2: Context sweep exists in design but not execution

The suite manifest describes context sweep tokens, but the benchmark runner does not yet synthesize filler contexts or run cases at different lengths.

Required refinement:

- Add synthetic context generation.
- Support `prompt_template` and `synthetic_context` case fields.
- Emit `context_tokens` in result and summary rows.
- Report largest reliable context per model.

### Gap 3: Hardware-to-model fit is heuristic only

The profiler captures useful hardware signals, but recommendation logic does not yet connect RAM/VRAM/runtime availability to model loading behavior.

Required refinement:

- Parse model names for likely quantization / parameter class.
- Track load success/failure by model.
- Estimate model memory class from name when exact metadata is unavailable.
- Warn when model class is likely larger than available VRAM/RAM.

### Gap 4: The CLI is still script-local rather than package-resource aware

The CLI now includes path fallbacks, but package data lookup should be more robust after non-editable installs.

Required refinement:

- Convert repo into a real package layout or use `importlib.resources` for `benchmarks/` and `schemas/`.
- Keep direct script execution supported for local hacking.

### Gap 5: Recommendations need task-specific routing

Current routing is `general` because benchmark summaries do not yet include per-family evaluator results.

Required refinement:

- Aggregate scores by `(model_key, task_family, context_tokens)`.
- Export routing rules for at least:
  - `general`
  - `coding`
  - `debugging`
  - `agent_planning`
  - `structured_output`
  - `long_context`
  - `repo_work`

Target YAML:

```yaml
routing:
  coding:
    preferred_model: "..."
    fallback_model: "..."
    base_url: "..."
    max_context_tokens: 16000
    min_score: 0.75
    evidence: "..."
```

### Gap 6: No run comparison yet

Agents need to know whether a machine improved or regressed after model changes, LM Studio updates, driver changes, or hardware changes.

Required refinement:

- Add `lms compare runs/a runs/b`.
- Compare:
  - model availability
  - OK rate
  - TTFT
  - tokens/sec
  - evaluator scores
  - context reliability
  - selected routes

### Gap 7: No safety classification for command/code outputs

The agent benchmark should identify when local models produce dangerous shell commands, destructive actions, hardcoded secrets, or unsafe network assumptions.

Required refinement:

- Add deterministic safety evaluators:
  - `forbidden_shell_patterns`
  - `requires_confirmation_for_destructive_command`
  - `secret_like_token_detected`
  - `unsafe_network_binding_detected`
  - `dangerous_permission_change_detected`

## Recommended next implementation cycle

### P0.5 Wire manifest cases into benchmark runner

Deliverables:

- `--cases-file` argument.
- Manifest loader.
- Support case fields:
  - `case_key`
  - `priority`
  - `task_family`
  - `system`
  - `prompt`
  - `prompt_template`
  - `temperature`
  - `max_output_tokens`
  - `evaluators`
  - `recommendation_signal`
- Add evaluator output columns.

Acceptance criteria:

- `lms quick` uses `benchmarks/agent_skill_suite.v1.json` by default.
- `run_results.csv` includes evaluator scores.
- `capability_matrix.csv` includes task-family rows.

### P0.6 Add run inspection polish

Deliverables:

- `lms show latest` compact summary.
- `lms route latest --task coding` route selection.
- `lms route latest --write` writes YAML/JSON.

Status:

- Initial implementation added.
- Needs testing against real run artifacts.

### P1.1 Context sweep execution

Deliverables:

- Synthetic context builder.
- Long-context case support.
- Largest reliable context by task/model.

### P1.2 Task-specific recommendations

Deliverables:

- Aggregate evaluator scores by task family.
- Generate task-specific `routing_rules.yaml`.
- Include fallback models.

### P1.3 Compare runs

Deliverables:

- `lms compare <run_a> <run_b>`.
- Markdown and CSV delta output.

## Design rules to preserve

- Default path must remain `lms quick`.
- The CLI must remain useful with zero config files.
- Every output must live under a run directory.
- Raw model outputs must remain available for human review.
- No cloud dependency.
- LLM-as-judge must be optional.
- Deterministic checks must run first.
- Routing recommendations must cite measurable evidence.

## Final assessment

The project direction is sound. The biggest missing piece is not more CLI polish; it is connecting the manifest and deterministic evaluator into the benchmark runner so that the system measures real agent skills instead of only general throughput and basic success rate.
