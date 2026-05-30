# LMS Design Review

This document reviews the current LMS design after the `0.3.0` CLI and benchmark runner updates. It focuses on whether the repo is now a usable agent skill rather than a loose collection of benchmark scripts.

## Current design state

LMS now has these layers:

```text
lms CLI (`lms_cli:main`)
  ├── doctor/probe/inventory/profile
  ├── quick workflow
  ├── runs/show/recommend/route
  └── eval wrapper

machine profiler
  ├── OS/CPU/RAM/storage/GPU probes
  ├── LM Studio endpoint probes
  └── machine_profile.json + machine_synopsis.md

manifest benchmark runner
  ├── inventory CSV input
  ├── benchmark suite JSON input
  ├── OpenAI-compatible chat completions
  ├── deterministic evaluator execution
  ├── run_results.csv
  ├── run_summary.csv
  ├── task_summary.csv
  └── sidecar Markdown/output files

evaluator layer
  ├── deterministic checks
  ├── manifest-driven evaluator specs
  └── eval_ok/eval_score persisted per result

recommendation layer
  ├── capability_matrix.csv
  ├── agent_recommendations.md
  └── routing_rules.yaml / routing_rules.json
```

## What is working well

### 1. Agent usability is now CLI-first

The default path remains simple and zero-config:

```bash
lms doctor
lms probe
lms quick
lms runs
lms show latest --task coding
lms route latest --task structured_output
```

This is the right product shape for agents because it avoids config-file setup for common paths.

### 2. Artifacts are self-contained

`lms quick` creates a run directory containing endpoint probes, profile output, inventory CSV, benchmark output, task summaries, recommendation output, routing rules, copied suite manifests, and raw model outputs.

### 3. The manifest is now the benchmark source of truth

`benchmark_lmstudio_cross_machine_models.py` accepts `--cases-file` and loads `benchmarks/agent_skill_suite.v1.json`. Manifest cases now define task family, priority, prompt/system, evaluator specs, recommendation signals, and optional synthetic long-context sweeps.

### 4. Deterministic evaluators run during benchmarks

`lms_eval.evaluate_output()` is now applied to each run case. Results are written into `run_results.csv` as:

```csv
eval_ok,eval_score,eval_failed_json,eval_result_json
```

This makes benchmark quality measurable without relying on LLM-as-judge.

### 5. Routing is task-aware

The runner emits `task_summary.csv`, and recommendation synthesis now prefers task-family rows when available. Agents can ask for a route by task:

```bash
lms route latest --task coding
lms route latest --task long_context --json
```

## Remaining design gaps

### Gap 1: Run comparison is still missing

Agents need to know whether a machine improved or regressed after model changes, LM Studio updates, quantization changes, driver updates, or hardware changes.

Required refinement:

- Add `lms compare runs/a runs/b`.
- Compare:
  - model availability
  - OK rate
  - evaluator OK rate
  - evaluator score
  - TTFT
  - tokens/sec
  - context reliability
  - selected routes

Target outputs:

```text
compare_summary.md
compare_delta.csv
```

### Gap 2: Safety classification is not yet implemented

The agent benchmark should identify when local models produce dangerous shell commands, destructive actions, hardcoded secrets, unsafe network bindings, or risky permission changes.

Required refinement:

- Add deterministic safety evaluators:
  - `forbidden_shell_patterns`
  - `requires_confirmation_for_destructive_command`
  - `secret_like_token_detected`
  - `unsafe_network_binding_detected`
  - `dangerous_permission_change_detected`

### Gap 3: Hardware-to-model fit is heuristic only

The profiler captures useful hardware signals, but recommendation logic does not yet connect RAM/VRAM/runtime availability to model loading behavior in a detailed way.

Required refinement:

- Parse model names for likely quantization / parameter class.
- Track load success/failure by model.
- Estimate model memory class from name when exact metadata is unavailable.
- Warn when model class is likely larger than available VRAM/RAM.

### Gap 4: Package assets need stronger non-editable install support

The active CLI includes path fallbacks, but non-editable installs should use more robust package resource lookup.

Required refinement:

- Move to a package layout such as `src/lms_agent_bench/`.
- Use `importlib.resources` for bundled `benchmarks/` and `schemas/`.
- Preserve direct script execution for local hacking.

### Gap 5: Fallback route selection is basic

Routing can pick the best model per task family, but fallback model selection is not yet explicit in `routing_rules.yaml`.

Required refinement:

- Export preferred model and fallback model per task family.
- Include minimum score thresholds.
- Include maximum reliable context once context reliability is aggregated.

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

### Gap 6: Long-context reliability is measured but not fully summarized

The runner can generate synthetic long-context benchmark cases and emit `context_tokens`, but recommendations do not yet summarize the largest reliable context by model/task.

Required refinement:

- Aggregate `(model_key, task_family, context_tokens)` reliability.
- Add `max_reliable_context_tokens` to `capability_matrix.csv`.
- Add route warnings when requested work exceeds measured context reliability.

## Recommended next implementation cycle

### P1.3 Add run comparison

Deliverables:

- `lms compare <run_a> <run_b>`.
- Markdown and CSV delta output.
- Detect routing changes by task family.

Acceptance criteria:

- Agents can tell whether a new model or hardware change improved or regressed benchmark performance.
- Comparison output identifies changed best routes.

### P1.4 Add safety evaluators

Deliverables:

- Shell/destructive/secrets/network safety deterministic evaluators.
- Safety benchmark cases in `agent_skill_suite.v1.json`.
- Safety task family rows in `task_summary.csv` and `capability_matrix.csv`.

### P1.5 Add fallback model export

Deliverables:

- Route export includes preferred and fallback model per task family.
- Fallback model must be a different model or endpoint when available.
- Route evidence includes both scores.

### P2.1 Package layout cleanup

Deliverables:

- Move modules into `src/lms_agent_bench/`.
- Use `importlib.resources` for benchmarks and schemas.
- Keep compatibility wrappers for old script paths.

### P2.2 Static dashboard

Deliverables:

- Simple static HTML or FastAPI dashboard over run artifacts.
- Show model rankings, task-family quality, context reliability, and routing rules.

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

The project is now past the initial design gap: the manifest and deterministic evaluator are integrated into the benchmark runner. The next highest-value improvement is `lms compare`, followed by safety evaluators and richer fallback routing.
