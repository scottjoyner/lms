from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from lms_agent_bench import benchmark_reliable


def args(**overrides):
    values = {
        "min_sample_completeness": 1.0,
        "min_success_rate": 0.98,
        "min_eval_success_rate": 0.90,
        "min_wilson_lower": 0.40,
        "max_trial_tps_cv": 0.20,
        "max_trial_ttft_cv": 0.35,
        "max_relative_mad": 0.25,
        "max_retry_rate": 0.25,
        "bootstrap_samples": 200,
        "seed": 20260802,
        "min_valid_trials": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def sample(
    *,
    trial: int,
    repeat: int = 1,
    ok: bool = True,
    eval_ok: bool = True,
    tps: float = 20.0,
    ttft: float = 0.4,
):
    return {
        "trial_index": trial,
        "phase": "run",
        "endpoint_id": "candidate-1",
        "model_key": "model.gguf",
        "case_key": "case-1",
        "repeat_index": repeat,
        "ok": ok,
        "eval_ok": eval_ok,
        "eval_score": 1.0 if eval_ok else 0.0,
        "tokens_per_sec": tps,
        "ttft_s": ttft,
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_robust_statistics_are_deterministic():
    values = [10.0, 11.0, 12.0, 13.0, 14.0]
    first = benchmark_reliable.bootstrap_median_ci(values, 42, 500)
    second = benchmark_reliable.bootstrap_median_ci(values, 42, 500)
    assert first == second
    assert first[0] <= 12.0 <= first[1]
    assert benchmark_reliable.percentile(values, 10) < 12.0
    assert benchmark_reliable.relative_mad(values) == 1.0 / 12.0


def test_trial_artifacts_require_exact_sample_matrix(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "config.json").write_text("{}", encoding="utf-8")
    (output_dir / "run_summary.csv").write_text("run_id\n1\n", encoding="utf-8")
    (output_dir / "task_summary.csv").write_text("run_id\n1\n", encoding="utf-8")
    write_csv(
        output_dir / "run_results.csv",
        [
            {
                "phase": "run",
                "endpoint_id": "candidate-1",
                "model_key": "model.gguf",
                "case_key": "case-1",
                "repeat_index": 1,
                "ok": True,
                "tokens_per_sec": 20,
                "output_file": "outputs/one.txt",
            }
        ],
    )
    expected = {
        ("candidate-1", "model.gguf", "case-1", 1),
        ("candidate-1", "model.gguf", "case-2", 1),
    }
    valid, errors, rows = benchmark_reliable.validate_trial_artifacts(
        output_dir, expected
    )
    assert valid is False
    assert rows
    assert any("missing 1 expected" in error for error in errors)


def test_stable_complete_measurements_pass_reliability_gate():
    rows = [
        sample(trial=1, tps=20.0, ttft=0.40),
        sample(trial=2, tps=20.5, ttft=0.41),
        sample(trial=3, tps=19.5, ttft=0.39),
    ]
    report = benchmark_reliable.aggregate_group(
        rows,
        valid_trials=3,
        required_trials=3,
        trial_attempts=3,
        expected_samples=3,
        preflight={"ok": True, "stable": True, "warmup_cv": 0.03},
        args=args(),
        seed_offset=0,
    )
    assert report["reliability_pass"] is True
    assert report["reliability_failures"] == []
    assert report["sample_completeness"] == "1.000000"
    assert float(report["trial_tps_cv"]) < 0.20
    assert float(report["success_wilson_lower_95"]) > 0.0


def test_unstable_retried_measurements_fail_reliability_gate():
    rows = [
        sample(trial=1, tps=5.0, ttft=0.2),
        sample(trial=2, tps=30.0, ttft=1.2),
        sample(trial=3, tps=8.0, ttft=0.3),
    ]
    report = benchmark_reliable.aggregate_group(
        rows,
        valid_trials=3,
        required_trials=3,
        trial_attempts=5,
        expected_samples=3,
        preflight={"ok": True, "stable": True, "warmup_cv": 0.02},
        args=args(),
        seed_offset=0,
    )
    assert report["reliability_pass"] is False
    failures = set(report["reliability_failures"])
    assert "trial_retry_rate_above_threshold" in failures
    assert "throughput_trial_variation_above_threshold" in failures


def test_preflight_requires_exact_model_identity(monkeypatch):
    monkeypatch.setattr(
        benchmark_reliable,
        "model_ids",
        lambda base_url, timeout_s, api_key: ["other-model"],
    )
    report = benchmark_reliable.preflight_endpoint(
        {
            "endpoint_id": "candidate-1",
            "base_url": "http://127.0.0.1:1234/v1",
            "model_key": "expected-model",
        },
        SimpleNamespace(
            preflight_timeout=1,
            warmup_runs=3,
            max_warmup_cv=0.5,
        ),
        None,
    )
    assert report["ok"] is False
    assert any("not exposed exactly" in error for error in report["errors"])


def fingerprint_args():
    return SimpleNamespace(
        timeout=10,
        repeats=1,
        max_context_tokens=4096,
        trials=3,
        min_valid_trials=3,
        max_trial_attempts=5,
        warmup_runs=3,
        preflight_timeout=10,
        trial_timeout=0,
        retry_backoff=1,
        cooldown_between_trials=1,
        seed=1,
        max_warmup_cv=0.5,
        min_sample_completeness=1.0,
        min_success_rate=0.98,
        min_eval_success_rate=0.9,
        min_wilson_lower=0.8,
        max_trial_tps_cv=0.2,
        max_trial_ttft_cv=0.35,
        max_relative_mad=0.25,
        max_retry_rate=0.25,
    )


def test_input_fingerprint_changes_with_suite_content(tmp_path):
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("endpoint_id,base_url,model_key\na,http://127.0.0.1,m\n")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"cases": [{"case_key": "one"}]}))
    first, _ = benchmark_reliable.input_fingerprint(
        fingerprint_args(),
        inventory,
        suite,
        ["one"],
    )
    suite.write_text(json.dumps({"cases": [{"case_key": "changed"}]}))
    second, _ = benchmark_reliable.input_fingerprint(
        fingerprint_args(),
        inventory,
        suite,
        ["changed"],
    )
    assert first != second
