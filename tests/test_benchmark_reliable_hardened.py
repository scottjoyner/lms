from __future__ import annotations

import csv
from pathlib import Path

from lms_agent_bench import benchmark_reliable as base
from lms_agent_bench import benchmark_reliable_hardened as hardened


def write_csv(path: Path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trial_fixture(tmp_path: Path):
    trial_root = tmp_path / "trials" / "trial_001"
    attempt_dir = trial_root / "attempt_001"
    output_dir = attempt_dir / "output"
    sidecar_dir = attempt_dir / "sidecars"
    run_dir = sidecar_dir / "run_123"
    output_dir.mkdir(parents=True)
    (run_dir / "outputs").mkdir(parents=True)
    rows = [
        {
            "phase": "run",
            "endpoint_id": "candidate-1",
            "model_key": "model.gguf",
            "case_key": "case-1",
            "repeat_index": 1,
            "ok": True,
            "tokens_per_sec": 20,
            "output_file": "outputs/sample.txt",
        }
    ]
    base.write_json(
        output_dir / "config.json",
        {
            "run_id": 123,
            "sidecar_dir": str(sidecar_dir.resolve()),
        },
    )
    write_csv(output_dir / "run_results.csv", rows)
    (output_dir / "run_summary.csv").write_text(
        "run_id\n123\n", encoding="utf-8"
    )
    (output_dir / "task_summary.csv").write_text(
        "run_id\n123\n", encoding="utf-8"
    )
    sample = run_dir / "outputs" / "sample.txt"
    sample.write_text("READY\n", encoding="utf-8")
    log = attempt_dir / "runner.log"
    log.write_text("complete\n", encoding="utf-8")
    input_fp = "sha256:" + "1" * 64
    manifest = {
        "schema_version": "reliable_benchmark_trial.v1",
        "trial_index": 1,
        "attempt_index": 1,
        "input_fingerprint": input_fp,
        "returncode": 0,
        "timed_out": False,
        "valid": True,
        "postflight": {"candidate-1": {"ok": True}},
        "artifacts": base.artifact_hashes(output_dir),
        "runner_log_sha256": base.file_sha256(log),
    }
    return trial_root, attempt_dir, output_dir, sample, rows, manifest, input_fp


def test_validation_requires_existing_nonempty_raw_output(tmp_path):
    _, _, output_dir, sample, _, _, _ = trial_fixture(tmp_path)
    expected = {("candidate-1", "model.gguf", "case-1", 1)}
    valid, errors, _ = hardened.validate_trial_artifacts(output_dir, expected)
    assert valid is True
    assert errors == []

    sample.unlink()
    valid, errors, _ = hardened.validate_trial_artifacts(output_dir, expected)
    assert valid is False
    assert any("output is missing" in error for error in errors)


def test_successful_outputs_must_be_unique_and_nonempty(tmp_path):
    _, _, output_dir, sample, rows, _, _ = trial_fixture(tmp_path)
    sample.write_text("", encoding="utf-8")
    try:
        hardened.sample_output_artifacts(output_dir, rows)
    except ValueError as exc:
        assert "output is empty" in str(exc)
    else:
        raise AssertionError("empty successful output was accepted")


def test_sealed_valid_trial_can_resume(tmp_path, monkeypatch):
    trial_root, attempt_dir, output_dir, _, rows, manifest, input_fp = trial_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        hardened,
        "_ORIGINAL_EXECUTE_TRIAL_ATTEMPT",
        lambda *args, **kwargs: {
            "trial_index": 1,
            "attempt_index": 1,
            "manifest": manifest,
            "rows": rows,
            "output_dir": str(output_dir),
            "resumed": False,
        },
    )
    result = hardened.execute_trial_attempt()
    assert result["manifest"]["trial_manifest_fingerprint"].startswith(
        "sha256:"
    )
    assert result["manifest"]["sample_outputs"][0]["size_bytes"] > 0

    resumed = hardened.existing_valid_trial(trial_root, input_fp)
    assert resumed is not None
    assert resumed["resumed"] is True
    assert resumed["trial_index"] == 1
    assert (attempt_dir / "trial_manifest.json").is_file()


def test_resume_rejects_tampered_sample_or_manifest(tmp_path, monkeypatch):
    trial_root, attempt_dir, output_dir, sample, rows, manifest, input_fp = trial_fixture(
        tmp_path
    )
    monkeypatch.setattr(
        hardened,
        "_ORIGINAL_EXECUTE_TRIAL_ATTEMPT",
        lambda *args, **kwargs: {
            "trial_index": 1,
            "attempt_index": 1,
            "manifest": manifest,
            "rows": rows,
            "output_dir": str(output_dir),
            "resumed": False,
        },
    )
    hardened.execute_trial_attempt()
    sample.write_text("tampered\n", encoding="utf-8")
    assert hardened.existing_valid_trial(trial_root, input_fp) is None

    sample.write_text("READY\n", encoding="utf-8")
    manifest_path = attempt_dir / "trial_manifest.json"
    sealed = base.read_json(manifest_path)
    sealed["returncode"] = 1
    base.write_json(manifest_path, sealed)
    assert hardened.existing_valid_trial(trial_root, input_fp) is None


def test_resume_rejects_unsealed_legacy_manifest(tmp_path):
    trial_root, attempt_dir, _, _, _, manifest, input_fp = trial_fixture(tmp_path)
    base.write_json(attempt_dir / "trial_manifest.json", manifest)
    assert hardened.existing_valid_trial(trial_root, input_fp) is None
