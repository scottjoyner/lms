import hashlib
import io
import json
import tarfile
from pathlib import Path

from lms_agent_bench import fleet_gate_entrypoint
from lms_agent_bench.fleet_loadout import canonical_hash

OBSERVATION_FP = "sha256:" + "1" * 64
PLAN_FP = "sha256:" + "2" * 64
EXECUTION_FP = "sha256:" + "3" * 64
SELECTION_FP = "sha256:" + "4" * 64
MODEL_FP = "sha256:" + "5" * 64
COMMIT = "a" * 40
RUN_ID = "run-1"


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def source_artifact(tampered=False):
    core = {
        "node_id": "x1-370",
        "run_id": RUN_ID,
        "expected_branch": "full-auto-reconciliation-20260730",
        "actual_branch": "full-auto-reconciliation-20260730",
        "expected_commit": COMMIT,
        "actual_commit": COMMIT,
        "dirty": False,
        "origin_fingerprint": "sha256:" + "7" * 64,
        "python_version": "3.12.0",
        "package_version": "0.24.0",
    }
    artifact = {
        "schema_version": "fleet_source_control.v1",
        "artifact_type": "source_control_provenance",
        "captured_at_utc": "2026-08-01T00:00:00+00:00",
        **core,
        "source_fingerprint": canonical_hash(core),
        "admission": {"admitted": False},
    }
    if tampered:
        artifact["package_version"] = "tampered"
    return artifact


def reliability_summary():
    return {
        "host_name": "x1-370",
        "model_key": "model.gguf",
        "reliability_pass": True,
        "reliability_score": "0.980000",
        "valid_trials": 3,
        "required_trials": 3,
        "trial_attempts": 3,
        "trial_retry_rate": "0.000000",
        "sample_completeness": "1.000000",
        "success_wilson_lower_95": "0.900000",
        "trial_tps_cv": "0.030000",
        "trial_ttft_cv": "0.040000",
        "tps_relative_mad": "0.020000",
        "ttft_relative_mad": "0.020000",
        "tps_p10": "19.000",
        "tps_p90": "21.000",
        "ttft_p90": "0.450",
        "tps_median_ci95_low": "19.500",
        "tps_median_ci95_high": "20.500",
        "ttft_median_ci95_low": "0.390",
        "ttft_median_ci95_high": "0.410",
        "warmup_cv": "0.020000",
        "warmup_stable": True,
        "reliability_failures": [],
    }


def reliability_artifact(tampered=False):
    core = {
        "schema_version": "reliable_benchmark.v1",
        "artifact_type": "benchmark_reliability",
        "input_fingerprint": "sha256:" + "8" * 64,
        "requested_trials": 3,
        "valid_trials": 3,
        "trial_attempts": 3,
        "passed": True,
        "summaries": [reliability_summary()],
        "admission": {"admitted": False},
    }
    artifact = {
        **core,
        "created_at_utc": "2026-08-01T00:30:00+00:00",
        "reliability_fingerprint": canonical_hash(core),
    }
    if tampered:
        artifact["trial_attempts"] = 4
    return artifact


def selected_metrics(reliability):
    summary = reliability["summaries"][0]
    metrics = {
        "ok_rate": "1.0",
        "eval_ok_rate": "1.0",
        "eval_score_avg": "1.0",
        "tps_med": "20.0",
        "ttft_med": "0.4",
        "memory_headroom_ratio": "0.25",
        "concurrency_ok": "true",
        "streaming_ok": "true",
        "crash_count": "0",
        "benchmark_exit_code": "0",
        "reliability_pass": True,
        "reliability_fingerprint": reliability["reliability_fingerprint"],
    }
    for field in fleet_gate_entrypoint._RELIABILITY_METRICS:
        metrics[field] = summary[field]
    return metrics


def bundle_files(
    mode="observe",
    tampered_source=False,
    tampered_reliability=False,
):
    files = {
        "source_control.json": encoded(source_artifact(tampered_source)),
        "machine_observation.json": encoded(
            {
                "observation_fingerprint": OBSERVATION_FP,
                "identity": {"hostname": "x1-370"},
                "hardware": {"memory": {"total_bytes": 1}},
            }
        ),
        "model_inventory.json": encoded(
            {
                "models": [
                    {
                        "id": "model.gguf",
                        "path": "/models/model.gguf",
                    },
                    {
                        "id": "other.gguf",
                        "path": "/models/other.gguf",
                    },
                ]
            }
        ),
        "benchmark_plan.json": encoded(
            {
                "observation_fingerprint": OBSERVATION_FP,
                "plan_fingerprint": PLAN_FP,
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "model": {"id": "model.gguf"},
                    }
                ],
                "rejected_candidates": [],
            }
        ),
    }
    if mode == "sweep":
        reliability = reliability_artifact(tampered=tampered_reliability)
        files.update(
            {
                "benchmark/execution_manifest.json": encoded(
                    {
                        "plan_fingerprint": PLAN_FP,
                        "execution_fingerprint": EXECUTION_FP,
                        "candidate_ids": ["candidate-1"],
                        "loopback_only": True,
                    }
                ),
                "benchmark/candidate-1/suite/reliability.json": encoded(
                    reliability
                ),
                "selected_loadout.json": encoded(
                    {
                        "selection_fingerprint": SELECTION_FP,
                        "selected": {
                            "candidate_id": "candidate-1",
                            "eligible": True,
                            "hard_failures": [],
                            "candidate": {
                                "candidate_id": "candidate-1",
                                "model": {"id": "model.gguf"},
                            },
                            "metrics": selected_metrics(reliability),
                            "gates": {
                                "completion": True,
                                "streaming": True,
                                "concurrency": True,
                                "memory_headroom": True,
                                "sustained_stability": True,
                                "measurement_reliability": True,
                            },
                        },
                        "admission": {"admitted": False},
                    }
                ),
                "model_inventory.selected.json": encoded(
                    {
                        "models": [
                            {
                                "id": "model.gguf",
                                "fingerprint_mode": "full",
                                "content_sha256": MODEL_FP,
                            },
                            {
                                "id": "other.gguf",
                                "fingerprint_mode": "quick",
                                "quick_fingerprint": "sha256:" + "6" * 64,
                            },
                        ]
                    }
                ),
            }
        )
    return files


def write_bundle(
    path: Path,
    mode="observe",
    bad_hash=False,
    tampered_source=False,
    tampered_reliability=False,
):
    files = bundle_files(
        mode,
        tampered_source=tampered_source,
        tampered_reliability=tampered_reliability,
    )
    entries = []
    for name, content in sorted(files.items()):
        digest = hashlib.sha256(content).hexdigest()
        if bad_hash and name == "benchmark_plan.json":
            digest = "0" * 64
        entries.append(
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": "sha256:" + digest,
            }
        )
    source_fp = source_artifact()["source_fingerprint"]
    core = {
        "schema_version": "fleet_artifact_bundle.v1",
        "node_id": "x1-370",
        "run_id": RUN_ID,
        "remote_exit_code": 0,
        "source_fingerprint": source_fp,
        "files": entries,
    }
    manifest = encoded(
        {
            **core,
            "bundle_fingerprint": canonical_hash(core),
        }
    )
    with tarfile.open(path, "w:gz") as archive:
        for name, content in {**files, "bundle_manifest.json": manifest}.items():
            info = tarfile.TarInfo(name=f"./{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def file_hash(path: Path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(
    path: Path,
    archive: Path,
    returncode=0,
    archive_hash=None,
):
    path.write_text(
        json.dumps(
            {
                "schema_version": "fleet_rollout.v1",
                "artifact_type": "fleet_rollout_results",
                "run_id": RUN_ID,
                "results": [
                    {
                        "node_id": "x1-370",
                        "returncode": returncode,
                        "timed_out": False,
                        "scp_returncode": 0,
                        "scp_timed_out": False,
                        "collected_artifact": str(archive),
                        "collected_artifact_size_bytes": archive.stat().st_size,
                        "collected_artifact_sha256": archive_hash
                        or file_hash(archive),
                    }
                ],
            }
        )
    )


def test_observation_gate_verifies_collected_archive_and_source(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="observe")
    write_results(results, archive)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="observe"
    )
    assert report["passed"] is True
    assert report["next_stage"] == "candidate_review"
    assert report["admission"]["admitted"] is False
    summary = report["nodes"][0]["archive"]
    assert summary["observation"]["candidate_count"] == 1
    assert summary["source"]["commit"] == COMMIT
    assert summary["archive_sha256"] == file_hash(archive)


def test_sweep_gate_verifies_model_and_reliability_evidence(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="sweep")
    write_results(results, archive)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="sweep"
    )
    assert report["passed"] is True
    sweep = report["nodes"][0]["archive"]["sweep"]
    assert sweep["candidate_id"] == "candidate-1"
    assert sweep["model_id"] == "model.gguf"
    assert sweep["model_content_sha256"] == MODEL_FP
    assert sweep["reliability"]["valid_trials"] == 3
    assert sweep["reliability"]["reliability_fingerprint"].startswith(
        "sha256:"
    )
    assert report["next_stage"] == "profile_import_review"


def test_gate_rejects_tampered_reliability_report(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="sweep", tampered_reliability=True)
    write_results(results, archive)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="sweep"
    )
    assert report["passed"] is False
    assert "reliability fingerprint mismatch" in report["nodes"][0][
        "errors"
    ][0]


def test_gate_rejects_tampered_bundle_member(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="observe", bad_hash=True)
    write_results(results, archive)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="observe"
    )
    assert report["passed"] is False
    assert "hash mismatch" in report["nodes"][0]["errors"][0]


def test_gate_rejects_tampered_source_provenance(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="observe", tampered_source=True)
    write_results(results, archive)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="observe"
    )
    assert report["passed"] is False
    assert "source provenance fingerprint mismatch" in report["nodes"][0][
        "errors"
    ][0]


def test_gate_rejects_outer_archive_digest_mismatch(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="observe")
    write_results(results, archive, archive_hash="sha256:" + "0" * 64)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="observe"
    )
    assert report["passed"] is False
    assert "archive SHA-256" in report["nodes"][0]["errors"][0]


def test_gate_rejects_nonzero_remote_result_even_with_archive(tmp_path):
    archive = tmp_path / "x1-370.tar.gz"
    results = tmp_path / "rollout_results.json"
    write_bundle(archive, mode="observe")
    write_results(results, archive, returncode=1)
    report = fleet_gate_entrypoint.evaluate_rollout(
        results, ["x1-370"], mode="observe"
    )
    assert report["passed"] is False
    assert "remote rollout returned 1" in report["nodes"][0]["errors"]
