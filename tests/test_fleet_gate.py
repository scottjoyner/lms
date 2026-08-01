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
        "package_version": "0.23.0",
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


def bundle_files(mode="observe", tampered_source=False):
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
                            "gates": {
                                "completion": True,
                                "streaming": True,
                                "concurrency": True,
                                "memory_headroom": True,
                                "sustained_stability": True,
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
):
    files = bundle_files(mode, tampered_source=tampered_source)
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


def test_sweep_gate_accepts_one_full_hash_among_quick_inventory_records(tmp_path):
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
    assert report["next_stage"] == "profile_import_review"


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
