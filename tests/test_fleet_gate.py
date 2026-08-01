import hashlib
import io
import json
import tarfile
from pathlib import Path

from lms_agent_bench import fleet_gate_entrypoint

OBSERVATION_FP = "sha256:" + "1" * 64
PLAN_FP = "sha256:" + "2" * 64
EXECUTION_FP = "sha256:" + "3" * 64
SELECTION_FP = "sha256:" + "4" * 64
MODEL_FP = "sha256:" + "5" * 64


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def bundle_files(mode="observe"):
    files = {
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


def write_bundle(path: Path, mode="observe", bad_hash=False):
    files = bundle_files(mode)
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
    manifest = encoded(
        {
            "schema_version": "fleet_artifact_bundle.v1",
            "node_id": "x1-370",
            "remote_exit_code": 0,
            "files": entries,
        }
    )
    with tarfile.open(path, "w:gz") as archive:
        for name, content in {**files, "bundle_manifest.json": manifest}.items():
            info = tarfile.TarInfo(name=f"./{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def write_results(path: Path, archive: Path, returncode=0):
    path.write_text(
        json.dumps(
            {
                "schema_version": "fleet_rollout.v1",
                "artifact_type": "fleet_rollout_results",
                "run_id": "run-1",
                "results": [
                    {
                        "node_id": "x1-370",
                        "returncode": returncode,
                        "scp_returncode": 0,
                        "collected_artifact": str(archive),
                    }
                ],
            }
        )
    )


def test_observation_gate_verifies_collected_archive(tmp_path):
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
    assert report["nodes"][0]["archive"]["observation"]["candidate_count"] == 1


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
