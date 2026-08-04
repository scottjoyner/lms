from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest

from lms_agent_bench import fleet_operator
from lms_agent_bench import loadout_qualification_operator as operator
from lms_agent_bench.model_loadout import validate_manifest


def make_inputs(tmp_path: Path):
    model_bytes = b"approved exact model artifact"
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(model_bytes)
    model_hash = fleet_operator.file_sha256(model_path)
    loadout = validate_manifest(
        {
            "schema_version": "model_loadout_manifest.v1",
            "node_id": "node-a",
            "candidate_id": "candidate-a",
            "model": {
                "id": "model-a",
                "content_sha256": model_hash,
                "format": "gguf",
            },
            "architecture": {
                "kind": "dense",
                "total_parameter_count": 1_000_000,
                "active_parameter_count_per_token": 1_000_000,
            },
            "weight_quantization": {"scheme": "q4_k_m"},
            "runtime": {
                "engine": "llama.cpp",
                "backend": "cpu",
                "version": "1",
                "build_commit": "runtime-commit",
            },
            "context": {
                "configured_tokens": 4096,
                "model_native_tokens": 4096,
            },
            "kv_cache": {
                "key_dtype": "q8_0",
                "value_dtype": "q8_0",
                "location": "cpu",
            },
            "concurrency": {"parallel_slots": 1},
            "speculative_decoding": {"enabled": False},
        }
    )
    loadout_path = tmp_path / "loadout.json"
    fleet_operator.write_json(loadout_path, loadout)
    inventory_path = tmp_path / "inventory.csv"
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "host_name",
                "host_ip",
                "endpoint_id",
                "base_url",
                "reachable",
                "model_id",
                "model_key",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "host_name": "node-a",
                "host_ip": "127.0.0.1",
                "endpoint_id": "node-a-loopback",
                "base_url": "http://127.0.0.1:8080/v1",
                "reachable": "true",
                "model_id": "model-a",
                "model_key": "model-a",
            }
        )
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "suite_id": "qualification-throughput.v1",
                "version": 1,
                "cases": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    lms_repo = tmp_path / "lms"
    hermes_repo = tmp_path / "hermes"
    lms_repo.mkdir()
    hermes_repo.mkdir()
    return {
        "loadout": loadout,
        "loadout_path": loadout_path,
        "inventory_path": inventory_path,
        "cases_path": cases_path,
        "model_path": model_path,
        "lms_repo": lms_repo,
        "hermes_repo": hermes_repo,
    }


def args_for(tmp_path: Path, inputs, **changes):
    values = {
        "command": "run",
        "loadout": inputs["loadout_path"],
        "inventory_csv": inputs["inventory_path"],
        "cases_file": inputs["cases_path"],
        "model_artifact": inputs["model_path"],
        "endpoint": "http://127.0.0.1:8080/v1",
        "api_key_env": "QUALIFICATION_API_KEY",
        "lms_repo": inputs["lms_repo"],
        "lms_branch": "reviewed",
        "lms_commit": "a" * 40,
        "hermes_repo": inputs["hermes_repo"],
        "hermes_branch": "reviewed",
        "hermes_commit": "b" * 40,
        "workspace": tmp_path / "runs",
        "run_id": "qualification-run-a",
        "recover_stale_lock": False,
        "throughput_trials": 3,
        "max_trial_attempts": 5,
        "warmup_runs": 3,
        "request_timeout_seconds": 30.0,
        "endpoint_timeout_seconds": 5.0,
        "throughput_phase_timeout_seconds": 60,
        "hermes_trials": 3,
        "hermes_trial_timeout_seconds": 30.0,
        "hermes_phase_timeout_seconds": 60,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def stable_source(repo, *, label, expected_branch, expected_commit, **kwargs):
    core = {
        "label": label,
        "repo": str(Path(repo).resolve()),
        "branch": expected_branch,
        "commit": expected_commit,
        "origin_fingerprint": "sha256:" + ("1" if label == "LMS" else "2") * 64,
    }
    return {**core, "source_fingerprint": fleet_operator.canonical_hash(core)}


def test_inventory_must_be_one_exact_loopback_identity(tmp_path):
    inputs = make_inputs(tmp_path)
    identity = operator.inventory_identity(
        inputs["inventory_path"],
        inputs["loadout"],
        "http://127.0.0.1:8080/v1",
    )
    assert identity["host_name"] == "node-a"
    assert identity["model_key"] == "model-a"

    rows = inputs["inventory_path"].read_text(encoding="utf-8")
    inputs["inventory_path"].write_text(rows.replace("node-a,", "wrong-node,"), encoding="utf-8")
    with pytest.raises(ValueError, match="node_id"):
        operator.inventory_identity(
            inputs["inventory_path"],
            inputs["loadout"],
            "http://127.0.0.1:8080/v1",
        )


def test_source_snapshot_rejects_dirty_checkout(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(operator, "_git", lambda *args: "modified-file")
    with pytest.raises(ValueError, match="not completely clean"):
        operator.source_snapshot(
            repo,
            label="LMS",
            expected_branch="reviewed",
            expected_commit="a" * 40,
        )


def test_model_hash_mismatch_stops_before_run_directory(tmp_path, monkeypatch):
    inputs = make_inputs(tmp_path)
    inputs["model_path"].write_bytes(b"changed model")
    monkeypatch.setattr(operator, "source_snapshot", stable_source)
    monkeypatch.setattr(operator, "endpoint_probe", lambda *args, **kwargs: {"ok": True})
    with pytest.raises(ValueError, match="model artifact SHA-256"):
        operator.run_qualification(args_for(tmp_path, inputs))
    assert not (tmp_path / "runs" / "qualification-run-a").exists()


def test_full_sequence_is_ordered_locked_and_verifiable(tmp_path, monkeypatch):
    inputs = make_inputs(tmp_path)
    monkeypatch.setattr(operator, "source_snapshot", stable_source)
    monkeypatch.setattr(
        operator,
        "endpoint_probe",
        lambda *args, **kwargs: {"ok": True, "endpoint": args[0]},
    )
    phases = []
    commands = []

    def fake_phase(state, name, command, root, timeout_seconds):
        phases.append(name)
        commands.append(list(command))
        state.setdefault("phases", {})[name] = {
            "returncode": 0,
            "timed_out": False,
            "timeout_seconds": timeout_seconds,
        }
        if name == "throughput":
            (root / "throughput").mkdir(parents=True, exist_ok=True)
            fleet_operator.write_json(root / "throughput" / "reliability.json", {"passed": True})
        elif name == "hermes-base":
            fleet_operator.write_json(root / "hermes-base.json", {"gate": {"passed": True}})
        elif name == "hermes-context":
            fleet_operator.write_json(root / "hermes-context.json", {"gate": {"passed": True}})
        elif name == "bind-throughput":
            fleet_operator.write_json(root / "throughput-evidence.json", {"throughput_qualified": True})
        elif name == "qualify":
            fleet_operator.write_json(
                root / "loadout-qualification.json",
                {
                    "schema_version": "loadout_qualification.v1",
                    "qualified": True,
                    "qualification_fingerprint": "sha256:" + "9" * 64,
                },
            )
        fleet_operator.write_json(root / "qualification-state.json", state)
        return True

    monkeypatch.setattr(operator, "_phase", fake_phase)
    monkeypatch.setattr(
        operator,
        "verify_qualification",
        lambda report, loadout: {"fingerprint": report["qualification_fingerprint"]},
    )
    assert operator.run_qualification(args_for(tmp_path, inputs)) == 0
    assert phases == [
        "throughput",
        "hermes-base",
        "hermes-context",
        "bind-throughput",
        "qualify",
    ]
    assert all("supersecret" not in " ".join(command) for command in commands)
    root = tmp_path / "runs" / "qualification-run-a"
    state = json.loads((root / "qualification-state.json").read_text(encoding="utf-8"))
    assert state["success"] is True
    assert state["admission"]["admitted"] is False
    assert state["qualification_fingerprint"] == "sha256:" + "9" * 64
    assert not (tmp_path / "runs" / ".qualification-locks" / ".fleet-operator.lock").exists()
    verified = operator.verify_manifest(root, require_success=True)
    assert verified["valid"] is True
    assert verified["success"] is True
    assert verified["artifact_count"] > 5


def test_phase_failure_finalizes_manifest_and_releases_lock(tmp_path, monkeypatch):
    inputs = make_inputs(tmp_path)
    monkeypatch.setattr(operator, "source_snapshot", stable_source)
    monkeypatch.setattr(operator, "endpoint_probe", lambda *args, **kwargs: {"ok": True})
    phases = []

    def fake_phase(state, name, command, root, timeout_seconds):
        phases.append(name)
        state.setdefault("phases", {})[name] = {
            "returncode": 1 if name == "hermes-base" else 0,
            "timed_out": False,
        }
        fleet_operator.write_json(root / "qualification-state.json", state)
        return name != "hermes-base"

    monkeypatch.setattr(operator, "_phase", fake_phase)
    assert operator.run_qualification(args_for(tmp_path, inputs)) == 1
    assert phases == ["throughput", "hermes-base"]
    root = tmp_path / "runs" / "qualification-run-a"
    state = json.loads((root / "qualification-state.json").read_text(encoding="utf-8"))
    assert state["success"] is False
    assert state["failure_stage"] == "hermes-base"
    assert (root / "qualification-run-manifest.json").is_file()
    assert not (tmp_path / "runs" / ".qualification-locks" / ".fleet-operator.lock").exists()


def test_manifest_detects_artifact_tampering(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    fleet_operator.write_json(root / "qualification-state.json", {"success": True})
    operator.build_manifest(
        root,
        {
            "run_id": "run",
            "success": True,
            "identity": {"loadout_fingerprint": "sha256:" + "1" * 64},
            "sources": {},
            "inputs": {},
            "qualification_fingerprint": "sha256:" + "2" * 64,
        },
    )
    assert operator.verify_manifest(root)["valid"] is True
    (root / "qualification-state.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        operator.verify_manifest(root)


def test_positive_limit_validation():
    args = argparse.Namespace(
        command="run",
        throughput_trials=3,
        max_trial_attempts=2,
        warmup_runs=3,
        request_timeout_seconds=1,
        endpoint_timeout_seconds=1,
        throughput_phase_timeout_seconds=1,
        hermes_trials=3,
        hermes_trial_timeout_seconds=1,
        hermes_phase_timeout_seconds=1,
    )
    with pytest.raises(ValueError, match="max_trial_attempts"):
        operator._validate_positive(args)
