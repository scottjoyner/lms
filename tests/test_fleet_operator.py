from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from lms_agent_bench import fleet_operator


def node(node_id="node-a"):
    return {
        "node_id": node_id,
        "ssh_target": f"operator@{node_id}",
        "repo_dir": "/srv/lms",
        "branch": "full-auto-reconciliation-20260730",
        "expected_commit": "a" * 40,
        "python": "/usr/bin/python3",
        "model_roots": ["/models"],
        "artifact_root": "/var/tmp/lms-fleet",
        "lock_root": "/var/tmp/lms-fleet-locks",
    }


def readiness(tmp_path):
    return {
        "schema_version": "fleet_controller_readiness.v1",
        "ok": True,
        "config": str(tmp_path / "config.json"),
        "config_sha256": "sha256:" + "1" * 64,
        "env_file": str(tmp_path / "private.env"),
        "env_file_sha256": "sha256:" + "2" * 64,
        "admission": {"admitted": False},
    }


def observe_args(tmp_path, **changes):
    values = {
        "config": str(tmp_path / "config.json"),
        "env_file": str(tmp_path / "private.env"),
        "workspace": str(tmp_path),
        "update_code": False,
        "accept_new_host_keys": False,
        "allow_insecure_env_file": False,
        "min_controller_free_bytes": 1,
        "preflight_timeout_seconds": 5,
        "run_id": "test-run",
        "recover_stale_lock": False,
        "render_timeout_seconds": 5,
        "rollout_timeout_seconds": 5,
        "gate_timeout_seconds": 5,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def command_result(command, log_path, returncode=0):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text("ok\n", encoding="utf-8")
    return {
        "command": list(command),
        "log": str(log_path),
        "started_at_utc": "2026-08-04T00:00:00+00:00",
        "finished_at_utc": "2026-08-04T00:00:01+00:00",
        "duration_seconds": 1.0,
        "timeout_seconds": 5,
        "timed_out": False,
        "returncode": returncode,
        "log_sha256": fleet_operator.file_sha256(Path(log_path)),
    }


def test_ssh_command_is_noninteractive_and_strict_by_default():
    command = fleet_operator.ssh_command("operator@node-a")
    assert command[0] == "ssh"
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=yes" in command
    assert "StrictHostKeyChecking=accept-new" not in command
    assert command[-3:] == ["operator@node-a", "bash", "-s"]

    bootstrap = fleet_operator.ssh_command(
        "operator@node-a", accept_new_host_keys=True
    )
    assert "StrictHostKeyChecking=accept-new" in bootstrap


def test_preflight_script_contains_source_capacity_lock_and_collision_checks():
    script = fleet_operator.preflight_script(
        node(), update_code=False, run_id_value="run-1"
    )
    assert "remote checkout is not completely clean" in script
    assert "expected branch" in script
    assert "expected commit" in script
    assert "required remote command is missing" in script
    assert "artifact filesystem free bytes" in script
    assert "open-file soft limit" in script
    assert "remote rollout lock is" in script
    assert "remote run ID already has artifacts" in script


def test_controller_readiness_rejects_insecure_secret_permissions(tmp_path):
    config = tmp_path / "config.json"
    environment = tmp_path / "private.env"
    config.write_text("{}\n", encoding="utf-8")
    environment.write_text("SECRET=value\n", encoding="utf-8")
    environment.chmod(0o644)
    report = fleet_operator.controller_readiness(
        str(config),
        str(environment),
        tmp_path / "workspace",
        minimum_free_bytes=1,
        allow_insecure_env_file=False,
    )
    if os.name == "posix":
        assert report["ok"] is False
        assert any("permissions" in item for item in report["errors"])


def test_local_lock_requires_explicit_safe_stale_recovery(tmp_path):
    lock = tmp_path / ".fleet-operator.lock"
    lock.mkdir()
    fleet_operator.write_json(
        lock / "owner.json",
        {
            "hostname": socket.gethostname(),
            "pid": 999999999,
            "boot_id": fleet_operator._boot_id(),
        },
    )
    with pytest.raises(RuntimeError, match="cannot be safely recovered"):
        fleet_operator.acquire_lock(
            tmp_path,
            current_run="run-a",
            config_sha256=None,
            recover_stale=False,
        )
    acquired, recovered = fleet_operator.acquire_lock(
        tmp_path,
        current_run="run-a",
        config_sha256=None,
        recover_stale=True,
    )
    assert acquired.is_dir()
    assert recovered is not None and recovered.is_dir()
    fleet_operator.release_lock(acquired)


def test_run_logged_terminates_timed_out_process_group(tmp_path):
    result = fleet_operator.run_logged(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tmp_path / "timeout.log",
        timeout_seconds=1,
    )
    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "timed out" in (tmp_path / "timeout.log").read_text()


def test_preflight_failure_is_written_and_returns_nonzero(tmp_path, monkeypatch):
    loaded = {
        "config": {"nodes": [node()]},
        "coverage": {"ready": True, "coverage_complete": True},
    }
    monkeypatch.setattr(fleet_operator, "load_operator_config", lambda *args: loaded)
    monkeypatch.setattr(
        fleet_operator,
        "controller_readiness",
        lambda *args, **kwargs: readiness(tmp_path),
    )
    monkeypatch.setattr(
        fleet_operator,
        "run_preflight",
        lambda *args, **kwargs: [
            {"node_id": "node-a", "ok": False, "returncode": 255}
        ],
    )
    args = observe_args(tmp_path)
    assert fleet_operator.preflight_command(args) == 1
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert report["ok"] is False
    assert report["admission"]["admitted"] is False


def test_observe_aborts_before_render_when_any_node_fails_preflight(
    tmp_path, monkeypatch
):
    loaded = {
        "config": {"nodes": [node()]},
        "coverage": {"ready": True, "coverage_complete": True},
    }
    monkeypatch.setattr(fleet_operator, "load_operator_config", lambda *args: loaded)
    monkeypatch.setattr(
        fleet_operator,
        "controller_readiness",
        lambda *args, **kwargs: readiness(tmp_path),
    )
    monkeypatch.setattr(
        fleet_operator,
        "run_preflight",
        lambda *args, **kwargs: [
            {"node_id": "node-a", "ok": False, "returncode": 255}
        ],
    )
    called = {"command": False}

    def fake_run_logged(*args, **kwargs):
        called["command"] = True
        raise AssertionError("render should not run")

    monkeypatch.setattr(fleet_operator, "run_logged", fake_run_logged)
    args = observe_args(tmp_path)
    assert fleet_operator.observe_command(args) == 1
    assert called["command"] is False
    state = json.loads(
        (tmp_path / "test-run" / "operator-state.json").read_text()
    )
    assert state["failure_stage"] == "preflight"
    assert state["success"] is False
    assert not (tmp_path / ".fleet-operator.lock").exists()
    assert (tmp_path / "test-run" / "operator-manifest.json").is_file()


def test_observe_runs_postflight_gate_and_verifiable_manifest(
    tmp_path, monkeypatch
):
    loaded = {
        "config": {"nodes": [node("node-a"), node("node-b")]},
        "coverage": {"ready": True, "coverage_complete": True},
    }
    monkeypatch.setattr(fleet_operator, "load_operator_config", lambda *args: loaded)
    monkeypatch.setattr(
        fleet_operator,
        "controller_readiness",
        lambda *args, **kwargs: readiness(tmp_path),
    )
    phases = []

    def fake_preflight(*args, **kwargs):
        phases.append(kwargs.get("phase"))
        return [
            {"node_id": "node-a", "ok": True, "returncode": 0},
            {"node_id": "node-b", "ok": True, "returncode": 0},
        ]

    monkeypatch.setattr(fleet_operator, "run_preflight", fake_preflight)
    commands = []

    def fake_run_logged(command, log_path, **kwargs):
        commands.append(list(command))
        if "fleet_rollout_complete" in " ".join(command) and "run" in command:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            fleet_operator.write_json(
                output / "rollout_results.json",
                {
                    "schema_version": "fleet_rollout.v1",
                    "run_id": "test-run",
                    "results": [],
                },
            )
        return command_result(command, log_path)

    monkeypatch.setattr(fleet_operator, "run_logged", fake_run_logged)
    args = observe_args(tmp_path, update_code=True)
    assert fleet_operator.observe_command(args) == 0
    assert phases == ["preflight", "postflight"]
    assert len(commands) == 3
    assert "--update-code" in commands[0]
    assert "--continue-on-error" in commands[1]
    assert "lms_agent_bench.fleet_gate_entrypoint" in commands[2]
    state = json.loads(
        (tmp_path / "test-run" / "operator-state.json").read_text()
    )
    assert state["success"] is True
    verified = fleet_operator.verify_run_manifest(
        tmp_path / "test-run", require_success=True
    )
    assert verified["valid"] is True
    assert verified["success"] is True


def test_postflight_failure_blocks_success_even_after_rollout(tmp_path, monkeypatch):
    loaded = {
        "config": {"nodes": [node()]},
        "coverage": {"ready": True, "coverage_complete": True},
    }
    monkeypatch.setattr(fleet_operator, "load_operator_config", lambda *args: loaded)
    monkeypatch.setattr(
        fleet_operator,
        "controller_readiness",
        lambda *args, **kwargs: readiness(tmp_path),
    )

    def fake_preflight(*args, **kwargs):
        if kwargs.get("phase") == "postflight":
            return [{"node_id": "node-a", "ok": False, "returncode": 1}]
        return [{"node_id": "node-a", "ok": True, "returncode": 0}]

    monkeypatch.setattr(fleet_operator, "run_preflight", fake_preflight)

    def fake_run_logged(command, log_path, **kwargs):
        if "fleet_rollout_complete" in " ".join(command) and "run" in command:
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True, exist_ok=True)
            fleet_operator.write_json(
                output / "rollout_results.json",
                {"schema_version": "fleet_rollout.v1", "run_id": "test-run", "results": []},
            )
        return command_result(command, log_path)

    monkeypatch.setattr(fleet_operator, "run_logged", fake_run_logged)
    assert fleet_operator.observe_command(observe_args(tmp_path)) == 1
    state = json.loads(
        (tmp_path / "test-run" / "operator-state.json").read_text()
    )
    assert state["failure_stage"] == "postflight"
    assert state["success"] is False


def test_run_manifest_detects_control_file_tampering(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    fleet_operator.write_json(root / "operator-state.json", {"success": True})
    fleet_operator.build_run_manifest(
        root,
        {
            "run_id": "run",
            "success": True,
            "config_sha256": "sha256:" + "1" * 64,
            "env_file_sha256": "sha256:" + "2" * 64,
        },
    )
    assert fleet_operator.verify_run_manifest(root)["valid"] is True
    (root / "operator-state.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        fleet_operator.verify_run_manifest(root)
