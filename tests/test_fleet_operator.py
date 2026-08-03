from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    }


def test_ssh_command_is_noninteractive_and_fixed():
    command = fleet_operator.ssh_command("operator@node-a")
    assert command[0] == "ssh"
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=accept-new" in command
    assert command[-3:] == ["operator@node-a", "bash", "-s"]


def test_preflight_script_contains_exact_source_and_clean_tree_checks():
    script = fleet_operator.preflight_script(node(), update_code=False)
    assert "git(\"status\", \"--porcelain\", \"--untracked-files=all\")" in script
    assert "expected branch" in script
    assert "expected commit" in script
    assert "remote Python cannot import requests" in script
    assert "missing model roots" in script


def test_preflight_failure_is_written_and_returns_nonzero(tmp_path, monkeypatch):
    loaded = {
        "config": {"nodes": [node()]},
        "coverage": {"ready": True, "coverage_complete": True},
    }
    monkeypatch.setattr(fleet_operator, "load_operator_config", lambda *args: loaded)
    monkeypatch.setattr(
        fleet_operator,
        "run_preflight",
        lambda *args: [{"node_id": "node-a", "ok": False, "returncode": 255}],
    )
    args = argparse.Namespace(
        config="config.json",
        env_file="private.env",
        workspace=str(tmp_path),
        update_code=False,
    )
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
        "run_preflight",
        lambda *args: [{"node_id": "node-a", "ok": False, "returncode": 255}],
    )
    called = {"command": False}

    def fake_run_logged(*args, **kwargs):
        called["command"] = True
        return 0

    monkeypatch.setattr(fleet_operator, "run_logged", fake_run_logged)
    args = argparse.Namespace(
        config="config.json",
        env_file="private.env",
        workspace=str(tmp_path),
        update_code=False,
        run_id="test-run",
    )
    assert fleet_operator.observe_command(args) == 1
    assert called["command"] is False
    state = json.loads(
        (tmp_path / "test-run" / "operator-state.json").read_text()
    )
    assert state["failure_stage"] == "preflight"
    assert state["success"] is False
    assert not (tmp_path / ".fleet-operator.lock").exists()


def test_observe_builds_fixed_render_rollout_and_gate_commands(
    tmp_path, monkeypatch
):
    loaded = {
        "config": {"nodes": [node("node-a"), node("node-b")]},
        "coverage": {"ready": True, "coverage_complete": True},
    }
    monkeypatch.setattr(fleet_operator, "load_operator_config", lambda *args: loaded)
    monkeypatch.setattr(
        fleet_operator,
        "run_preflight",
        lambda *args: [
            {"node_id": "node-a", "ok": True, "returncode": 0},
            {"node_id": "node-b", "ok": True, "returncode": 0},
        ],
    )
    commands = []

    def fake_run_logged(command, log_path):
        commands.append(list(command))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("ok\n")
        return 0

    monkeypatch.setattr(fleet_operator, "run_logged", fake_run_logged)
    args = argparse.Namespace(
        config="config.json",
        env_file="private.env",
        workspace=str(tmp_path),
        update_code=True,
        run_id="test-run",
    )
    assert fleet_operator.observe_command(args) == 0
    assert len(commands) == 3
    assert "lms_agent_bench.fleet_rollout_complete" in commands[0]
    assert "render" in commands[0]
    assert "--update-code" in commands[0]
    assert "run" in commands[1]
    assert "--continue-on-error" in commands[1]
    assert "lms_agent_bench.fleet_gate_entrypoint" in commands[2]
    assert commands[2].count("--required-node") == 2
    state = json.loads(
        (tmp_path / "test-run" / "operator-state.json").read_text()
    )
    assert state["success"] is True
    assert state["admission"]["admitted"] is False
