from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from lms_agent_bench import fleet_transfer_hardening as transfer


class Base:
    @staticmethod
    def utc_now_iso():
        return "2026-08-04T00:00:00+00:00"

    @staticmethod
    def ssh_command(target, options):
        return ["ssh", *options, target]

    @staticmethod
    def scp_command(target, remote_path, local_path, options):
        return ["scp", *options, f"{target}:{remote_path}", str(local_path)]

    @staticmethod
    def remote_artifact_path(node, run_id):
        return f"/remote/{run_id}/{node['node_id']}"

    @staticmethod
    def safe_slug(value):
        return value


def entrypoint():
    return SimpleNamespace(_base=Base)


def node(**changes):
    value = {
        "node_id": "node-a",
        "ssh_target": "operator@node-a",
        "remote_timeout_seconds": 30,
        "scp_timeout_seconds": 10,
        "scp_attempts": 3,
        "scp_retry_backoff_seconds": 0,
    }
    value.update(changes)
    return value


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_transfer_retry_does_not_rerun_remote_workload(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[0] == "ssh":
            return completed(stdout="remote complete")
        partial = Path(command[-1])
        if sum(item[0] == "scp" for item in calls) == 1:
            partial.write_bytes(b"partial")
            return completed(returncode=1, stderr="transient failure")
        partial.write_bytes(b"complete archive")
        return completed()

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.execute_remote_reliably(
        entrypoint(),
        node(),
        "echo test",
        "run-a",
        tmp_path,
        ["BatchMode=yes"],
        True,
    )

    assert sum(command[0] == "ssh" for command in calls) == 1
    assert sum(command[0] == "scp" for command in calls) == 2
    assert result["returncode"] == 0
    assert result["scp_attempt_count"] == 2
    assert result["scp_returncode"] == 0
    archive = Path(result["collected_artifact"])
    assert archive.read_bytes() == b"complete archive"
    assert result["collected_artifact_size_bytes"] == len(b"complete archive")
    assert result["collected_artifact_sha256"].startswith("sha256:")
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))


def test_failed_attempt_never_replaces_existing_final_archive(tmp_path, monkeypatch):
    final = tmp_path / "node-a.tar.gz"
    final.write_bytes(b"previous verified archive")

    def fake_run(command, **kwargs):
        if command[0] == "ssh":
            return completed()
        Path(command[-1]).write_bytes(b"truncated")
        return completed(returncode=1, stderr="network down")

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.execute_remote_reliably(
        entrypoint(),
        node(scp_attempts=2),
        "echo test",
        "run-a",
        tmp_path,
        [],
        True,
    )

    assert result["collected_artifact"] is None
    assert result["scp_attempt_count"] == 2
    assert result["scp_returncode"] == 1
    assert final.read_bytes() == b"previous verified archive"
    assert not list(tmp_path.glob(".*.partial"))


def test_successful_transfer_is_atomically_promoted(tmp_path, monkeypatch):
    observed_destinations = []

    def fake_run(command, **kwargs):
        if command[0] == "ssh":
            return completed()
        observed_destinations.append(Path(command[-1]))
        Path(command[-1]).write_bytes(b"archive")
        return completed()

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.execute_remote_reliably(
        entrypoint(), node(), "echo test", "run-a", tmp_path, [], True
    )

    assert observed_destinations[0].name.startswith(".node-a.tar.gz.attempt-1")
    assert observed_destinations[0].name.endswith(".partial")
    assert not observed_destinations[0].exists()
    assert Path(result["collected_artifact"]).name == "node-a.tar.gz"


def test_collection_exhaustion_records_every_attempt(tmp_path, monkeypatch):
    remote_calls = 0

    def fake_run(command, **kwargs):
        nonlocal remote_calls
        if command[0] == "ssh":
            remote_calls += 1
            return completed()
        return completed(returncode=255, stderr="connection reset")

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.execute_remote_reliably(
        entrypoint(),
        node(scp_attempts=4),
        "echo test",
        "run-a",
        tmp_path,
        [],
        True,
    )

    assert remote_calls == 1
    assert result["collected_artifact"] is None
    assert result["scp_attempt_count"] == 4
    assert len(result["scp_attempts"]) == 4
    assert all(item["returncode"] == 255 for item in result["scp_attempts"])


def test_no_collect_never_invokes_scp(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return completed()

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.execute_remote_reliably(
        entrypoint(), node(), "echo test", "run-a", tmp_path, [], False
    )

    assert len(commands) == 1 and commands[0][0] == "ssh"
    assert result["collected_artifact"] is None
    assert result["scp_attempts"] == []
