from __future__ import annotations

import json
from pathlib import Path

import pytest

from lms_agent_bench import fleet_operational_hardening as hardening
from lms_agent_bench import fleet_rollout_complete


def test_harden_ssh_argv_rejects_disabled_host_verification():
    with pytest.raises(ValueError, match="may not be disabled"):
        hardening.harden_ssh_argv(
            ["run", "--ssh-option", "StrictHostKeyChecking=no"]
        )
    with pytest.raises(ValueError, match="/dev/null"):
        hardening.harden_ssh_argv(
            ["run", "--ssh-option", "UserKnownHostsFile=/dev/null"]
        )


def test_accept_new_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="requires"):
        hardening.harden_ssh_argv(
            ["run", "--ssh-option", "StrictHostKeyChecking=accept-new"]
        )
    argv, mode = hardening.harden_ssh_argv(
        [
            "run",
            "--ssh-option",
            "StrictHostKeyChecking=accept-new",
            "--allow-accept-new-host-keys",
        ]
    )
    assert mode == "accept_new_explicit"
    assert "--allow-accept-new-host-keys" not in argv


def test_safe_ssh_defaults_are_injected_for_run_only():
    argv, mode = hardening.harden_ssh_argv(["run", "--config", "fleet.json"])
    assert mode == "strict_known_hosts"
    assert "StrictHostKeyChecking=yes" in argv
    assert "ServerAliveInterval=15" in argv

    render, render_mode = hardening.harden_ssh_argv(
        ["render", "--config", "fleet.json"]
    )
    assert render_mode == "not_applicable"
    assert render == ["render", "--config", "fleet.json"]


def test_update_code_is_pinned_to_fetched_exact_commit():
    original = '''git -C "$REPO_DIR" fetch --prune origin "$EXPECTED_BRANCH"
git -C "$REPO_DIR" checkout "$EXPECTED_BRANCH"
git -C "$REPO_DIR" pull --ff-only origin "$EXPECTED_BRANCH"'''
    hardened = hardening.harden_exact_update_script(
        original,
        {"expected_commit": "a" * 40},
        True,
    )
    assert "pull --ff-only" not in hardened
    assert "FETCHED_COMMIT=$(git -C \"$REPO_DIR\" rev-parse FETCH_HEAD)" in hardened
    assert "origin branch moved" in hardened
    assert "merge-base --is-ancestor" in hardened
    assert "merge --ff-only \"$FETCHED_COMMIT\"" in hardened
    assert "a" * 40 in hardened


def test_update_hardening_rejects_unexpected_script_shape():
    with pytest.raises(RuntimeError, match="changed unexpectedly"):
        hardening.harden_exact_update_script(
            "git pull origin main",
            {"expected_commit": "a" * 40},
            True,
        )


def test_remote_lock_snippet_records_actual_shell_owner_and_stale_logic():
    class Base:
        @staticmethod
        def safe_slug(value):
            return value

        @staticmethod
        def q(value):
            return repr(str(value))

    snippet = hardening.remote_lock_and_provenance_snippet(
        Base,
        {
            "node_id": "node-a",
            "expected_commit": "a" * 40,
            "lock_root": "/tmp/locks",
        },
        "run-a",
    )
    assert "SHELL_OWNER_PID=${BASHPID:-$$}" in snippet
    assert "fleet_remote_lock.v2" in snippet
    assert "boot_id" in snippet
    assert "archived stale LMS fleet lock" in snippet
    assert "unknown" in snippet and "foreign" in snippet and "active" in snippet


def test_rollout_complete_reaches_final_coverage_enforcement(tmp_path, monkeypatch):
    out = tmp_path / "validation.json"

    def loader(path, env_file):
        return {"schema_version": "fleet_rollout.v1", "nodes": []}

    monkeypatch.setattr(fleet_rollout_complete._entrypoint, "load_rollout_config", loader)
    monkeypatch.setattr(
        fleet_rollout_complete,
        "validate_rollout_coverage",
        lambda config, path: {"ready": False, "errors": ["missing node"]},
    )

    def command_main(argv):
        fleet_rollout_complete._entrypoint.load_rollout_config("fleet.json", None)
        out.write_text(
            json.dumps(
                {
                    "schema_version": "fleet_rollout_validation.v1",
                    "ready_for_observation": True,
                    "admission": {"admitted": False},
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(fleet_rollout_complete._command, "main", command_main)
    assert (
        fleet_rollout_complete.main(
            ["validate", "--config", "fleet.json", "--out", str(out)]
        )
        == 1
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ready_for_observation"] is False
    assert report["coverage"]["ready"] is False


def test_rollout_complete_restores_loader_after_failure(monkeypatch):
    original = lambda path, env_file: {"nodes": []}
    monkeypatch.setattr(
        fleet_rollout_complete._entrypoint, "load_rollout_config", original
    )

    def explode(argv):
        raise RuntimeError("boom")

    monkeypatch.setattr(fleet_rollout_complete._command, "main", explode)
    with pytest.raises(RuntimeError, match="boom"):
        fleet_rollout_complete.main(["render", "--config", "fleet.json"])
    assert fleet_rollout_complete._entrypoint.load_rollout_config is original
