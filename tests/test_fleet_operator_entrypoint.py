from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from lms_agent_bench import fleet_operator as base
from lms_agent_bench import fleet_operator_entrypoint as entrypoint


def test_run_id_rejects_path_traversal_and_separators():
    for value in ("../run", "a/b", "a\\b", "run..copy", "", " space"):
        with pytest.raises(ValueError, match="run_id"):
            entrypoint.validate_run_id(value)
    assert entrypoint.validate_run_id("run-20260804.001") == "run-20260804.001"


def test_numeric_argument_validation_is_fail_closed():
    args = argparse.Namespace(
        command="observe",
        min_controller_free_bytes=0,
        preflight_timeout_seconds=60,
        run_id="run-a",
        render_timeout_seconds=300,
        rollout_timeout_seconds=0,
        gate_timeout_seconds=900,
    )
    with pytest.raises(ValueError, match="min_controller_free_bytes"):
        entrypoint.validate_args(args)
    args.min_controller_free_bytes = 1
    args.render_timeout_seconds = 0
    with pytest.raises(ValueError, match="render_timeout_seconds"):
        entrypoint.validate_args(args)


def test_transient_preflight_is_retried_but_deterministic_failure_is_not(monkeypatch):
    results = [
        {"ok": False, "returncode": 255, "timed_out": False},
        {"ok": True, "returncode": 0, "timed_out": False},
    ]
    calls = []

    def original(node, update_code, **kwargs):
        calls.append(1)
        return results.pop(0)

    final = entrypoint._retry_preflight_node(
        original,
        {"preflight_attempts": 3, "preflight_retry_backoff_seconds": 0},
        False,
    )
    assert final["ok"] is True
    assert final["attempt_count"] == 2
    assert len(calls) == 2

    calls.clear()

    def deterministic(node, update_code, **kwargs):
        calls.append(1)
        return {"ok": False, "returncode": 1, "timed_out": False}

    final = entrypoint._retry_preflight_node(
        deterministic,
        {"preflight_attempts": 5, "preflight_retry_backoff_seconds": 0},
        False,
    )
    assert final["attempt_count"] == 1
    assert len(calls) == 1


def test_preflight_script_redacts_git_remote_url():
    raw = 'payload = {"origin": git("remote", "get-url", "origin"),}'
    sanitized = entrypoint._sanitize_preflight_script(lambda: raw)
    assert '"origin":' not in sanitized
    assert "origin_fingerprint" in sanitized
    assert "sha256:" in sanitized


def test_controller_readiness_rejects_symlinked_inputs(tmp_path):
    config_target = tmp_path / "config-target.json"
    env_target = tmp_path / "env-target"
    config_target.write_text("{}\n", encoding="utf-8")
    env_target.write_text("SECRET=value\n", encoding="utf-8")
    config_link = tmp_path / "config.json"
    env_link = tmp_path / "private.env"
    config_link.symlink_to(config_target)
    env_link.symlink_to(env_target)

    def original(*args, **kwargs):
        return {"errors": [], "warnings": [], "ok": True}

    report = entrypoint._hardened_controller_readiness(
        original,
        str(config_link),
        str(env_link),
        tmp_path / "workspace",
        minimum_free_bytes=1,
        allow_insecure_env_file=False,
    )
    assert report["ok"] is False
    assert sum("symbolic link" in item for item in report["errors"]) == 2


def test_manifest_verification_rejects_archive_outside_run(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.tar.gz"
    outside.write_bytes(b"archive")
    (root / "operator-manifest.json").write_text(
        json.dumps(
            {
                "control_files": [],
                "archives": [{"path": str(outside)}],
            }
        ),
        encoding="utf-8",
    )
    called = False

    def original(*args, **kwargs):
        nonlocal called
        called = True
        return {"valid": True}

    with pytest.raises(ValueError, match="escapes"):
        entrypoint._harden_verify_run_manifest(original, root)
    assert called is False


def test_observe_preacquires_real_lock_before_delegation(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(
        workspace=str(tmp_path),
        config=str(config),
        run_id="run-a",
        recover_stale_lock=False,
    )
    observed = {}

    def fake_observe(namespace):
        lock = tmp_path / ".fleet-operator.lock"
        observed["lock_exists"] = lock.is_dir()
        acquired, recovered = base.acquire_lock(
            tmp_path,
            current_run="run-a",
            config_sha256=None,
            recover_stale=False,
        )
        observed["same_lock"] = acquired == lock and recovered is None
        root = tmp_path / "run-a"
        root.mkdir()
        base.release_lock(acquired)
        return 0

    monkeypatch.setattr(base, "observe_command", fake_observe)
    assert entrypoint._observe_with_preacquired_lock(args) == 0
    assert observed == {"lock_exists": True, "same_lock": True}
    assert not (tmp_path / ".fleet-operator.lock").exists()
