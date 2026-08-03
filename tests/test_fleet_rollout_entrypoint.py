import hashlib
import json
import subprocess

import pytest

from lms_agent_bench import fleet_rollout_entrypoint


COMMIT = "a" * 40
NODE = {
    "node_id": "x1-370",
    "ssh_target": "scott@x1-370",
    "repo_dir": "/home/scott/git/lms",
    "branch": "full-auto-reconciliation-20260730",
    "expected_commit": COMMIT,
    "python": "/home/scott/venvs/lms/bin/python",
    "model_roots": ["~/models", "/opt/models"],
    "contexts": [4096, 8192],
    "metadata": {"tier": 1},
}


def test_dry_run_routes_hardened_entrypoints_without_selection():
    script = fleet_rollout_entrypoint.build_remote_script(NODE, "run-dry")
    assert "lms_agent_bench.fleet_loadout_entrypoint discover" in script
    assert "lms_agent_bench.fleet_loadout_entrypoint plan" in script
    assert "lms_agent_bench.fleet_bench_entrypoint" in script
    assert "lms_agent_bench.fleet_loadout_entrypoint select" not in script
    assert "lms_agent_bench.fleet_bench_plan" not in script


def test_remote_script_attests_source_and_holds_one_node_lock():
    script = fleet_rollout_entrypoint.build_remote_script(NODE, "run-dry")
    assert "export RUN_ID=run-dry" in script
    assert "fleet_provenance" in script
    assert f"--expected-commit {COMMIT}" in script
    assert "source_control.json" in script
    assert "LOCK_ACQUIRED=0" in script
    assert "another LMS fleet rollout holds" in script
    assert "bundle_fingerprint" in script
    assert 'rm -rf -- "$LOCK_DIR"' in script


def test_execute_routes_hardened_selection_entrypoint():
    script = fleet_rollout_entrypoint.build_remote_script(
        NODE,
        "run-live",
        execute_candidates=["candidate-1"],
    )
    assert "lms_agent_bench.fleet_bench_entrypoint" in script
    assert "lms_agent_bench.fleet_loadout_entrypoint select" in script
    assert "--candidate candidate-1" in script


def test_env_file_overrides_ambient_values(tmp_path, monkeypatch):
    config_path = tmp_path / "rollout.json"
    env_path = tmp_path / "rollout.env"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fleet_rollout.v1",
                "nodes": [
                    {
                        "node_id": "x1-370",
                        "ssh_target": "$SSH_TARGET",
                        "repo_dir": "$REPO_DIR",
                        "branch": "full-auto-reconciliation-20260730",
                        "expected_commit": "$EXPECTED_COMMIT",
                        "python": "$PYTHON_BIN",
                        "model_roots": ["$MODEL_ROOT"],
                        "contexts": [4096],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    env_path.write_text(
        "\n".join(
            [
                "SSH_TARGET=scott@x1-370",
                "REPO_DIR=/srv/lms",
                "EXPECTED_COMMIT=" + COMMIT,
                "PYTHON_BIN=/srv/venv/bin/python",
                "MODEL_ROOT=/srv/models",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REPO_DIR", "/wrong/ambient/path")

    config = fleet_rollout_entrypoint.load_rollout_config(
        str(config_path), str(env_path)
    )
    node = config["nodes"][0]
    assert node["repo_dir"] == "/srv/lms"
    assert node["python"] == "/srv/venv/bin/python"
    assert node["expected_commit"] == COMMIT
    assert fleet_rollout_entrypoint.validate_resolved_config(config)[0][
        "ready_for_observation"
    ] is True


def test_missing_placeholder_is_rejected(tmp_path):
    config_path = tmp_path / "rollout.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fleet_rollout.v1",
                "nodes": [
                    {
                        "node_id": "x1-370",
                        "ssh_target": "$MISSING_TARGET",
                        "repo_dir": "/srv/lms",
                        "branch": "full-auto-reconciliation-20260730",
                        "expected_commit": COMMIT,
                        "model_roots": ["/srv/models"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="MISSING_TARGET"):
        fleet_rollout_entrypoint.load_rollout_config(str(config_path), None)


def test_resolved_validation_rejects_ambiguous_paths_remote_map_and_commit():
    config = {
        "schema_version": "fleet_rollout.v1",
        "nodes": [
            {
                **NODE,
                "repo_dir": "~/git/lms",
                "python": "venv/bin/python",
                "expected_commit": "short",
                "remote_timeout_seconds": 0,
                "endpoint_map": {
                    "candidate-1": "http://100.64.43.123:1236/v1"
                },
            }
        ],
    }
    finding = fleet_rollout_entrypoint.validate_resolved_config(config)[0]
    assert finding["ready_for_observation"] is False
    assert "repo_dir must be an absolute remote path" in finding["errors"]
    assert any("python must be" in item for item in finding["errors"])
    assert any("expected_commit" in item for item in finding["errors"])
    assert any("remote_timeout_seconds" in item for item in finding["errors"])
    assert any("not loopback-local" in item for item in finding["errors"])


def test_execute_remote_records_outer_archive_digest(tmp_path, monkeypatch):
    archive_bytes = b"test archive bytes"

    def fake_run(command, **kwargs):
        if command[0] == "ssh":
            return subprocess.CompletedProcess(command, 0, "remote ok", "")
        assert command[0] == "scp"
        local_path = tmp_path / "x1-370.tar.gz"
        assert str(local_path) == command[-1]
        local_path.write_bytes(archive_bytes)
        return subprocess.CompletedProcess(command, 0, "copied", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = fleet_rollout_entrypoint.execute_remote(
        NODE,
        "echo test",
        "run-1",
        tmp_path,
        ["BatchMode=yes"],
        collect=True,
    )
    expected = "sha256:" + hashlib.sha256(archive_bytes).hexdigest()
    assert result["collected_artifact_sha256"] == expected
    assert result["collected_artifact_size_bytes"] == len(archive_bytes)
    assert result["timed_out"] is False
    assert result["scp_timed_out"] is False


def test_execute_remote_bounds_ssh_hangs(tmp_path, monkeypatch):
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(command, timeout=1)
        return subprocess.CompletedProcess(command, 1, "", "missing archive")

    monkeypatch.setattr(subprocess, "run", fake_run)
    node = {**NODE, "remote_timeout_seconds": 1}
    result = fleet_rollout_entrypoint.execute_remote(
        node,
        "echo test",
        "run-1",
        tmp_path,
        ["BatchMode=yes"],
        collect=True,
    )
    assert result["returncode"] == 124
    assert result["timed_out"] is True
    assert "timed out after 1s" in result["stderr"]
