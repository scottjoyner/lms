import json

import pytest

from lms_agent_bench import fleet_rollout_entrypoint


def write_config(tmp_path):
    path = tmp_path / "rollout.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "fleet_rollout.v1",
                "nodes": [
                    {
                        "node_id": "x1-370",
                        "ssh_target": "${TEST_SSH_TARGET}",
                        "repo_dir": "${TEST_REPO_DIR}",
                        "branch": "full-auto-reconciliation-20260730",
                        "python": "python3",
                        "model_roots": ["${TEST_MODEL_ROOT}"],
                        "contexts": [4096, 8192],
                        "endpoint_map": {},
                        "metadata": {"tailscale_ip": "100.64.43.123"},
                    }
                ],
            }
        )
    )
    return path


def test_environment_file_resolves_rollout_placeholders(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    env_file = tmp_path / "tier1.env"
    env_file.write_text(
        "TEST_SSH_TARGET=scott@x1-370.tail.example\n"
        "TEST_REPO_DIR=/home/scott/git/lms\n"
        "TEST_MODEL_ROOT=/models\n"
    )
    monkeypatch.delenv("TEST_SSH_TARGET", raising=False)
    monkeypatch.delenv("TEST_REPO_DIR", raising=False)
    monkeypatch.delenv("TEST_MODEL_ROOT", raising=False)
    resolved = fleet_rollout_entrypoint.load_rollout_config(
        str(config), str(env_file)
    )
    node = resolved["nodes"][0]
    assert node["ssh_target"] == "scott@x1-370.tail.example"
    assert node["repo_dir"] == "/home/scott/git/lms"
    assert node["model_roots"] == ["/models"]


def test_process_environment_overrides_environment_file(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    env_file = tmp_path / "tier1.env"
    env_file.write_text(
        "TEST_SSH_TARGET=file-user@x1-370\n"
        "TEST_REPO_DIR=/from/file\n"
        "TEST_MODEL_ROOT=/models\n"
    )
    monkeypatch.setenv("TEST_REPO_DIR", "/from/process")
    resolved = fleet_rollout_entrypoint.load_rollout_config(
        str(config), str(env_file)
    )
    assert resolved["nodes"][0]["repo_dir"] == "/from/process"


def test_empty_required_placeholder_is_rejected(tmp_path, monkeypatch):
    config = write_config(tmp_path)
    env_file = tmp_path / "tier1.env"
    env_file.write_text(
        "TEST_SSH_TARGET=\n"
        "TEST_REPO_DIR=/home/scott/git/lms\n"
        "TEST_MODEL_ROOT=/models\n"
    )
    monkeypatch.delenv("TEST_SSH_TARGET", raising=False)
    with pytest.raises(ValueError, match="TEST_SSH_TARGET"):
        fleet_rollout_entrypoint.load_rollout_config(
            str(config), str(env_file)
        )


def test_resolved_validation_rejects_non_loopback_endpoint_map():
    config = {
        "schema_version": "fleet_rollout.v1",
        "nodes": [
            {
                "node_id": "x1-370",
                "ssh_target": "scott@x1-370",
                "repo_dir": "/home/scott/git/lms",
                "branch": "full-auto-reconciliation-20260730",
                "model_roots": ["/models"],
                "contexts": [4096],
                "endpoint_map": {
                    "candidate-1": "http://100.64.43.123:1236/v1"
                },
                "metadata": {"tailscale_ip": "100.64.43.123"},
            }
        ],
    }
    findings = fleet_rollout_entrypoint.validate_resolved_config(config)
    assert findings[0]["ready_for_observation"] is False
    assert "not loopback-local" in findings[0]["errors"][0]
