import json

from lms_agent_bench import fleet_rollout_command


def write_results(path, *, remote=0, scp=0, artifact=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "rollout_results.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "results": [
                    {
                        "node_id": "x1-370",
                        "returncode": remote,
                        "timed_out": False,
                        "scp_returncode": scp,
                        "scp_timed_out": False,
                        "collected_artifact": "archive.tar.gz" if artifact else None,
                        "collected_artifact_sha256": (
                            "sha256:" + "a" * 64 if artifact else None
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_rollout_command_returns_nonzero_when_collection_fails(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    write_results(output_dir, scp=1, artifact=False)
    monkeypatch.setattr(fleet_rollout_command._entrypoint, "main", lambda argv: 0)
    assert (
        fleet_rollout_command.main(
            ["run", "--output-dir", str(output_dir)]
        )
        == 1
    )


def test_rollout_command_preserves_success_with_complete_evidence(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    write_results(output_dir)
    monkeypatch.setattr(fleet_rollout_command._entrypoint, "main", lambda argv: 0)
    assert (
        fleet_rollout_command.main(
            ["run", "--output-dir", str(output_dir)]
        )
        == 0
    )


def test_no_collect_preserves_remote_command_result(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_rollout_command._entrypoint, "main", lambda argv: 0)
    assert (
        fleet_rollout_command.main(
            ["run", "--no-collect", "--output-dir", str(tmp_path)]
        )
        == 0
    )
