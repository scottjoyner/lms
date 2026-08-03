import pytest

from lms_agent_bench.fleet_rollout import (
    build_remote_script,
    parse_execute_candidates,
    resolve_nodes,
    validate_config,
)


CONFIG = {
    "schema_version": "fleet_rollout.v1",
    "nodes": [
        {
            "node_id": "x1-370",
            "ssh_target": "scott@x1-370",
            "repo_dir": "/home/scott/git/lms",
            "branch": "full-auto-reconciliation-20260730",
            "model_roots": ["/models"],
            "endpoints": ["http://127.0.0.1:1234/v1"],
            "contexts": [4096, 8192],
        }
    ],
}


def test_config_and_node_selection_are_explicit():
    validate_config(CONFIG)
    with pytest.raises(ValueError):
        resolve_nodes(CONFIG, [], False)
    assert resolve_nodes(CONFIG, ["x1-370"], False)[0]["node_id"] == "x1-370"


def test_default_script_is_read_only_and_dry_run():
    script = build_remote_script(CONFIG["nodes"][0], "run-1")
    assert "--dry-run" in script
    assert "--all --limit 4" in script
    assert "git -C \"$REPO_DIR\" pull" not in script
    assert "selected_loadout.json" not in script
    assert "export NODE_ID=" in script


def test_execute_script_is_candidate_scoped_and_fingerprints_selection():
    script = build_remote_script(
        CONFIG["nodes"][0],
        "run-2",
        execute_candidates=["abc123"],
        update_code=True,
    )
    assert "--candidate abc123" in script
    assert "git -C \"$REPO_DIR\" pull --ff-only" in script
    assert "selected_loadout.json" in script
    assert "model_inventory.selected.json" in script
    assert "--all" not in script


def test_execute_candidate_parser():
    assert parse_execute_candidates(["x1-370=abc", "x1-370=def"]) == {
        "x1-370": ["abc", "def"]
    }
    with pytest.raises(ValueError):
        parse_execute_candidates(["bad"])
