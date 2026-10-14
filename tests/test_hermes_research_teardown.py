"""Offline negative teardown checks; never invokes real Docker."""

import subprocess
from pathlib import Path
from runpy import run_path

import pytest

_VERIFY = run_path(str(
    Path(__file__).resolve().parents[1] / "scripts/hermes_native_three_trial_research.py"
))
_verify_exact_container_removed = _VERIFY["_verify_exact_container_removed"]

CID = "a" * 64


def test_exact_owned_container_absence_requires_healthy_docker(monkeypatch):
    operations = []

    def fake(args, **kwargs):
        operations.append(args)
        if args[1] == "info":
            return subprocess.CompletedProcess(args, 0, "29.8.2\n", "")
        assert args[1] == "inspect" and args[-1] == CID
        return subprocess.CompletedProcess(args, 1, "", f"error: no such object: {CID}\n")

    monkeypatch.setattr(subprocess, "run", fake)
    receipt = _verify_exact_container_removed(CID)
    assert receipt == {
        "docker_daemon_healthy": True,
        "docker_inspect_exit_code": 1,
        "exact_container_absent": True,
    }
    assert [op[1] for op in operations] == ["info", "inspect"]


@pytest.mark.parametrize("daemon_exit, inspect_exit, stderr", [
    (1, 1, f"error: no such object: {CID}"),
    (0, 0, ""),
    (0, 2, "connection refused"),
    (0, 1, "failed to access Docker socket"),
    (0, 1, "error: no such object: " + "b" * 64),
])
def test_broken_docker_or_mismatched_absence_never_passes(
    monkeypatch, daemon_exit, inspect_exit, stderr,
):
    def fake(args, **kwargs):
        if args[1] == "info":
            return subprocess.CompletedProcess(
                args, daemon_exit, "29.8.2\n" if daemon_exit == 0 else "", "down",
            )
        return subprocess.CompletedProcess(args, inspect_exit, "", stderr)

    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError):
        _verify_exact_container_removed(CID)


def test_bad_container_identifier_refused_before_docker(monkeypatch):
    def bad(*args, **kwargs):
        raise AssertionError("should not inspect Docker for malformed identity")

    monkeypatch.setattr(subprocess, "run", bad)
    with pytest.raises(RuntimeError, match="invalid exact"):
        _verify_exact_container_removed("unsafe-name")


def test_three_trial_acceptance_requires_independent_engine_and_custody():
    _research_three_trial_accepted = _VERIFY["_research_three_trial_accepted"]

    def source_trial():
        return {
            "native_complete_source_provenance": True,
            "teardown_verified": True,
            "exact_container_removed": True,
        }

    good = {
        "trials": [source_trial() for _ in range(3)],
        "owned_container_ids": ["a" * 64, "b" * 64, "c" * 64],
        "native_correlated_count": 3,
        "engine_provenance_count": 3,
        "model_server_removed": True,
    }
    assert _research_three_trial_accepted(good)
    from copy import deepcopy

    missing_engine = deepcopy(good)
    missing_engine["engine_provenance_count"] = 2
    assert not _research_three_trial_accepted(missing_engine)
    swapped_graph = deepcopy(good)
    swapped_graph["owned_container_ids"][1] = "a" * 64
    assert not _research_three_trial_accepted(swapped_graph)
    unverified_teardown = deepcopy(good)
    unverified_teardown["trials"][0]["teardown_verified"] = False
    assert not _research_three_trial_accepted(unverified_teardown)
    legacy_only = deepcopy(good)
    legacy_only["trials"][1]["native_complete_source_provenance"] = False
    assert not _research_three_trial_accepted(legacy_only)
    remaining_server = deepcopy(good)
    remaining_server["model_server_removed"] = False
    assert not _research_three_trial_accepted(remaining_server)
    assert not _research_three_trial_accepted({**good, "trials": good["trials"][:2]})
