import subprocess
from pathlib import Path

import pytest

from lms_agent_bench import fleet_provenance


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.name", "Fleet Test")
    git(repo, "config", "user.email", "fleet@example.invalid")
    (repo / "README.md").write_text("fleet\n", encoding="utf-8")
    git(repo, "add", "README.md")
    git(repo, "commit", "-m", "initial")
    branch = git(repo, "branch", "--show-current")
    commit = git(repo, "rev-parse", "HEAD")
    return repo, branch, commit


def test_capture_and_verify_exact_clean_source(tmp_path):
    repo, branch, commit = make_repo(tmp_path)
    artifact = fleet_provenance.capture_source_control(
        repo_dir=str(repo),
        node_id="x1-370",
        run_id="run-1",
        expected_branch=branch,
        expected_commit=commit,
    )
    summary = fleet_provenance.verify_source_control(
        artifact,
        expected_node_id="x1-370",
        expected_run_id="run-1",
    )
    assert summary["commit"] == commit
    assert summary["branch"] == branch
    assert artifact["dirty"] is False
    assert artifact["admission"]["admitted"] is False


def test_capture_rejects_dirty_checkout(tmp_path):
    repo, branch, commit = make_repo(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        fleet_provenance.capture_source_control(
            repo_dir=str(repo),
            node_id="x1-370",
            run_id="run-1",
            expected_branch=branch,
            expected_commit=commit,
        )


def test_capture_rejects_commit_drift(tmp_path):
    repo, branch, _ = make_repo(tmp_path)
    with pytest.raises(ValueError, match="commit mismatch"):
        fleet_provenance.capture_source_control(
            repo_dir=str(repo),
            node_id="x1-370",
            run_id="run-1",
            expected_branch=branch,
            expected_commit="0" * 40,
        )


def test_verify_rejects_tampered_fingerprint(tmp_path):
    repo, branch, commit = make_repo(tmp_path)
    artifact = fleet_provenance.capture_source_control(
        repo_dir=str(repo),
        node_id="x1-370",
        run_id="run-1",
        expected_branch=branch,
        expected_commit=commit,
    )
    artifact["package_version"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        fleet_provenance.verify_source_control(
            artifact,
            expected_node_id="x1-370",
            expected_run_id="run-1",
        )
