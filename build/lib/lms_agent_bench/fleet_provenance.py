#!/usr/bin/env python3
"""Capture and verify exact source-control provenance for physical fleet runs."""
from __future__ import annotations

import argparse
import hashlib
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import lms_agent_bench
from lms_agent_bench.fleet_loadout import canonical_hash, utc_now_iso, write_json

SCHEMA_VERSION = "fleet_source_control.v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _git(repo: Path, args: Sequence[str], timeout_s: int = 20) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "git command failed"
        raise ValueError(f"git {' '.join(args)}: {detail}")
    return proc.stdout.strip()


def _origin_fingerprint(repo: Path) -> Optional[str]:
    try:
        value = _git(repo, ["remote", "get-url", "origin"])
    except ValueError:
        return None
    if not value:
        return None
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def capture_source_control(
    repo_dir: str,
    node_id: str,
    run_id: str,
    expected_branch: str,
    expected_commit: str,
) -> Dict[str, Any]:
    repo = Path(repo_dir).expanduser().resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"not a git repository: {repo}")
    expected_commit = expected_commit.strip().lower()
    if not COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("expected_commit must be a full 40-character Git SHA-1")

    actual_branch = _git(repo, ["branch", "--show-current"])
    actual_commit = _git(repo, ["rev-parse", "HEAD"]).lower()
    status = _git(repo, ["status", "--porcelain=v1", "--untracked-files=all"])
    dirty = bool(status)

    if actual_branch != expected_branch:
        raise ValueError(
            f"source branch mismatch: expected {expected_branch}, found "
            f"{actual_branch or '<detached>'}"
        )
    if actual_commit != expected_commit:
        raise ValueError(
            f"source commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
    if dirty:
        raise ValueError("source checkout is dirty; refuse physical evidence capture")

    core = {
        "node_id": node_id,
        "run_id": run_id,
        "expected_branch": expected_branch,
        "actual_branch": actual_branch,
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "dirty": False,
        "origin_fingerprint": _origin_fingerprint(repo),
        "python_version": platform.python_version(),
        "package_version": lms_agent_bench.__version__,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "source_control_provenance",
        "captured_at_utc": utc_now_iso(),
        **core,
        "source_fingerprint": canonical_hash(core),
        "admission": {"admitted": False},
    }


def verify_source_control(
    artifact: Mapping[str, Any],
    expected_node_id: str,
    expected_run_id: str,
) -> Dict[str, Any]:
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source-control provenance schema")
    if str(artifact.get("node_id")) != expected_node_id:
        raise ValueError("source provenance node_id does not match rollout result")
    if str(artifact.get("run_id")) != expected_run_id:
        raise ValueError("source provenance run_id does not match rollout result")
    if artifact.get("dirty") is not False:
        raise ValueError("source provenance does not prove a clean checkout")

    expected_branch = str(artifact.get("expected_branch") or "")
    actual_branch = str(artifact.get("actual_branch") or "")
    expected_commit = str(artifact.get("expected_commit") or "").lower()
    actual_commit = str(artifact.get("actual_commit") or "").lower()
    if not expected_branch or actual_branch != expected_branch:
        raise ValueError("source provenance branch mismatch")
    if not COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("source provenance expected commit is invalid")
    if actual_commit != expected_commit:
        raise ValueError("source provenance commit mismatch")

    source_fp = str(artifact.get("source_fingerprint") or "")
    core = {
        "node_id": str(artifact.get("node_id")),
        "run_id": str(artifact.get("run_id")),
        "expected_branch": expected_branch,
        "actual_branch": actual_branch,
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "dirty": False,
        "origin_fingerprint": artifact.get("origin_fingerprint"),
        "python_version": artifact.get("python_version"),
        "package_version": artifact.get("package_version"),
    }
    if source_fp != canonical_hash(core):
        raise ValueError("source provenance fingerprint mismatch")
    admission = artifact.get("admission")
    if not isinstance(admission, Mapping) or admission.get("admitted") is not False:
        raise ValueError("source provenance must remain non-admitted")
    return {
        "source_fingerprint": source_fp,
        "branch": actual_branch,
        "commit": actual_commit,
        "package_version": artifact.get("package_version"),
        "python_version": artifact.get("python_version"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture exact clean-tree source provenance for a fleet run"
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = capture_source_control(
        repo_dir=args.repo,
        node_id=args.node_id,
        run_id=args.run_id,
        expected_branch=args.expected_branch,
        expected_commit=args.expected_commit,
    )
    write_json(args.out, artifact)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
