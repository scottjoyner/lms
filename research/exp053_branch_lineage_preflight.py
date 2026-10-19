#!/usr/bin/env python3
"""Read-only EXP-053 Git history *negative* preflight. Never merge or admit."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "EXP053-BRANCH-LINEAGE-NEGATIVE-PREFLIGHT-SPEC-2026-10-19.md"
SPEC_SHA256 = "ec5805ba4b51d2d318250d61fea0f184ed07bf8ff75a199163c270b978a53fe8"
SCHEMA = "exp053-branch-lineage-negative-preflight-v1"
MAIN_REF = "refs/remotes/origin/main"
PR32_REF = "refs/remotes/origin/research/exp053-race-human-review-gate-20261016"
PR32_BASE_REF = "refs/remotes/origin/research/k2-minicpm-v47-exp052-20261016"
HANDOFF = (
    "research/exp053_custodian_handoff_packet.py",
    "research/EXP053-INDEPENDENT-CUSTODIAN-REQUEST-2026-10-19.md",
    "research/EXP053-CUSTODY-BOUNDARY-NEGATIVE-AUDIT-SPEC-2026-10-18.md",
)


class Hold(ValueError):
    """No input-dependent values in exception text."""


def _frozen():
    if SPEC.is_symlink() or not SPEC.is_file() or hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256:
        raise Hold("frozen lineage preflight contract unavailable")


def _git(repo, *args, ok=(0,)):
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True,
            text=True, errors="replace", timeout=10, env=env, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Hold("read-only Git inspection unavailable") from exc
    if proc.returncode not in ok or len(proc.stdout) > 1024 * 1024:
        raise Hold("read-only Git inspection refused")
    return proc


def _ref(repo, ref):
    result = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}", ok=(0, 128))
    if result.returncode != 0:
        return None
    candidate = result.stdout.strip()
    if len(candidate) != 40 or any(c not in "0123456789abcdef" for c in candidate):
        raise Hold("unrecognized repository object id")
    return candidate


def _lineage(repo, src, dst):
    if src is None or dst is None:
        return "UNRESOLVED"
    result = _git(repo, "merge-base", src, dst, ok=(0, 1))
    return "DISCONNECTED" if result.returncode == 1 else "CONNECTED"


def _tree_delta(repo, src, dst):
    if src is None or dst is None:
        return None
    result = _git(repo, "diff-tree", "--no-renames", "--name-status", "-r", src, dst)
    counts = {name: 0 for name in "AMDT"}
    for line in result.stdout.splitlines():
        status, sep, path = line.partition("\t")
        if not sep or not path or status not in counts:
            raise Hold("unexpected tree delta encoding")
        counts[status] += 1
    return counts


def _handoff_presence(repo, rev):
    if rev is None:
        return []
    return [path for path in HANDOFF
            if _git(repo, "cat-file", "-e", f"{rev}:{path}", ok=(0, 1, 128)).returncode == 0]


def inspect(repo):
    _frozen()
    repo = Path(repo)
    if not repo.is_absolute() or repo.is_symlink() or not repo.is_dir():
        raise Hold("repository must be an existing absolute non-symlink directory")
    sources = {"head": "HEAD", "origin_main": MAIN_REF, "pr32": PR32_REF}
    refs = {name: _ref(repo, ref) for name, ref in sources.items()}
    if refs["head"] is None:
        raise Hold("HEAD is not a valid commit")
    tracked = _git(repo, "status", "--porcelain=v1", "--untracked-files=no")
    return {
        "schema": SCHEMA,
        "references": {name: "FOUND" if val else "MISSING" for name, val in refs.items()},
        "lineage": {
            "main": _lineage(repo, refs["head"], refs["origin_main"]),
            "pr32": _lineage(repo, refs["head"], refs["pr32"]),
        },
        "tree_delta_counts": {
            "main": _tree_delta(repo, refs["origin_main"], refs["head"]),
            "pr32": _tree_delta(repo, refs["pr32"], refs["head"]),
        },
        "source_only_handoff_file_presence": {
            "local": _handoff_presence(repo, refs["head"]),
            "pr32": _handoff_presence(repo, refs["pr32"]),
        },
        "tracked_worktree_modification_count": len(tracked.stdout.splitlines()),
        "target_pr_base_ref_status": "FOUND" if _ref(repo, PR32_BASE_REF) else "MISSING",
        "integration_authorized": False,
        "scientific_execution_authorized": False,
        "decision": "NO_GO_RESEARCH_ONLY",
        "next_action": "HUMAN_REVIEW_SOURCE_ONLY_DELTA_WITH_CORRECT_ANCESTRY",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path,
                        help="existing absolute local repository path (read-only)")
    args = parser.parse_args(argv)
    try:
        report = inspect(args.repo)
    except (Hold, OSError, ValueError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"integration_authorized":false,"status":"UNVERIFIABLE_REPOSITORY_LINEAGE"}')
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
