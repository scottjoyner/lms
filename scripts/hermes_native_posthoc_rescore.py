#!/usr/bin/env python3
"""Independent offline posthoc rescore of one completed native Hermes research case.

Read-only: never starts models, contacts Neo4j, or modifies prior trial evidence.
The original v1 qualification report remains unchanged; the posthoc report
is written to a *new* path. A live container existence check is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from lms_agent_bench.hermes_native_provenance import audit_native_trial

CASE = "mcp_tool_distractor_selection"
ID_RE = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_existing_run(root: Path) -> dict:
    root = root.resolve(strict=True)
    original = root / "native-research-summary.json"
    if not original.is_file():
        raise ValueError("original native research summary missing")
    before = json.loads(original.read_text())
    if before.get("run_count_expected") != 3 or len(before.get("trials", [])) != 3:
        raise ValueError("run does not contain exactly three original trials")
    if before.get("research_only") is not True:
        raise ValueError("original run not marked research-only")

    outcome = {
        "schema": "lms-hermes-pr18-posthoc-native-provenance-v2",
        "evidence_path": str(root),
        "original_summary_sha256": sha256(original),
        "original_summary_native_success_count": before.get("native_correlated_count"),
        "correction_reason": (
            "Original offline checker expected direct MCP JSON; native Hermes "
            "used the tool_call bridge plus exact untrusted_tool_result wrapper."
        ),
        "original_evidence_overwritten": False,
        "research_only": True,
        "case_gate_passed": False,
        "qualification_authorized": False,
        "release_authorized": False,
        "model_server_teardown_reported": before.get("model_server_removed") is True,
        "trial_count": 3,
        "trials": [],
    }
    base = root / "trial-roots" / CASE
    for index in (1, 2, 3):
        trial = base / f"trial-{index:02d}"
        if not trial.is_dir():
            raise ValueError(f"native trial {index} missing")
        old_path = trial / "native-independent-provenance.json"
        report_path = trial / "trial-report.json"
        if not old_path.is_file() or not report_path.is_file():
            raise ValueError(f"original native trial {index} lacks immutable evidence")
        old = json.loads(old_path.read_text())
        report = json.loads(report_path.read_text())
        if report.get("trial_index") != index:
            raise ValueError(f"trial {index} identity mismatch")
        if old.get("exact_container_removed") is not True:
            raise ValueError(f"trial {index} missing original teardown proof")
        updated = audit_native_trial(trial)
        container_id = None
        for record in (trial / "fixture-state" / "calls.jsonl").read_text().splitlines():
            parsed = json.loads(record)
            summary = (parsed.get("result") or {}).get("summary") or {}
            candidate = summary.get("container_id")
            if isinstance(candidate, str) and ID_RE.fullmatch(candidate):
                container_id = candidate
                break
        # The original research runner verified teardown at physical run time.
        # Fresh Docker inspection is supplementary; daemon errors remain UNKNOWN.
        now_absent = None
        if container_id:
            try:
                current = subprocess.run(
                    ["docker", "inspect", "--format", "{{.Id}}", container_id],
                    capture_output=True, text=True, timeout=12, check=False,
                )
                now_absent = current.returncode != 0 and not current.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                now_absent = None
        outcome["trials"].append({
            "index": index,
            "trial_report_sha256": sha256(report_path),
            "original_audit_sha256": sha256(old_path),
            "original_audit_native_success": old.get("native_transcript_http_correlated"),
            "original_runner_valid": old.get("runner_valid"),
            "timed_out": report.get("timed_out"),
            "original_teardown_verified": True,
            "current_container_absent_from_docker": now_absent,
            "native_evidence": updated,
        })
    outcome["posthoc_native_success_count"] = sum(
        trial["native_evidence"]["native_transcript_http_correlated"] is True
        for trial in outcome["trials"]
    )
    outcome["physical_three_trial_acceptance"] = (
        outcome["posthoc_native_success_count"] == 3
        and outcome["model_server_teardown_reported"]
        and all(t["original_teardown_verified"] for t in outcome["trials"])
    )
    # Even if a future run passes 3/3, review, PR merge and routing remain
    # separate decisions that this script cannot authorize.
    outcome["case_gate_passed"] = False
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists: never overwrite historical evidence")
    outcome = score_existing_run(args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "trial_count": outcome["trial_count"],
        "posthoc_native_success_count": outcome["posthoc_native_success_count"],
        "original_summary_native_success_count": outcome["original_summary_native_success_count"],
        "case_gate_passed": outcome["case_gate_passed"],
        "release_authorized": outcome["release_authorized"],
    }, sort_keys=True))
    return 0 if outcome["physical_three_trial_acceptance"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
