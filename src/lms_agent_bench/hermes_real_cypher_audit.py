#!/usr/bin/env python3
"""Independent re-score of archived Hermes graph-read agent trials.

Consumes immutable trial-report.json and fixture-state/calls.jsonl.
Qualifying evidence requires real Neo4j owner rows plus a completed final
response. This module only reads files and invokes the separately validated
read-only ScratchCypherGate; it never creates or writes a graph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hermes_real_cypher_gate import ScratchCypherGate, WitnessDenied, verify_final_answer

CASE_KEY = "mcp_tool_distractor_selection"


def score_archived_trial(folder: Path, gate: ScratchCypherGate) -> dict[str, Any]:
    folder = folder.resolve()
    scorefile = folder / "trial-report.json"
    callsfile = folder / "fixture-state" / "calls.jsonl"
    if not scorefile.is_file() or not callsfile.is_file():
        raise ValueError("incomplete archived evidence")
    original_bytes = scorefile.read_bytes()
    call_bytes = callsfile.read_bytes()
    report = json.loads(original_bytes)
    calls = [json.loads(s) for s in call_bytes.decode().splitlines()]
    if report.get("case_key") != CASE_KEY:
        raise ValueError("unexpected Hermes case")
    if not isinstance(calls, list):
        raise ValueError("invalid tool receipt structure")
    results = []
    qualified = False
    valid_queries = 0
    timed_out = bool(report.get("timed_out"))
    final_response = str(report.get("final_response") or "")
    for record in calls:
        if record.get("tool") != "graph_query":
            continue
        args = record.get("arguments") or {}
        if not isinstance(args, dict):
            results.append({"accepted": False, "reason": "arguments are not an object"})
            continue
        query = args.get("query")
        if not isinstance(query, str):
            results.append({"accepted": False, "reason": "query is not a string"})
            continue
        try:
            proof = gate.check(
                query, parameters=args.get("parameters"),
                read_only=args.get("read_only") is True,
            )
            results.append(proof.as_dict())
            valid_queries += int(proof.accepted)
            qualified |= verify_final_answer(proof, final_response, timed_out=timed_out)
        except WitnessDenied as exc:
            results.append({"query": query, "accepted": False, "reason": str(exc)})
    completed = (
        qualified
        and bool(report.get("valid"))
        and not timed_out
        and report.get("process_returncode") == 0
        and bool(final_response.strip())
    )
    return {
        "schema": "lms-hermes-independent-real-graph-trial.v1",
        "case_key": CASE_KEY,
        "trial_index": report.get("trial_index"),
        "old_canned_fixture_pass": bool(report.get("passed")),
        "old_canned_fixture_valid": bool(report.get("valid")),
        "agent_timed_out": timed_out,
        "graph_attempt_count": len(results),
        "real_owner_query_count": valid_queries,
        "real_graph_and_final_answer_pass": completed,
        "real_proof_results": results,
        "source_sha256": {
            "trial_report": hashlib.sha256(original_bytes).hexdigest(),
            "fixture_calls": hashlib.sha256(call_bytes).hexdigest(),
        },
        "release_authorized": False,
    }


def score_archived_case(root: Path, gate: ScratchCypherGate) -> dict[str, Any]:
    root = root.resolve()
    trials = [score_archived_trial(root / f"trial-{i:02d}", gate) for i in (1, 2, 3)]
    if [x["trial_index"] for x in trials] != [1, 2, 3]:
        raise ValueError("missing or mislabeled trial receipts")
    accepted = sum(x["real_graph_and_final_answer_pass"] for x in trials)
    return {
        "schema": "lms-hermes-independent-real-graph-case.v1",
        "case_key": CASE_KEY,
        "trial_count": len(trials),
        "real_end_to_end_pass_count": accepted,
        "case_gate_passed": accepted == len(trials),
        "release_authorized": False,
        "trials": trials,
    }
