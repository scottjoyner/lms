#!/usr/bin/env python3
"""Independent, offline, metadata-only audit of paired native Hermes trials.

No graph, model, Docker, or production access. This is NOT model qualification:
even a 2/2 diagnostic pass never substitutes for the 3/3 native gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MODES = ("control", "minimal")
BRIDGES = ["tool_search", "tool_describe", "tool_call"]
GENERATION = {"temperature": 0.0, "seed": 43, "max_tokens": 640, "stream": True}
CASE = "mcp_tool_distractor_selection"


def _read(path: Path):
    return json.loads(path.read_text())


def _lines(path: Path):
    if not path.is_file():
        raise ValueError(f"missing required research receipt: {path.name}")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    if not rows or any(not isinstance(x, dict) for x in rows):
        raise ValueError(f"invalid or empty research receipt: {path.name}")
    return rows


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_pair(root: Path):
    root = root.resolve(strict=True)
    original = root / "native-research-summary.json"
    state = _read(original)
    if (state.get("research_only") is not True
            or state.get("release_authorized") is not False
            or state.get("qualification_authorized") is not False
            or state.get("research_pair_only") is not True
            or state.get("run_count_expected") != 2
            or state.get("research_modes") != list(MODES)
            or state.get("model_server_removed") is not True):
        raise ValueError("not a closed, safely torn-down two-condition research experiment")
    ids = state.get("owned_container_ids") or []
    if len(ids) != 2 or len(set(ids)) != 2 or any(
        not isinstance(i, str) or re.fullmatch(r"[0-9a-f]{64}", i) is None for i in ids
    ):
        raise ValueError("missing two distinct, scoped research Docker IDs")
    outcome = {
        "schema": "lms-hermes-native-preamble-pair-audit.v1",
        "source_summary_sha256": _sha(original),
        "research_only": True,
        "release_authorized": False,
        "qualification_authorized": False,
        "native_three_trial_gate_passed": False,
        "mode_order": list(MODES),
        "source_receipts": {},
        "trials": [],
    }
    wires, constructors = [], []
    for index, mode in enumerate(MODES, 1):
        trial = root / "trial-roots" / CASE / f"trial-{index:02d}"
        paths = {
            "report": trial / "trial-report.json",
            "custody": trial / "native-independent-provenance.json",
            "wire": trial / "fixture-state/model-http-receipts.jsonl",
            "constructor": trial / "fixture-state/model-request-ablation.jsonl",
            "calls": trial / "fixture-state/calls.jsonl",
        }
        if not all(p.is_file() for p in paths.values()):
            raise ValueError(f"missing source receipts for {mode}")
        report, custody = _read(paths["report"]), _read(paths["custody"])
        outbound, generated, calls = (
            _lines(paths["wire"]), _lines(paths["constructor"]),
            _lines(paths["calls"]),
        )
        wire = outbound[0].get("wire_surface")
        ctor = generated[0]
        if not isinstance(wire, dict) or ctor.get("mode") != mode:
            raise ValueError("model-facing first request mode/wire evidence missing")
        if (custody.get("scratch_container_id") != ids[index-1]
                or custody.get("teardown_verified") is not True
                or custody.get("exact_container_removed") is not True):
            raise ValueError("original scoped teardown custody incomplete")
        if wire.get("generation") != GENERATION:
            raise ValueError("unmatched model sampling at first request")
        if ([x.get("name") for x in wire.get("advertised_tool_contracts", [])]
                != BRIDGES):
            raise ValueError("changed original native three-bridge tool surface")
        if wire.get("model_message_roles") != ["system", "user"]:
            raise ValueError("unexpected first-turn message shape")
        if any(x.get("tool") in ("send_email", "graph_delete_all") for x in calls):
            raise ValueError("prohibited research side-effect tool invoked")
        record = {
            "trial": index,
            "mode": mode,
            "original_system_chars": ctor["original_system_chars"],
            "effective_system_chars": ctor["effective_system_chars"],
            "original_system_sha256": ctor["original_system_sha256"],
            "effective_system_sha256": ctor["effective_system_sha256"],
            "accepted_graph_query_count": sum(
                x.get("tool") == "graph_query" and x.get("is_error") is False
                for x in calls
            ),
            "rejected_graph_query_count": sum(
                x.get("tool") == "graph_query" and x.get("is_error") is True
                for x in calls
            ),
            "model_http_request_count": len(outbound),
            "native_answer_verified": custody.get("native_transcript_http_correlated") is True,
            "engine_answer_verified": custody.get("native_complete_source_provenance") is True,
            "timed_out": report.get("timed_out") is True,
            "runner_valid": report.get("valid") is True,
            "harness_exception": str(report.get("final_response") or "").startswith(
                "inconsistent original model sampling constraint"
            ),
            "original_teardown_reported": True,
        }
        wires.append(wire)
        constructors.append(ctor)
        outcome["trials"].append(record)
        outcome["source_receipts"][mode] = {
            key: _sha(path) for key, path in paths.items()
        }
    a, b = wires
    matched = (
        a.get("user_content_sha256") == b.get("user_content_sha256")
        and a.get("advertised_tool_contracts") == b.get("advertised_tool_contracts")
        and a.get("all_other_options_sha256") == b.get("all_other_options_sha256")
        and a.get("model_message_roles") == b.get("model_message_roles")
        and a.get("generation") == b.get("generation")
        and a.get("system_content_sha256") != b.get("system_content_sha256")
    )
    outcome["actual_first_wire_only_system_differs"] = bool(matched)
    outcome["original_system_prompt_drifted"] = (
        constructors[0].get("original_system_sha256")
        != constructors[1].get("original_system_sha256")
    )
    outcome["causal_preamble_effect_established"] = False
    outcome["native_verified_success_count"] = sum(
        t["native_answer_verified"] for t in outcome["trials"]
    )
    outcome["engine_verified_success_count"] = sum(
        t["engine_answer_verified"] for t in outcome["trials"]
    )
    outcome["harness_exception_count"] = sum(
        t["harness_exception"] for t in outcome["trials"]
    )
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; never overwrite original trial evidence")
    result = score_pair(args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "actual_first_wire_only_system_differs", "original_system_prompt_drifted",
        "native_verified_success_count", "engine_verified_success_count",
        "harness_exception_count", "native_three_trial_gate_passed",
    )}, sort_keys=True))
    return 3  # Diagnostic pair never qualifies models or releases.


if __name__ == "__main__":
    raise SystemExit(main())
