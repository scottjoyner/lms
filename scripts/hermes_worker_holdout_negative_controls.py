#!/usr/bin/env python3
"""Offline worker-api provenance tampering probes; copies only, no Docker/model."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from lms_agent_bench.hermes_native_provenance import audit_native_trial


def _report(trial: Path):
    return audit_native_trial(trial, service_name="worker-api")


def _alter_json(path: Path, mutate):
    obj = json.loads(path.read_text())
    mutate(obj)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")


def _alter_jsonl(path: Path, mutate):
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def evaluate(original: Path) -> list[tuple[str, bool, str]]:
    original = original.resolve(strict=True)
    if original.name != "trial-01" or original.parent.name != "mcp_tool_distractor_selection":
        raise ValueError("unexpected worker research native trial path")
    normal = _report(original)
    if not normal.get("native_complete_source_provenance"):
        raise ValueError("original worker native provenance not independently verified")
    results = [("genuine held-out source", True, "verified")]
    with tempfile.TemporaryDirectory(prefix="lms-worker-negative-") as td:
        for label in (
            "wrong service scorer", "fabricated MCP owner",
            "altered Neo4j engine receipt", "missing engine receipt",
            "substituted final answer",
        ):
            target = Path(td) / label / "trial-01"
            target.parent.mkdir()
            shutil.copytree(original, target)
            if label == "wrong service scorer":
                outcome = audit_native_trial(target, service_name="router-api")
            else:
                state = target / "fixture-state"
                if label == "fabricated MCP owner":
                    def change(rows):
                        assert len(rows) == 1 and rows[0]["result"]["records"] == [{"owner": "team-beta"}]
                        rows[0]["result"]["records"] = [{"owner": "team-atlas"}]
                    _alter_jsonl(state / "calls.jsonl", change)
                elif label == "altered Neo4j engine receipt":
                    def change(rows):
                        assert len(rows) == 1 and rows[0]["records"] == [["team-beta"]]
                        rows[0]["records"] = [["team-atlas"]]
                    _alter_jsonl(state / "neo4j-engine-witnesses.jsonl", change)
                elif label == "missing engine receipt":
                    (state / "neo4j-engine-witnesses.jsonl").unlink()
                else:
                    def change(obj):
                        assert "team-beta" in obj["final_response"]
                        obj["final_response"] = "The owner of worker-api is team-atlas."
                    _alter_json(target / "trial-report.json", change)
                outcome = _report(target)
            denied = not outcome.get("native_complete_source_provenance", False)
            if not denied:
                raise RuntimeError(f"unsafe provenance acceptance under {label}")
            results.append((label, False, str(outcome.get("denial") or "engine witness absent")))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-research-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_only:
        parser.error("explicit research-only opt-in required")
    if args.output.exists():
        raise FileExistsError("refuse to overwrite original evidence")
    results = evaluate(args.trial_dir)
    manifest = hashlib.sha256(
        (args.trial_dir / "native-independent-provenance.json").read_bytes()
    ).hexdigest()
    lines = ["# Native Hermes worker-api independent negative controls",
             "", f"Original native provenance receipt SHA256: `{manifest}`", "",
             "| Input | Expected | Observed |", "|---|---|---|"]
    for label, accepted, reason in results:
        lines.append(f"| {label} | {'PASS' if label.startswith('genuine') else 'DENY'} | "
                     f"{'PASS' if accepted else 'DENY'} |")
    lines.extend(["", "All negative controls were performed on temporary local copies.",
                  "Original evidence remained unchanged. No Neo4j, model inference, "
                  "credentials, production graph, routing, or deployment operations.", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print("NEGATIVE_CONTROLS", len(results) - 1, "DENIED; ORIGINAL PASS")


if __name__ == "__main__":
    main()
