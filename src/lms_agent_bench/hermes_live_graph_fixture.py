#!/usr/bin/env python3
"""Isolated native Hermes MCP server backed by REAL disposable read-only Neo4j.

Explicit opt-in, exact scoped Docker ID required. Does not seed or create graph.
All tools other than schema and graph_query fail closed; every call is receipted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from lms_agent_bench.hermes_mcp_fixture import Fixture, serve, TOOLS
from lms_agent_bench.hermes_real_cypher_gate import (
    ScratchCypherGate, WitnessDenied,
)


# Preserve the nine-tool distractor-selection benchmark surface. Only the
# graph owner read and schema can succeed; every other tool fails closed.
LIVE_TOOLS = [
    {**tool, "inputSchema": {
        "type": "object", "properties": {
            "query": {"type": "string", "minLength": 1},
            "read_only": {"type": "boolean", "const": True},
        }, "required": ["query", "read_only"], "additionalProperties": False,
    }, "description": "Read-only owner lookup on isolated research Neo4j; supply read_only=true; parameters unsupported."}
    if tool["name"] == "graph_query" else tool
    for tool in TOOLS
]


class LiveCypherFixture(Fixture):
    tools = LIVE_TOOLS
    def __init__(self, state_dir, workspace, scenario, container_id):
        if scenario != "tool_distractor_selection":
            raise WitnessDenied("unsupported live-scoped Hermes scenario")
        super().__init__(state_dir, workspace, scenario)
        self.gate = ScratchCypherGate(expected_container_id=container_id)
        self.gate._checked_container()

    def call(self, name, arguments):
        if name == "graph_schema":
            result = {"nodes": ["Service", "Team"],
                      "relationships": ["OWNED_BY"], "read_only": True}
            self._record(name, arguments, True, result, False)
            return result, False
        if name not in {"graph_query"}:
            denial = {"error": "not enabled in isolated real-graph live research",
                      "retryable": False}
            self._record(name, arguments, False, denial, True)
            return denial, True
        return super().call(name, arguments)

    def _graph_query(self, arguments, call_number):
        if arguments.get("read_only") is not True:
            return {"error": "explicit read_only=true required", "retryable": False}, True
        try:
            proof = self.gate.check(
                arguments.get("query"),
                parameters=arguments.get("parameters"),
                read_only=True,
            )
        except WitnessDenied as exc:
            return {"error": str(exc), "retryable": False}, True
        if not proof.accepted:
            return {"error": proof.reason, "retryable": False}, True
        # These values were parsed from actual Neo4j stdout, never canned.
        idx = next(i for i, col in enumerate(proof.columns)
                   if col in {"owner", "s.owner", "n.owner", "t.name"})
        result = {
            "columns": ["owner"],
            "records": [{"owner": proof.records[0][idx]}],
            "summary": {
                "read_only": True, "records": 1,
                "source": "disposable_real_neo4j",
                "container_id": proof.container_id,
            },
        }
        # Separate native engine receipt: tie the actual cypher-shell exit,
        # parsed columns/rows and exact Docker ID to one MCP request/result.
        # A replayed canned fixture can never emit this physical witness.
        if hasattr(self, "_mcp_request_id"):
            if (getattr(proof, "execution_exit_code", None) != 0
                    or not isinstance(getattr(proof, "engine_stdout_sha256", None), str)
                    or len(proof.engine_stdout_sha256) != 64):
                return {"error": "missing zero-exit Neo4j engine stdout witness"}, True
            canonical = json.dumps(result, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode("utf-8")
            result_digest = hashlib.sha256(canonical).hexdigest()
            engine_receipt = {
                "schema": "lms-hermes-neo4j-engine-witness.v2",
                "mcp_request_id": self._mcp_request_id,
                "query": proof.query,
                "accepted": proof.accepted,
                "execution_exit_code": proof.execution_exit_code,
                "cypher_shell_stdout_sha256": proof.engine_stdout_sha256,
                "columns": list(proof.columns),
                "records": [list(row) for row in proof.records],
                "exact_container_id": proof.container_id,
                "mcp_result_sha256": result_digest,
                "timestamp_unix_ns": time.time_ns(),
            }
            with (self.state_dir / "neo4j-engine-witnesses.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(engine_receipt, sort_keys=True) + "\n")
        return result, False



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--research-container-id", required=True)
    parser.add_argument("--confirm-research-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_only:
        parser.error("research-only confirmation required")
    fixture = LiveCypherFixture(
        Path(args.state_dir), Path(args.workspace), args.scenario,
        args.research_container_id,
    )
    return serve(fixture)


if __name__ == "__main__":
    raise SystemExit(main())
