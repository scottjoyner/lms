#!/usr/bin/env python3
"""Isolated native Hermes MCP server backed by REAL disposable read-only Neo4j.

Explicit opt-in, exact scoped Docker ID required. Does not seed or create graph.
All tools other than schema and graph_query fail closed; every call is receipted.
"""
from __future__ import annotations

import argparse
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
