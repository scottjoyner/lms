#!/usr/bin/env python3
"""Isolated native Hermes MCP server backed by REAL disposable read-only Neo4j.

Explicit opt-in, exact scoped Docker ID required. Does not seed or create graph.
All tools other than schema and graph_query fail closed; every call is receipted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from lms_agent_bench.hermes_mcp_fixture import Fixture, TOOLS, serve
from lms_agent_bench.hermes_real_cypher_gate import (
    ScratchCypherGate, WitnessDenied,
)


class LiveCypherFixture(Fixture):
    def __init__(self, state_dir, workspace, scenario, container_id):
        if scenario != "tool_distractor_selection":
            raise WitnessDenied("unsupported live-scoped Hermes scenario")
        super().__init__(state_dir, workspace, scenario)
        # Contract applies only to the live, independently verified graph tool.
        # Historical canned fixtures keep their original advertised schema.
        self.tools = deepcopy(TOOLS)
        graph = next(t for t in self.tools if t["name"] == "graph_query")
        schema = graph["inputSchema"]
        schema["required"] = ["query", "read_only"]
        graph["description"] = (
            "Run a read-only Cypher owner lookup against disposable Neo4j. "
            "Required: query and read_only=true. Bind to the named :Service "
            "router-api and return an owner value, not node objects. "
            "Research gate denies parameters, writes, broad scans, and "
            "unsupported query shapes."
        )
        self.gate = ScratchCypherGate(expected_container_id=container_id)
        self.gate._checked_container()

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

    def call(self, name, arguments):
        if name not in {"graph_schema", "graph_query"}:
            # Do not execute workspace writes, external actions or commands.
            denial = {"error": "not enabled in isolated real-graph live research",
                      "retryable": False}
            self._record(name, arguments, False, denial, True)
            return denial, True
        return super().call(name, arguments)


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
