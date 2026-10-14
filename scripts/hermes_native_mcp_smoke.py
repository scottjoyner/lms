#!/usr/bin/env python3
"""One opt-in physical research-only Neo4j/MCP smoke; NO Hermes inference.

Refuses to overwrite any evidence folder. ScratchHermesArchive owns the
container and verifies custody before removal. Never touches production Neo4j.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive

QUERY = "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-research-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_research_only:
        parser.error("research-only opt-in required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    state = output_dir / "state"
    workspace = output_dir / "workspace"
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "graph_schema", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "graph_query",
                    "arguments": {"query": QUERY, "read_only": True}}},
    ]
    result = {"research_only": True, "release_authorized": False,
              "native_agent_test": False, "passed": False}
    scratch = ScratchHermesArchive()
    try:
        with scratch:
            result["exact_container_id"] = scratch.container_id
            result["fixed_fixture_readback_sha256"] = hashlib.sha256(
                scratch.readback.encode()).hexdigest()
            command = [
                sys.executable, "-m", "lms_agent_bench.hermes_live_graph_fixture",
                "--state-dir", str(state), "--workspace", str(workspace),
                "--scenario", "tool_distractor_selection",
                "--research-container-id", scratch.container_id,
                "--confirm-research-only",
            ]
            proc = subprocess.run(
                command, input="".join(json.dumps(q) + "\n" for q in requests),
                capture_output=True, text=True, timeout=45, check=False,
                env={**os.environ, "NO_COLOR": "1"},
            )
            if proc.returncode != 0:
                raise RuntimeError("isolated MCP bridge did not exit successfully")
            lines = [json.loads(line) for line in proc.stdout.splitlines()]
            if [line["id"] for line in lines] != [1, 2, 3, 4]:
                raise RuntimeError("MCP request sequence mismatch")
            expected = lines[-1]["result"]
            payload = expected["structuredContent"]
            if (expected["isError"] or payload["records"] != [{"owner": "team-atlas"}]
                    or payload["summary"]["container_id"] != scratch.container_id):
                raise RuntimeError("real-Neo4j owner reply failed")
            tool_names = [
                item["name"] for item in lines[1]["result"]["tools"]
            ]
            if len(tool_names) != 9:
                raise RuntimeError("original nine-tool distractor surface changed")
            calls = [json.loads(x) for x in (state / "calls.jsonl").read_text().splitlines()]
            wire = [json.loads(x) for x in
                    (state / "mcp-wire-responses.jsonl").read_text().splitlines()]
            if [x["mcp_request_id"] for x in calls] != [3, 4]:
                raise RuntimeError("missing call-to-MCP request IDs")
            if [x["mcp_request_id"] for x in wire] != [3, 4]:
                raise RuntimeError("missing flushed MCP wire receipts")
            from lms_agent_bench.hermes_native_provenance import _hash_json

            engine = [json.loads(x) for x in
                      (state / "neo4j-engine-witnesses.jsonl").read_text().splitlines()]
            if (len(engine) != 1 or engine[0].get("schema") !=
                    "lms-hermes-neo4j-engine-witness.v2"
                    or not isinstance(engine[0].get("cypher_shell_stdout_sha256"), str)
                    or len(engine[0]["cypher_shell_stdout_sha256"]) != 64
                    or engine[0].get("mcp_request_id") != 4
                    or engine[0].get("query") != QUERY
                    or engine[0].get("execution_exit_code") != 0
                    or engine[0].get("accepted") is not True
                    or engine[0].get("records") != [["team-atlas"]]
                    or engine[0].get("exact_container_id") != scratch.container_id
                    or engine[0].get("mcp_result_sha256") != _hash_json(payload)
                    or not engine[0].get("timestamp_unix_ns", 0)
                    < wire[1]["timestamp_unix_ns"]):
                raise RuntimeError("native Neo4j engine receipt failed independent check")
            result["neo4j_engine_receipt_verified"] = True
            result["graph_owner_verified"] = True
            result["nine_tools_advertised"] = True
            result["mcp_request_ids_linked"] = True
            result["stdout_sha256"] = hashlib.sha256(proc.stdout.encode()).hexdigest()
        if not scratch.destroyed:
            raise RuntimeError("scratch cleanup unverified")
        check = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", scratch.container_id],
            timeout=12, text=True, capture_output=True, check=False,
        )
        if check.returncode == 0:
            raise RuntimeError("exact scratch Docker ID still exists after teardown")
        result["exact_container_removed"] = True
        result["passed"] = True
    except Exception as exc:
        result["denial"] = str(exc)
        result["exact_container_removed"] = bool(scratch.destroyed)
    (output_dir / "physical-smoke-report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
