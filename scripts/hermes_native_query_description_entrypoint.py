#!/usr/bin/env python3
"""Tool description A/B native Hermes trial, disposable real-Neo4j only.

No global Hermes config, production graph, or credentials are changed. The
parent passes an exact Docker ID and an isolated per-trial research directory.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from lms_agent_bench import hermes_agent_bench as bench
from lms_agent_bench import hermes_agent_secure_entrypoint as secure
from lms_agent_bench.hermes_live_graph_fixture import LIVE_TOOLS
from lms_agent_bench.hermes_real_cypher_gate import ScratchCypherGate

_ORIGINAL_TRIAL = bench.trial_process
_EXPECTED = {"mcp__lms_benchmark__" + item["name"] for item in LIVE_TOOLS}


def scoped_trial(args):
    root = Path(os.environ["LMS_HERMES_RESEARCH_ROOT"]).resolve(strict=True)
    home = Path(args.hermes_home).resolve()
    workspace = Path(args.workspace).resolve()
    state = Path(args.state_dir).resolve()
    for path in (home, workspace, state):
        if root not in path.parents:
            raise RuntimeError("native trial path outside isolated research root")
    if os.environ.get("LMS_HERMES_NATIVE_PROVENANCE_RESEARCH") != "1":
        raise RuntimeError("native HTTP witness opt-in is mandatory")
    research_id = os.environ.get("LFM25_DISPOSABLE_NEO4J_ID", "")
    ScratchCypherGate(expected_container_id=research_id)._checked_container()
    fixture = Path(__file__).resolve().parents[1] / "src/lms_agent_bench/hermes_live_graph_fixture.py"
    if not fixture.is_file():
        raise RuntimeError("pinned real-graph fixture unavailable")

    original_writer = bench.write_trial_config

    def configure(home, state, workspace, scenario):
        original_writer(home, state, workspace, scenario)
        config = home / "config.yaml"
        value = json.loads(config.read_text())
        server = value["mcp_servers"]["lms-benchmark"]
        existing = [index for index, arg in enumerate(server["args"])
                    if Path(arg).name == "hermes_mcp_fixture.py"]
        if len(existing) != 1:
            raise RuntimeError("missing canonical benchmark fixture")
        server["args"][existing[0]] = str(fixture)
        server["args"].extend([
            "--research-container-id", research_id,
            "--confirm-research-only",
        ])
        server["env"] = {"PYTHONPATH": str(fixture.parents[1])}
        config.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    bench.write_trial_config = configure
    sys.path.insert(0, str(Path(args.hermes_repo).resolve()))
    try:
        from tools.mcp_tool import discover_mcp_tools, shutdown_mcp_servers
        from model_tools import get_tool_definitions

        configure(home, state, workspace, "tool_distractor_selection")
        os.environ["HERMES_HOME"] = str(home)
        try:
            discovered = set(discover_mcp_tools())
            definitions = get_tool_definitions(
                enabled_toolsets=["mcp-lms-benchmark"], quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            visible = {
                item["function"]["name"] for item in definitions
                if isinstance(item, dict) and "function" in item
            }
            if visible != _EXPECTED or not _EXPECTED.issubset(discovered):
                raise RuntimeError("exact nine-tool MCP surface failed preflight")
            named = next(d for d in definitions
                         if d["function"]["name"] == "mcp__lms_benchmark__graph_query")
            properties = named["function"]["parameters"].get("properties", {})
            if not {"query", "read_only"}.issubset(properties):
                raise RuntimeError("native graph_query lost required read-only fields")
            (state / "preflight-tool-inventory.json").write_text(
                json.dumps({"tools": sorted(visible), "passed": True}, indent=2) + "\n"
            )
            mode = os.environ.get("LMS_HERMES_QUERY_DESCRIPTION_RESEARCH")
            if mode not in ("baseline", "example"):
                raise RuntimeError("mandatory isolated presentation arm missing")
            from agent import chat_completion_helpers as helpers
            from lms_agent_bench.hermes_native_query_description import adapt_description
            import model_tools
            import run_agent
            pinned = Path(os.environ["LMS_HERMES_PINNED_PREAMBLE_FILE"])
            if pinned.parent.resolve() != root.parent.resolve():
                raise RuntimeError("preamble staging outside scratch research scope")
            original_builder = helpers.build_api_kwargs
            original_get_defs = model_tools.get_tool_definitions
            original_run_defs = run_agent.get_tool_definitions

            if mode in ("baseline", "example"):
                def visible_direct(*arguments, **kw):
                    if kw.get("enabled_toolsets") == ["mcp-lms-benchmark"]:
                        kw = dict(kw)
                        kw["skip_tool_search_assembly"] = True
                    return original_get_defs(*arguments, **kw)
                model_tools.get_tool_definitions = visible_direct
                run_agent.get_tool_definitions = visible_direct

            tool_visibility = run_agent.get_tool_definitions(
                enabled_toolsets=["mcp-lms-benchmark"], quiet_mode=True,
            )
            names = [t["function"]["name"] for t in tool_visibility]
            allowed = _EXPECTED
            if len(names) != len(allowed) or set(names) != allowed:
                raise RuntimeError("unexpected native research presentation toolset")

            def scoped_build(agent, messages):
                outgoing = original_builder(agent, messages)
                transformed, receipt = adapt_description(
                    outgoing, mode=mode, pinned=pinned,
                )
                with (state / "query-description-request-receipts.jsonl").open(
                    "a", encoding="utf-8"
                ) as output:
                    output.write(json.dumps(receipt, sort_keys=True) + "\n")
                return transformed

            helpers.build_api_kwargs = scoped_build
            try:
                return _ORIGINAL_TRIAL(args)
            finally:
                helpers.build_api_kwargs = original_builder
                model_tools.get_tool_definitions = original_get_defs
                run_agent.get_tool_definitions = original_run_defs
        finally:
            shutdown_mcp_servers()
    finally:
        bench.write_trial_config = original_writer


def main():
    secure.apply_patches()
    bench.trial_process = scoped_trial
    return secure.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
