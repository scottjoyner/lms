"""Research-only MCP advertised-contract/actual-dispatch consistency checks."""
import io
import json
from unittest.mock import patch

from lms_agent_bench.hermes_live_graph_fixture import LiveCypherFixture
from lms_agent_bench.hermes_mcp_fixture import Fixture, TOOLS, serve

CID = "a" * 64
QUERY = "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner"


class FakeGate:
    def __init__(self, *, expected_container_id):
        assert expected_container_id == CID

    def _checked_container(self):
        return CID


def make_live(tmp_path):
    with patch("lms_agent_bench.hermes_live_graph_fixture.ScratchCypherGate", FakeGate):
        return LiveCypherFixture(tmp_path / "state", tmp_path / "workspace",
                                 "tool_distractor_selection", CID)


def test_live_schema_does_not_mutate_canned_historical_contract(tmp_path):
    original = next(t for t in TOOLS if t["name"] == "graph_query")
    fixture = make_live(tmp_path)
    live = next(t for t in fixture.tools if t["name"] == "graph_query")
    assert original["inputSchema"]["required"] == ["query"]
    assert live["inputSchema"]["required"] == ["query", "read_only"]
    assert "read_only=true" in live["description"]

    missing, failed = fixture.call("graph_query", {"query": QUERY})
    assert failed and "read_only" in missing["error"]
    false, failed = fixture.call("graph_query", {"query": QUERY, "read_only": False})
    assert failed and "read_only" in false["error"]
    legacy = Fixture(tmp_path / "old-state", tmp_path / "old-work", "tool_distractor_selection")
    assert next(t for t in TOOLS if t["name"] == "graph_query")["inputSchema"]["required"] == ["query"]
    assert not hasattr(legacy, "tools")


def test_mcp_tools_list_reports_actual_live_dispatch_contract(tmp_path):
    fixture = make_live(tmp_path)
    stdin = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}) + "\n")
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        assert serve(fixture) == 0
    receipt = json.loads(stdout.getvalue().strip())
    exposed = next(t for t in receipt["result"]["tools"] if t["name"] == "graph_query")
    assert exposed["inputSchema"]["required"] == ["query", "read_only"]
    assert len(receipt["result"]["tools"]) == 9
