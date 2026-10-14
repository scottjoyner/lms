"""Research-only native MCP real graph tool boundary tests (offline)."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lms_agent_bench.hermes_live_graph_fixture import LiveCypherFixture
from lms_agent_bench.hermes_real_cypher_gate import WitnessDenied

CID = "a" * 64


class FakeGate:
    def __init__(self, *, expected_container_id):
        assert expected_container_id == CID

    def _checked_container(self):
        return CID

    def check(self, query, *, parameters=None, read_only=False):
        if query != "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner":
            raise WitnessDenied("query denied")
        assert read_only is True and not parameters
        return SimpleNamespace(
            accepted=True,
            columns=("owner",), records=(("team-atlas",),),
            container_id=CID,
        )


@pytest.fixture
def fixture(tmp_path):
    with patch("lms_agent_bench.hermes_live_graph_fixture.ScratchCypherGate", FakeGate):
        yield LiveCypherFixture(
            tmp_path / "state", tmp_path / "workspace",
            "tool_distractor_selection", CID,
        )


def test_no_unverified_container_is_admitted(tmp_path):
    with pytest.raises(WitnessDenied):
        LiveCypherFixture(tmp_path / "state", tmp_path / "workspace",
                          "tool_distractor_selection", "wrong-id")


def test_wrong_scenario_denied(tmp_path):
    with pytest.raises(WitnessDenied):
        LiveCypherFixture(tmp_path / "state", tmp_path / "workspace",
                          "cross_tool_report", CID)


def test_graph_query_returns_bound_real_owner(fixture):
    result, is_error = fixture.call(
        "graph_query", {"query": "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner", "read_only": True},
    )
    assert not is_error
    assert result["columns"] == ["owner"]
    assert result["records"] == [{"owner": "team-atlas"}]
    assert result["summary"]["source"] == "disposable_real_neo4j"
    assert CID == result["summary"]["container_id"]
    assert "disposable_real_neo4j" in fixture.calls_path.read_text()


@pytest.mark.parametrize("arguments", [
    {"query": "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner"},
    {"query": "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner", "read_only": False},
    {"query": "MATCH (n) RETURN n LIMIT 50", "read_only": True},
])
def test_invalid_query_or_permission_denied(fixture, arguments):
    result, error = fixture.call("graph_query", arguments)
    assert error and "error" in result
    assert '"is_error": true' in fixture.calls_path.read_text()


@pytest.mark.parametrize("name", [
    "workspace_write", "workspace_patch", "workspace_read", "workspace_run_tests", "send_email", "graph_delete_all", "weather_lookup",
])
def test_other_tool_effects_not_available(fixture, name):
    result, error = fixture.call(name, {})
    assert error and "not enabled" in result["error"]
    assert not (fixture.workspace / "unsafe.txt").exists()


def test_graph_schema_remains_visible(fixture):
    result, err = fixture.call("graph_schema", {})
    assert err is False and result["read_only"] is True
    assert "OWNED_BY" in result["relationships"]
