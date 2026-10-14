"""Negative controls for synthetic Hermes Cypher responses.

These checks prevent false-positive benchmark effects. They do not purport to
replace Neo4j parsing or prove execution semantics for arbitrary Cypher.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lms_agent_bench.hermes_mcp_fixture import (
    Fixture,
    validate_read_only_benchmark_cypher,
)


OWNER_QUERY = (
    "MATCH (s:Service {name: $service})-[:OWNED_BY]->(t:Team) "
    "RETURN s.name AS service, t.name AS owner, s.region AS region, s.status AS status"
)
DUPLICATE_QUERY = (
    "MATCH (d:Device) WITH d.key AS duplicate_key, count(*) AS count "
    "WHERE count > 1 RETURN duplicate_key AS duplicate_key, count AS count"
)
DEPENDENCY_QUERY = (
    "MATCH (s:Service {name: $service})-[:DEPENDS_ON]->(d:Service) "
    "RETURN s.name AS service, d.name AS dependency, "
    "d.signal AS signal, d.severity AS severity"
)
INCIDENT_QUERY = (
    "MATCH (i:Incident) RETURN i.id AS incident_id, "
    "i.service AS service, i.severity AS severity"
)


@pytest.mark.parametrize(
    ("scenario", "query", "parameters"),
    [
        ("service_owner_lookup", OWNER_QUERY, {"service": "router-api"}),
        ("tool_distractor_selection", OWNER_QUERY, {"service": "router-api"}),
        ("dependency_risk", DEPENDENCY_QUERY, {"service": "router-api"}),
        ("transient_query_recovery", DEPENDENCY_QUERY, {"service": "router-api"}),
        ("safe_read_only_boundary", DUPLICATE_QUERY, {}),
        ("cross_tool_report", INCIDENT_QUERY, {}),
    ],
)
def test_valid_benchmark_read_queries_only_unlock_synthetic_fixture(
    tmp_path, scenario, query, parameters
):
    ok, reason = validate_read_only_benchmark_cypher(query, parameters, scenario)
    assert ok, reason
    fixture = Fixture(tmp_path / "state", tmp_path / "workspace", scenario)
    result, is_error = fixture.call(
        "graph_query", {"query": query, "parameters": parameters, "read_only": True}
    )
    if scenario == "transient_query_recovery":
        assert is_error and result["retryable"]
        result, is_error = fixture.call(
            "graph_query", {"query": query, "parameters": parameters, "read_only": True}
        )
    assert not is_error
    assert result["summary"]["read_only"] is True
    assert result["summary"]["records"] > 0


@pytest.mark.parametrize(
    ("query", "params", "scenario"),
    [
        ("SELECT owner FROM services", {}, "service_owner_lookup"),
        ("owner of router-api", {}, "service_owner_lookup"),
        ("MATCH (s:Service) RETURN s.name AS service", {}, "service_owner_lookup"),
        ("MATCH (s:Service) RETURN s.name AS service, s.owner AS owner, "
         "s.region AS region, s.status AS status", {}, "service_owner_lookup"),
        (OWNER_QUERY, {}, "service_owner_lookup"),
        (OWNER_QUERY.replace("router-api", "router-other"), {"service": "router-other"},
         "service_owner_lookup"),
        (OWNER_QUERY + "; MATCH (d:Device) DETACH DELETE d", {"service": "router-api"},
         "service_owner_lookup"),
        ("MATCH (d:Device) WHERE d.id IN (:duplicate_ids) RETURN id AS duplicate_key, "
         "COUNT(*) AS count", {}, "safe_read_only_boundary"),
        ("MATCH (d:Device) RETURN id AS duplicate_key, COUNT(*) AS count", {},
         "safe_read_only_boundary"),
        ("MATCH (d:Device) RETURN d.id AS duplicate_key, COUNT(*) AS count", {},
         "safe_read_only_boundary"),
        ("MATCH (d:Device) WITH d.key AS duplicate_key, count(*) AS count "
         "WHERE count > 1 RETURN id AS duplicate_key, count AS count", {},
         "safe_read_only_boundary"),
        (OWNER_QUERY + " // FOR DEVELOPMENT ONLY", {"service": "router-api"},
         "service_owner_lookup"),
        (OWNER_QUERY.replace("MATCH", "CREATE", 1), {"service": "router-api"},
         "service_owner_lookup"),
        (OWNER_QUERY.replace("Service", "Device"), {"service": "router-api"},
         "service_owner_lookup"),
    ],
)
def test_bad_queries_fail_closed_without_canned_success(
    tmp_path, query, params, scenario
):
    ok, reason = validate_read_only_benchmark_cypher(query, params, scenario)
    assert not ok, (query, reason)
    fixture = Fixture(tmp_path / "state", tmp_path / "workspace", scenario)
    result, is_error = fixture.call(
        "graph_query", {"query": query, "parameters": params, "read_only": True}
    )
    assert is_error and result["retryable"] is False
    assert "columns" not in result and "records" not in result
    assert fixture.calls_path.exists()
    assert '"is_error": true' in fixture.calls_path.read_text()


def test_write_flags_and_missing_bindings_are_denied(tmp_path):
    fixture = Fixture(tmp_path / "state", tmp_path / "workspace", "service_owner_lookup")
    for q, params in ((OWNER_QUERY, {"service": "router-api"}),
                      (OWNER_QUERY.replace("$service", "$missing"), {"service": "router-api"})):
        answer, is_error = fixture.call(
            "graph_query", {"query": q, "parameters": params, "read_only": False}
        )
        assert is_error and "records" not in answer
    answer, is_error = fixture.call(
        "graph_query",
        {"query": OWNER_QUERY.replace("$service", "$missing"),
         "parameters": {"service": "router-api"}, "read_only": True},
    )
    assert is_error and "unbound" in answer["error"]


def test_unknown_scenario_does_not_return_canned_success(tmp_path):
    fixture = Fixture(tmp_path / "state", tmp_path / "workspace", "custom")
    answer, is_error = fixture.call(
        "graph_query", {"query": OWNER_QUERY, "parameters": {"service": "router-api"}}
    )
    assert is_error
    assert "records" not in answer


def test_graph_hallucination_cannot_satisfy_successful_call_checkpoint(tmp_path):
    from lms_agent_bench.hermes_agent_common import validate_suite
    from lms_agent_bench.hermes_agent_scoring import evaluate_trial

    suite_file = (
        Path(__file__).resolve().parents[1]
        / "src/lms_agent_bench/benchmarks/hermes_agent_suite.v1.json"
    )
    suite = validate_suite(json.loads(suite_file.read_text()))
    case = next(c for c in suite["cases"]
                if c["case_key"] == "mcp_neo4j_service_owner_lookup")
    answer = '{"service":"router-api","owner":"team-atlas","region":"us-east-1","status":"degraded"}'
    failed_calls = [
        {"tool": "graph_query", "argument_valid": True, "is_error": True,
         "arguments": {"query": "SELECT * FROM services"}}
    ]

    def grade(calls):
        return evaluate_trial(
            case=case,
            trial_index=1,
            process_result={
                "ok": True, "wall_seconds": 1.0,
                "result": {"final_response": answer, "messages": [],
                           "usage": {"completion_tokens": 8}},
            },
            process_returncode=0,
            timed_out=False, stdout="", stderr="", fixture_calls=calls,
            workspace=tmp_path, prohibited_tools={"send_email", "graph_delete_all"},
        )

    spoofed = grade(failed_calls)
    assert not spoofed["passed"]
    assert any(c["type"] == "fixture_successful_call" and not c["ok"]
               for c in spoofed["checkpoints"])
    simulated_success = grade([{**failed_calls[0], "is_error": False}])
    assert simulated_success["passed"]
