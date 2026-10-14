"""Offline proof tests: no Docker, production graph, fleet, or model required."""
import hashlib
import json

import httpx
import pytest

from lms_agent_bench.hermes_native_http_probe import observe_loopback_http
from lms_agent_bench.hermes_native_provenance import audit_native_trial, _hash_json

QUERY = "MATCH (s:Service {name:'router-api'}) RETURN s.owner AS owner"
CID = "a" * 64
TOOL = "mcp__lms_benchmark__graph_query"
ANSWER = "The owner of router-api is team-atlas."


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


@pytest.fixture
def trial(tmp_path):
    root = tmp_path / "trial-01"
    state = root / "fixture-state"
    args = {"query": QUERY, "read_only": True}
    result = {
        "columns": ["owner"], "records": [{"owner": "team-atlas"}],
        "summary": {
            "read_only": True, "records": 1,
            "source": "disposable_real_neo4j", "container_id": CID,
        },
    }
    content = json.dumps({
        "result": json.dumps(result, sort_keys=True),
        "structuredContent": result,
    })
    messages = [
        {"role": "user", "content": "Who owns router-api?"},
        {"role": "assistant", "tool_calls": [{
            "id": "native-id-7", "type": "function",
            "function": {"name": TOOL, "arguments": json.dumps(args)},
        }]},
        {"role": "tool", "tool_call_id": "native-id-7", "content": content},
        {"role": "assistant", "content": ANSWER},
    ]
    _write(root / "trial-report.json", {
        "case_key": "mcp_tool_distractor_selection", "trial_index": 1,
        "timed_out": False, "process_returncode": 0,
        "valid": True, "final_response": ANSWER,
    })
    _write(root / "loadout.json", {
        "model": {"id": "research-model", "content_sha256": "b" * 64},
    })
    _write(root / "agent-result.json", {
        "ok": True, "result": {
            "completed": True, "failed": False, "final_response": ANSWER,
            "messages": messages,
        },
    })
    _jsonl(state / "calls.jsonl", [{
        "mcp_request_id": 7, "sequence": 1,
        "tool": "graph_query", "arguments": args, "is_error": False,
        "argument_valid": True, "result": result,
    }])
    _jsonl(state / "mcp-wire-responses.jsonl", [{
        "mcp_request_id": 7, "kind": "mcp_stdout_flushed_only",
        "is_error": False, "result_sha256": _hash_json(result),
        "timestamp_unix_ns": 100,
    }])
    _jsonl(state / "model-http-receipts.jsonl", [{
        "kind": "loopback_http_response_observed",
        "http_status": 200, "timestamp_unix_ns": 200,
        "model": "research-model", "request_sha256": "f" * 64,
        "tool_messages": [{
            "tool_call_id": "native-id-7",
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }],
    }])
    return root


def test_matching_native_transcript_and_outbound_http_is_correlated(trial):
    output = audit_native_trial(trial)
    assert output["native_transcript_http_correlated"] is True, output
    assert output["teardown_verified"] is False
    assert output["qualification_authorized"] is False
    assert output["release_authorized"] is False
    assert output["exact_container_id"] == CID
    assert len(output["source_sha256"]) == 6


@pytest.mark.parametrize("filename, mutate, expected", [
    ("model-http-receipts.jsonl", lambda x: x.update(http_status=503), "model HTTP"),
    ("model-http-receipts.jsonl", lambda x: x.update(timestamp_unix_ns=90), "model HTTP"),
    ("model-http-receipts.jsonl", lambda x: x.update(model="other-model"), "model HTTP"),
    ("model-http-receipts.jsonl", lambda x: x.update(tool_messages=[]), "model HTTP"),
    ("mcp-wire-responses.jsonl", lambda x: x.update(mcp_request_id=8), "MCP stdout"),
    ("mcp-wire-responses.jsonl", lambda x: x.update(result_sha256="f" * 64), "MCP response"),
    ("calls.jsonl", lambda x: x.update(is_error=True), "one accepted"),
])
def test_forged_or_stale_receipts_are_denied(trial, filename, mutate, expected):
    path = trial / "fixture-state" / filename
    record = json.loads(path.read_text())
    mutate(record)
    _jsonl(path, [record])
    output = audit_native_trial(trial)
    assert not output["native_transcript_http_correlated"]
    assert expected in output["denial"]


def test_archived_replay_without_native_http_receipt_is_denied(trial):
    (trial / "fixture-state" / "model-http-receipts.jsonl").unlink()
    output = audit_native_trial(trial)
    assert output["native_transcript_http_correlated"] is False
    assert "missing evidence" in output["denial"]


@pytest.mark.parametrize("change, expected", [
    (lambda x: x["result"].update(final_response="router-api is NOT owned by team-atlas"),
     "final response"),
    (lambda x: x["result"]["messages"].pop(), "never continued"),
    (lambda x: x["result"]["messages"][2].update(content="{}"), "pairing"),
    (lambda x: x["result"]["messages"][1]["tool_calls"][0].update(id="wrong-id"), "pairing"),
])
def test_mutated_hermes_history_is_denied(trial, change, expected):
    path = trial / "agent-result.json"
    document = json.loads(path.read_text())
    change(document)
    _write(path, document)
    outcome = audit_native_trial(trial)
    assert not outcome["native_transcript_http_correlated"]
    assert expected in outcome["denial"]


def test_http_probe_observes_real_httpx_send_without_collecting_prompt(tmp_path):
    seen = []
    def backend(request):
        seen.append(request.content)
        return httpx.Response(200, json={"choices": []})
    path = tmp_path / "http.jsonl"
    request_data = {"model": "research-model", "messages": [
        {"role": "user", "content": "private research prompt"},
        {"role": "tool", "tool_call_id": "id", "content": "synthetic-result"},
    ]}
    with httpx.Client(transport=httpx.MockTransport(backend)) as client:
        with observe_loopback_http(path, endpoint="http://127.0.0.1:12345/v1"):
            response = client.post(
                "http://127.0.0.1:12345/v1/chat/completions", json=request_data,
            )
            assert response.status_code == 200
    assert seen
    witness = json.loads(path.read_text())
    assert witness["http_status"] == 200
    assert witness["tool_messages"][0]["tool_call_id"] == "id"
    assert "private research prompt" not in path.read_text()
    assert "synthetic-result" not in path.read_text()


def test_mcp_wire_receipt_matches_actual_stdout(monkeypatch, tmp_path):
    import io
    from lms_agent_bench.hermes_mcp_fixture import Fixture, serve
    state, workspace = tmp_path / "state", tmp_path / "workspace"
    fixture = Fixture(state, workspace, "tool_distractor_selection")
    arguments = {"query": QUERY, "read_only": True}
    request = {"jsonrpc": "2.0", "id": 23, "method": "tools/call",
               "params": {"name": "graph_query", "arguments": arguments}}
    output = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr("sys.stdout", output)
    assert serve(fixture) == 0
    response = json.loads(output.getvalue())
    assert response["id"] == 23
    record = json.loads((state / "calls.jsonl").read_text())
    receipt = json.loads((state / "mcp-wire-responses.jsonl").read_text())
    assert record["mcp_request_id"] == 23
    assert receipt["mcp_request_id"] == 23
    assert receipt["result_sha256"] == _hash_json(record["result"])
    assert receipt["kind"] == "mcp_stdout_flushed_only"


def test_duplicate_http_delivery_is_not_independent_proof(trial):
    path = trial / "fixture-state" / "model-http-receipts.jsonl"
    row = json.loads(path.read_text())
    _jsonl(path, [row, row])
    outcome = audit_native_trial(trial)
    assert not outcome["native_transcript_http_correlated"]
    assert "model HTTP" in outcome["denial"]


def test_unexpected_research_mcp_tool_blocks_qualification(trial):
    path = trial / "fixture-state" / "calls.jsonl"
    row = json.loads(path.read_text())
    _jsonl(path, [row, {**row, "sequence": 2, "tool": "send_email", "is_error": True}])
    outcome = audit_native_trial(trial)
    assert not outcome["native_transcript_http_correlated"]
    assert "non-graph MCP" in outcome["denial"]


def _rewrite_as_hermes_bridge(trial):
    from lms_agent_bench.hermes_native_provenance import TRUST_HEADER, TRUST_END
    agent_path = trial / "agent-result.json"
    agent = json.loads(agent_path.read_text())
    messages = agent["result"]["messages"]
    inner = json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])
    messages[1]["tool_calls"][0]["function"] = {
        "name": "tool_call",
        "arguments": json.dumps({"name": TOOL, "arguments": inner}),
    }
    wrapped = TRUST_HEADER + messages[2]["content"] + TRUST_END
    messages[2]["content"] = wrapped
    messages[2]["name"] = TOOL
    _write(agent_path, agent)
    http_path = trial / "fixture-state" / "model-http-receipts.jsonl"
    http_receipt = json.loads(http_path.read_text())
    http_receipt["tool_messages"][0]["content_sha256"] = hashlib.sha256(
        wrapped.encode()
    ).hexdigest()
    _jsonl(http_path, [http_receipt])


def test_native_hermes_tool_call_bridge_and_untrusted_wrapper_are_valid(trial):
    _rewrite_as_hermes_bridge(trial)
    result = audit_native_trial(trial)
    assert result["native_transcript_http_correlated"] is True, result
    assert result["qualification_authorized"] is False


@pytest.mark.parametrize("tamper, reason", [
    ("wrong_source", "pairing"),
    ("duplicate_close", "pairing"),
    ("wrong_bridge_name", "pairing"),
    ("forged_owner_text", "pairing"),
])
def test_native_bridge_tampering_denied(trial, tamper, reason):
    _rewrite_as_hermes_bridge(trial)
    agent_path = trial / "agent-result.json"
    agent = json.loads(agent_path.read_text())
    if tamper == "wrong_source":
        agent["result"]["messages"][2]["content"] = (
            agent["result"]["messages"][2]["content"].replace(
                'source="mcp__lms_benchmark__graph_query"',
                'source="mcp__lms_benchmark__graph_delete_all"',
            )
        )
    elif tamper == "duplicate_close":
        agent["result"]["messages"][2]["content"] += "</untrusted_tool_result>"
    elif tamper == "wrong_bridge_name":
        call = agent["result"]["messages"][1]["tool_calls"][0]["function"]
        args = json.loads(call["arguments"])
        args["name"] = "mcp__lms_benchmark__send_email"
        call["arguments"] = json.dumps(args)
    else:
        agent["result"]["messages"][2]["content"] = (
            agent["result"]["messages"][2]["content"].replace(
                "team-atlas", "team-fake"
            )
        )
    _write(agent_path, agent)
    outcome = audit_native_trial(trial)
    assert outcome["native_transcript_http_correlated"] is False
    assert reason in outcome["denial"]

def test_independent_cypher_shape_check_rejects_canned_result(trial):
    """A forged owner row cannot rescue an unbound query, even with hashes."""
    record_path = trial / "fixture-state" / "calls.jsonl"
    record = json.loads(record_path.read_text())
    forged = "MATCH (s:Service {name: 'router-api'}) RETURN 'team-atlas' AS owner"
    record["arguments"]["query"] = forged
    _jsonl(record_path, [record])
    artifact = trial / "agent-result.json"
    agent = json.loads(artifact.read_text())
    call = agent["result"]["messages"][1]["tool_calls"][0]["function"]
    call["arguments"] = json.dumps(record["arguments"])
    _write(artifact, agent)
    result = audit_native_trial(trial)
    assert result["native_transcript_http_correlated"] is False
    assert "owner projection is not bound" in result["denial"]


def test_scoped_container_id_must_match_native_owner_witness(trial):
    accepted = audit_native_trial(trial, expected_container_id=CID)
    assert accepted["native_transcript_http_correlated"] is True
    rejected = audit_native_trial(trial, expected_container_id="f" * 64)
    assert rejected["native_transcript_http_correlated"] is False
    assert "different scoped research container" in rejected["denial"]


def _engine_receipt_for(trial):
    state = trial / "fixture-state"
    record = json.loads((state / "calls.jsonl").read_text())
    wire = json.loads((state / "mcp-wire-responses.jsonl").read_text())
    return {
        "schema": "lms-hermes-neo4j-engine-witness.v2",
        "mcp_request_id": record["mcp_request_id"],
        "query": record["arguments"]["query"],
        "accepted": True,
        "execution_exit_code": 0,
        "cypher_shell_stdout_sha256": "d" * 64,
        "columns": ["owner"],
        "records": [["team-atlas"]],
        "exact_container_id": CID,
        "mcp_result_sha256": _hash_json(record["result"]),
        "timestamp_unix_ns": wire["timestamp_unix_ns"] - 1,
    }


def test_engine_witness_strengthens_native_provenance(trial):
    assert audit_native_trial(trial)["legacy_http_only"] is True
    _jsonl(trial / "fixture-state" / "neo4j-engine-witnesses.jsonl", [
        _engine_receipt_for(trial),
    ])
    output = audit_native_trial(trial)
    assert output["native_transcript_http_correlated"] is True
    assert output["neo4j_engine_witness_verified"] is True
    assert output["native_complete_source_provenance"] is True
    assert output["legacy_http_only"] is False
    assert "neo4j_engine" in output["source_sha256"]
    assert output["qualification_authorized"] is False


@pytest.mark.parametrize("field, bad_value", [
    ("schema", "other-schema"),
    ("mcp_request_id", 99),
    ("query", "MATCH (n) RETURN n"),
    ("accepted", False),
    ("execution_exit_code", 1),
    ("execution_exit_code", True),
    ("cypher_shell_stdout_sha256", "not-a-digest"),
    ("columns", ["name"]),
    ("records", [["wrong-owner"]]),
    ("records", [["team-atlas"], ["team-atlas"]]),
    ("exact_container_id", "c" * 64),
    ("mcp_result_sha256", "f" * 64),
    ("timestamp_unix_ns", 101),
])
def test_bad_engine_witness_never_passes(trial, field, bad_value):
    record = _engine_receipt_for(trial)
    record[field] = bad_value
    _jsonl(trial / "fixture-state" / "neo4j-engine-witnesses.jsonl", [record])
    output = audit_native_trial(trial)
    assert output["native_complete_source_provenance"] is False
    assert output["native_transcript_http_correlated"] is False
    assert "Neo4j engine witness" in output["denial"]


def test_duplicate_engine_receipt_is_denied(trial):
    record = _engine_receipt_for(trial)
    _jsonl(trial / "fixture-state" / "neo4j-engine-witnesses.jsonl", [record, record])
    output = audit_native_trial(trial)
    assert output["native_complete_source_provenance"] is False
    assert "Neo4j engine witness" in output["denial"]


def test_legacy_native_run_remains_nonpromotable_without_engine_receipt(trial):
    output = audit_native_trial(trial)
    assert output["native_transcript_http_correlated"] is True
    assert output["native_complete_source_provenance"] is False
    assert output["neo4j_engine_witness_verified"] is False
    assert output["qualification_authorized"] is False


def test_posthoc_three_replays_with_same_container_are_not_physical_acceptance(
    trial, tmp_path, monkeypatch,
):
    """Three otherwise valid transcripts cannot reuse one scratch Neo4j ID."""
    import shutil
    import subprocess

    from pathlib import Path
    from runpy import run_path

    script = run_path(str(
        Path(__file__).resolve().parents[1] / "scripts/hermes_native_posthoc_rescore.py"
    ))
    score_existing_run = script["score_existing_run"]

    original = tmp_path / "native-experiment"
    originals = []
    for i in (1, 2, 3):
        target = original / "trial-roots" / "mcp_tool_distractor_selection" / f"trial-{i:02d}"
        shutil.copytree(trial, target)
        report_file = target / "trial-report.json"
        report = json.loads(report_file.read_text())
        report["trial_index"] = i
        _write(report_file, report)
        old = {
            "exact_container_removed": True,
            "scratch_container_id": CID,
            "teardown_receipt": {
                "docker_daemon_healthy": True,
                "exact_container_absent": True,
            },
            "native_transcript_http_correlated": True,
            "runner_valid": True,
        }
        _write(target / "native-independent-provenance.json", old)
        originals.append(old)
    _write(original / "native-research-summary.json", {
        "run_count_expected": 3,
        "trials": originals,
        "owned_container_ids": [CID, CID, CID],
        "research_only": True,
        "native_correlated_count": 3,
        "model_server_removed": True,
    })

    def missing_container(command, **kwargs):
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, "29.8.2\n", "")
        assert command[:2] == ["docker", "inspect"]
        return subprocess.CompletedProcess(
            command, 1, "", f"error: no such object: {CID}",
        )

    monkeypatch.setattr(subprocess, "run", missing_container)
    rescore = score_existing_run(original)
    assert rescore["posthoc_native_success_count"] == 3
    assert rescore["distinct_owned_container_ids_verified"] is False
    assert rescore["physical_source_custody_verified"] is False
    assert rescore["physical_three_trial_acceptance"] is False
    assert rescore["case_gate_passed"] is False
