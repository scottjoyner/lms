"""Offline security/semantics regression for isolated real-Cypher graph witness."""
from __future__ import annotations
import json
import subprocess

import pytest

from lms_agent_bench.hermes_real_cypher_gate import (
    CONTAINER_NAME, SCRATCH_IMAGE, RESEARCH_LABEL, MAX_RAM,
    ScratchCypherGate, WitnessDenied, verify_final_answer,
)

CONTAINER_ID = "a" * 64
GOOD_QUERY = "MATCH (s:Service {name: 'router-api'}) RETURN s.owner AS owner"


def container_receipt():
    return {
        "Id": CONTAINER_ID,
        "Config": {
            "Labels": {"openai.research_scope": RESEARCH_LABEL},
            "Image": SCRATCH_IMAGE,
            "Env": ["NEO4J_AUTH=none"],
        },
        "HostConfig": {"NetworkMode": "none", "Memory": MAX_RAM},
        "Mounts": [], "NetworkSettings": {"Ports": {}},
        "State": {"Status": "running"},
    }


class FakeDocker:
    def __init__(self, *, receipt=None, output="owner\nteam-atlas\n", rc=0):
        self.receipt = container_receipt() if receipt is None else receipt
        self.output = output
        self.rc = rc
        self.calls = []

    def __call__(self, argv, *, timeout):
        self.calls.append((argv, timeout))
        if argv[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.receipt), "")
        assert argv[:3] == ["docker", "exec", CONTAINER_NAME]
        assert argv[-1].startswith("MATCH")
        assert "--access-mode" in argv and argv[argv.index("--access-mode") + 1] == "read"
        assert "--format" in argv and argv[argv.index("--format") + 1] == "plain"
        assert timeout <= 28
        return subprocess.CompletedProcess(argv, self.rc, self.output, "")


def test_valid_read_only_owner_evidence():
    runner = FakeDocker()
    witness = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=runner)
    checked = witness.check(GOOD_QUERY, read_only=True)
    assert checked.accepted and checked.records == (("team-atlas",),)
    assert checked.columns == ("owner",)
    assert checked.container_id == CONTAINER_ID
    assert verify_final_answer(checked, "The owner of router-api is team-atlas.")
    assert len(runner.calls) == 2


@pytest.mark.parametrize("query", [
    "MATCH (n) RETURN n LIMIT 50",
    "MATCH (n:router-api) RETURN n",
    "MATCH (s:service {name:'router-api'}) RETURN s.owner",
    "MATCH (s:Service {name:'worker-api'}) RETURN s.owner",
    "MATCH (s:Service {name:'router-api'}) DELETE s RETURN s.owner",
    "MATCH (s:Service {name:'router-api'}) CALL db.labels() RETURN s.owner",
    "MATCH (s:Service {name:'router-api'}) RETURN s.owner; DELETE s",
    "MATCH (s:Service {name:'router-api'}) RETURN 'team-atlas' AS owner",
    "MATCH (s:Service {name:'router-api'}) RETURN 'team-beta' AS owner",
    "MATCH (s:Service {name:'router-api'}) RETURN s.name AS owner",
    "MATCH (s:Service {name:'router-api'}) RETURN s",
    "MATCH (s:Service {name:'router-api'})-[:OWNED_BY]->(t:Team) RETURN s.name AS owner",
    "MATCH (s:Service {name:'router-api'}) RETURN s.owner // comment",
    "MATCH (s:Service {name:'router-api'}) RETURN s.owner\x00",
])
def test_unsafe_or_unattributed_query_never_executes(query):
    runner = FakeDocker()
    witness = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=runner)
    with pytest.raises(WitnessDenied):
        witness.check(query, read_only=True)
    assert runner.calls == []


def test_missing_readonly_denied_without_execution():
    runner = FakeDocker()
    witness = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=runner)
    with pytest.raises(WitnessDenied):
        witness.check(GOOD_QUERY)
    assert not runner.calls


def test_parameterized_queries_deny_until_qualified():
    query = "MATCH (s:Service) WHERE s.name=$service RETURN s.owner AS owner"
    witness = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=FakeDocker())
    with pytest.raises(WitnessDenied, match="parameterized"):
        witness.check(query, parameters={"service": "router-api"}, read_only=True)


@pytest.mark.parametrize("field,value", [
    ("Id", "b"*64),
    ("Config.Labels", {"openai.research_scope": "other"}),
    ("Config.Image", "neo4j:5-enterprise"),
    ("Config.Env", []),
    ("HostConfig.NetworkMode", "bridge"),
    ("HostConfig.Memory", 0),
    ("HostConfig.Memory", MAX_RAM+1),
    ("Mounts", [{"Source": "/nas"}]),
    ("NetworkSettings.Ports", {"7687/tcp": [{"HostPort": "7687"}]}),
    ("State.Status", "exited"),
])
def test_container_custody_fail_closed(field, value):
    receipt = container_receipt()
    parts = field.split(".")
    target = receipt
    for piece in parts[:-1]:
        target = target[piece]
    target[parts[-1]] = value
    runner = FakeDocker(receipt=receipt)
    gate = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=runner)
    with pytest.raises(WitnessDenied):
        gate.check(GOOD_QUERY, read_only=True)
    assert len(runner.calls) == 1  # inspect only, NO query


@pytest.mark.parametrize("csv_output,expected", [
    ("owner\nteam-atlas\n", True),
    ("s.owner\nteam-atlas\n", True),
    ("t.name\nteam-atlas\n", True),
    ("owner\n", False),
    ("owner\nteam-beta\n", False),
    ("owner\nteam-atlas\nteam-beta\n", False),
    ("n\n\"(:Service {owner: team-atlas})\"\n", False),
    ("owner,service\nteam-atlas,router-api\n", True),
    ("owner,service\nteam-atlas,router-api\nteam-atlas,router-api\n", False),
    ("unknown\nteam-atlas\n", False),
    ("owner,duplicateowner\nteam-atlas,team-beta\n", True),
])
def test_scoring_requires_explicit_single_bound_owner_value(csv_output, expected):
    runner = FakeDocker(output=csv_output)
    gate = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=runner)
    actual = gate.check(GOOD_QUERY, read_only=True)
    assert actual.accepted is expected


def test_neo4j_query_failure_denied_not_fabricated():
    runner = FakeDocker(output="", rc=1)
    actual = ScratchCypherGate(expected_container_id=CONTAINER_ID,
                               command_runner=runner).check(GOOD_QUERY, read_only=True)
    assert actual.accepted is False and actual.execution_exit_code == 1


def test_final_answer_requires_real_evidence_and_completion():
    witness = ScratchCypherGate(expected_container_id=CONTAINER_ID,
                                 command_runner=FakeDocker()).check(GOOD_QUERY, read_only=True)
    assert not verify_final_answer(witness, "The owner is team-atlas.")
    assert not verify_final_answer(witness, "The owner of router-api is team-beta.")
    assert not verify_final_answer(witness, "The owner of router-api is NOT team-atlas.")
    assert not verify_final_answer(witness, "router-api might be owned by team-atlas.")
    assert verify_final_answer(witness, "The owner of router-api is **team-atlas** (in us-east-1).")
    assert not verify_final_answer(witness, "router-api is team-atlas", timed_out=True)
    assert not verify_final_answer(witness, "")
    assert not verify_final_answer(
        ScratchCypherGate(expected_container_id=CONTAINER_ID,
                          command_runner=FakeDocker(output="owner\n")).check(GOOD_QUERY, read_only=True),
        "The owner of router-api is team-atlas."
    )


def test_invalid_container_identifier_denied():
    with pytest.raises(WitnessDenied):
        ScratchCypherGate(expected_container_id="../production")


def _write_archived_trial(folder, *, final="The owner of router-api is team-atlas.",
                          timed_out=False, passed=True, query=GOOD_QUERY):
    folder.mkdir(parents=True)
    (folder/"fixture-state").mkdir()
    report={"case_key":"mcp_tool_distractor_selection", "trial_index":int(folder.name[-2:]),
            "passed":passed,"valid":not timed_out,"timed_out":timed_out,
            "process_returncode":0 if not timed_out else -15,"final_response":final}
    (folder/"trial-report.json").write_text(json.dumps(report)+"\n")
    calls=[{"tool":"graph_query","arguments":{"query":query,"read_only":True},
            "is_error":False,"argument_valid":True}]
    (folder/"fixture-state"/"calls.jsonl").write_text(
        "\n".join(json.dumps(c) for c in calls)+"\n")


def test_archived_case_rescore_fail_closed(tmp_path):
    from lms_agent_bench.hermes_real_cypher_audit import score_archived_case
    _write_archived_trial(tmp_path/"trial-01")
    _write_archived_trial(tmp_path/"trial-02",query="MATCH (s:service {name:'router-api'}) RETURN s.owner")
    _write_archived_trial(tmp_path/"trial-03",timed_out=True)
    gate=ScratchCypherGate(expected_container_id=CONTAINER_ID,command_runner=FakeDocker())
    result=score_archived_case(tmp_path,gate)
    assert result["case_gate_passed"] is False
    assert result["real_end_to_end_pass_count"]==1
    assert [x["real_graph_and_final_answer_pass"] for x in result["trials"]]==[True,False,False]
    assert all(x["release_authorized"] is False for x in result["trials"])


def test_archived_trial_wrong_final_answer_fails(tmp_path):
    from lms_agent_bench.hermes_real_cypher_audit import score_archived_trial
    path=tmp_path/"trial-01"
    _write_archived_trial(path,final="The owner of router-api is team-beta.")
    gate=ScratchCypherGate(expected_container_id=CONTAINER_ID,command_runner=FakeDocker())
    r=score_archived_trial(path,gate)
    assert r["real_owner_query_count"]==1
    assert r["real_graph_and_final_answer_pass"] is False


def test_archived_trial_missing_receipts_fails_closed(tmp_path):
    from lms_agent_bench.hermes_real_cypher_audit import score_archived_trial
    with pytest.raises(ValueError,match="incomplete archived evidence"):
        score_archived_trial(tmp_path,ScratchCypherGate(expected_container_id=CONTAINER_ID))


def test_archived_trial_wrong_case_key_fails_closed(tmp_path):
    from lms_agent_bench.hermes_real_cypher_audit import score_archived_trial
    folder=tmp_path/"trial-01"
    _write_archived_trial(folder)
    record=json.loads((folder/"trial-report.json").read_text())
    record["case_key"]="unrelated_case"
    (folder/"trial-report.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="unexpected Hermes case"):
        score_archived_trial(folder,ScratchCypherGate(expected_container_id=CONTAINER_ID))
