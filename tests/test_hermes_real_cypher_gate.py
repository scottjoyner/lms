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


def test_scratch_run_command_hard_isolation():
    from lms_agent_bench.hermes_scratch_replay import RUN_COMMAND, CONTAINER_NAME
    assert RUN_COMMAND[:4] == ["docker", "run", "-d", "--name"]
    assert RUN_COMMAND[4] == CONTAINER_NAME
    assert RUN_COMMAND[RUN_COMMAND.index("--network") + 1] == "none"
    assert RUN_COMMAND[RUN_COMMAND.index("--memory") + 1] == "3g"
    assert "NEO4J_AUTH=none" in RUN_COMMAND
    assert "--publish" not in RUN_COMMAND and "-p" not in RUN_COMMAND
    assert "--volume" not in RUN_COMMAND and "-v" not in RUN_COMMAND
    assert "--mount" not in RUN_COMMAND
    assert "/data:rw,size=2g" in RUN_COMMAND
    assert RUN_COMMAND[-1] == SCRATCH_IMAGE


class FakeResearchLifecycle:
    def __init__(self, *, initially_exists=False, seed_fails=False):
        self.exists = initially_exists
        self.seed_fails = seed_fails
        self.destroyed = False
        self.ops = []

    def __call__(self, args, *, timeout):
        self.ops.append(args)
        if args[1] == "inspect":
            if not self.exists:
                return subprocess.CompletedProcess(args, 1, "", "")
            if "{{json .}}" in args:
                return subprocess.CompletedProcess(args, 0, json.dumps(container_receipt()), "")
            return subprocess.CompletedProcess(args, 0, CONTAINER_ID + "\n", "")
        if args[1] == "run":
            assert not self.exists
            self.exists = True
            return subprocess.CompletedProcess(args, 0, CONTAINER_ID + "\n", "")
        if args[1] == "rm":
            assert self.exists
            self.exists = False
            self.destroyed = True
            return subprocess.CompletedProcess(args, 0, CONTAINER_ID + "\n", "")
        assert args[1] == "exec"
        query = args[-1]
        if query.startswith("RETURN 1"):
            return subprocess.CompletedProcess(args, 0, "ready\n1\n", "")
        if "count(n)" in query:
            return subprocess.CompletedProcess(args, 0, "node_count\n0\n", "")
        if query.startswith("CREATE "):
            return subprocess.CompletedProcess(args, 1 if self.seed_fails else 0, "", "")
        if "ORDER BY service" in query:
            return subprocess.CompletedProcess(
                args, 0,
                'service, owner\n"router-api", "team-atlas"\n"worker-api", "team-beta"\n', ""
            )
        raise AssertionError("unexpected query in fake harness")


def test_scratch_launcher_denies_existing_named_container():
    from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive
    fake = FakeResearchLifecycle(initially_exists=True)
    with pytest.raises(WitnessDenied, match="exists"):
        with ScratchHermesArchive(runner=fake):
            pass
    assert fake.exists and not fake.destroyed
    assert not any(x[1] in ("run", "rm", "exec") for x in fake.ops)


def test_scratch_launcher_starts_seeds_and_always_tears_down():
    from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive
    fake = FakeResearchLifecycle()
    with ScratchHermesArchive(runner=fake) as research:
        assert research.container_id == CONTAINER_ID
        assert "team-atlas" in research.readback
        assert fake.exists
    assert fake.destroyed and not fake.exists
    assert sum(x[1] == "run" for x in fake.ops) == 1
    assert sum(x[1] == "rm" for x in fake.ops) == 1
    assert sum(x[-1].startswith("CREATE ") for x in fake.ops if x[1] == "exec") == 1


def test_scratch_launcher_destroys_on_seed_failure():
    from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive
    fake = FakeResearchLifecycle(seed_fails=True)
    with pytest.raises(WitnessDenied):
        with ScratchHermesArchive(runner=fake):
            pass
    assert fake.destroyed and not fake.exists
    assert sum(x[1] == "rm" for x in fake.ops) == 1


def test_scratch_run_rejects_existing_output_before_docker(tmp_path):
    from lms_agent_bench.hermes_scratch_replay import run_archived_research
    output = tmp_path / "receipt.json"
    output.write_text("existing")
    with pytest.raises(WitnessDenied, match="overwrite"):
        run_archived_research(tmp_path, output)
    assert output.read_text() == "existing"


def test_harness_cli_exit_code_does_not_promote_failed_model(monkeypatch, tmp_path):
    import sys
    from lms_agent_bench import hermes_scratch_replay as entry
    monkeypatch.setattr(sys, "argv", ["hermes-scratch", "--confirm-research-only",
                                       "--case-dir", str(tmp_path), "--output",
                                       str(tmp_path/"report.json")])
    monkeypatch.setattr(entry, "run_archived_research", lambda *a, **kw: {
        "real_end_to_end_pass_count": 1, "trial_count": 3,
        "case_gate_passed": False,
    })
    assert entry.main() == 3


def test_harness_cli_never_promotes_archived_replay(monkeypatch, tmp_path):
    import sys
    from lms_agent_bench import hermes_scratch_replay as entry
    monkeypatch.setattr(sys, "argv", ["hermes-scratch", "--confirm-research-only",
                                       "--case-dir", str(tmp_path), "--output",
                                       str(tmp_path/"report.json")])
    monkeypatch.setattr(entry, "run_archived_research", lambda *a, **kw: {
        "real_end_to_end_pass_count": 3, "trial_count": 3,
        "case_gate_passed": True,
    })
    assert entry.main() == 3


def test_teardown_of_stopped_owned_container_uses_exact_id():
    from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive
    class Stopped(FakeResearchLifecycle):
        def __call__(self, args, *, timeout):
            if args[1] == "inspect" and "{{json .}}" in args and self.exists:
                self.ops.append(args)
                receipt = container_receipt()
                receipt["State"]["Status"] = "exited"
                return subprocess.CompletedProcess(args, 0, json.dumps(receipt), "")
            return super().__call__(args, timeout=timeout)
    fake = Stopped(initially_exists=True)
    scratch = ScratchHermesArchive(runner=fake)
    scratch.container_id = CONTAINER_ID
    scratch.close()
    assert scratch.destroyed and not fake.exists
    assert [x for x in fake.ops if x[1] == "rm"] == [["docker", "rm", "-f", CONTAINER_ID]]


def test_teardown_rejects_wrong_container_identity():
    from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive
    fake = FakeResearchLifecycle(initially_exists=True)
    scratch = ScratchHermesArchive(runner=fake)
    scratch.container_id = "f" * 64
    with pytest.raises(WitnessDenied, match="identity changed"):
        scratch.close()
    assert fake.exists and not fake.destroyed
    assert not any(x[1] == "rm" for x in fake.ops)


def test_archived_replay_never_qualifies_even_three_matching_answers(tmp_path):
    from lms_agent_bench.hermes_real_cypher_audit import score_archived_case
    for i in (1, 2, 3):
        _write_archived_trial(tmp_path / f"trial-{i:02d}")
    gate = ScratchCypherGate(expected_container_id=CONTAINER_ID, command_runner=FakeDocker())
    result = score_archived_case(tmp_path, gate)
    assert result["real_end_to_end_pass_count"] == 3
    assert result["archived_replay_all_passed"] is True
    assert result["case_gate_passed"] is False
    assert result["native_end_to_end_verified"] is False
    assert all(t["archived_replay_only"] for t in result["trials"])
