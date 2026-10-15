"""Fixed worker-api holdout safety: existing router default remains fail-closed."""

import pytest

from lms_agent_bench.hermes_real_cypher_gate import (
    OWNER, SERVICE, OwnerWitness, ScratchCypherGate, WitnessDenied,
    verify_final_answer,
)
from lms_agent_bench.hermes_mcp_fixture import owner_lookup_query_admissible
from lms_agent_bench.hermes_native_worker_holdout import adapt_worker_request
from lms_agent_bench.hermes_native_generic_query import GENERIC
from lms_agent_bench.hermes_native_query_description import ORIGINAL
from lms_agent_bench.hermes_native_tool_order import CANONICAL

ROUTER = "MATCH (s:Service {name: 'router-api'}) RETURN s.owner AS owner"
WORKER = "MATCH (s:Service {name: 'worker-api'}) RETURN s.owner AS owner"
CID = "a" * 64


def test_defaults_are_still_router_scoped():
    assert (OWNER, SERVICE) == ("team-atlas", "router-api")
    ScratchCypherGate._safe_query(ROUTER, None, True)
    with pytest.raises(WitnessDenied):
        ScratchCypherGate._safe_query(WORKER, None, True)
    assert not owner_lookup_query_admissible(WORKER)[0]
    assert not owner_lookup_query_admissible(ROUTER, service="worker-api")[0]


def test_holdout_accepts_exact_worker_only():
    gate = ScratchCypherGate(expected_container_id=CID, service_name="worker-api")
    assert gate.expected_owner == "team-beta"
    gate._safe_query(WORKER, None, True, service_name=gate.service_name)
    with pytest.raises(WitnessDenied):
        gate._safe_query(ROUTER, None, True, service_name=gate.service_name)
    for query in (
        "MATCH (s:Service) RETURN s.owner AS owner",
        "MATCH (s:Service {name: 'worker-api'}) RETURN s",
        "MATCH (s:Service {name: 'worker-api'}) DELETE s",
        "MATCH (s:Service {name: 'worker-api'}) RETURN s.owner AS owner; CREATE (x:X)",
    ):
        with pytest.raises(WitnessDenied):
            gate._safe_query(query, None, True, service_name="worker-api")


def test_unknown_target_and_writes_denied():
    with pytest.raises(WitnessDenied):
        ScratchCypherGate(expected_container_id=CID, service_name="production-api")
    with pytest.raises(WitnessDenied):
        ScratchCypherGate._safe_query(WORKER, None, False, service_name="worker-api")
    with pytest.raises(WitnessDenied):
        ScratchCypherGate._safe_query(WORKER, {"owner": "team-beta"}, True, service_name="worker-api")


def test_final_answer_requires_correct_bound_identity():
    w = OwnerWitness(True, "verified owner", WORKER, ("owner",),
                     (("team-beta",),), CID, 0)
    assert verify_final_answer(w, "The owner of worker-api is **team-beta**.",
                               service_name="worker-api")
    assert not verify_final_answer(w, "The owner of router-api is team-atlas.",
                                   service_name="worker-api")
    assert not verify_final_answer(w, "The owner of worker-api is NOT team-beta.",
                                   service_name="worker-api")
    assert not verify_final_answer(w, "The owner of worker-api is team-beta.",
                                   service_name="production-api")


def test_native_worker_adapter_has_generic_no_answer(tmp_path):
    obj = {
        "model": "liquid-24b-parity-candidate",
        "messages": [
            {"role": "system", "content": "native instruction " * 60},
            {"role": "user", "content": "Find owner of worker-api"},
        ],
        "tools": [
            {"type": "function", "function": {
                "name": name,
                "description": ORIGINAL if name.endswith("__graph_query") else name,
                "parameters": {"type": "object", "properties": {}},
            }} for name in CANONICAL
        ],
    }
    first, receipt = adapt_worker_request(obj, pinned=tmp_path / "private.txt")
    assert receipt["mode"] == "worker-api-holdout"
    assert first["tools"][1]["function"]["description"] == GENERIC
    assert "worker-api" not in GENERIC
    assert "team-beta" not in GENERIC
    assert "team-atlas" not in GENERIC
    assert receipt["qualification_authorized"] is False
    assert (tmp_path / "private.txt").stat().st_mode & 0o077 == 0


def test_entrypoint_explicit_worker_disclosure_branch():
    from pathlib import Path
    # Source guard on the research-specific child process, not Hermes global code.
    path = Path(__file__).resolve().parents[1] / (
        "scripts/hermes_native_worker_holdout_entrypoint.py"
    )
    src = path.read_text()
    assert 'if mode == "worker":' in src
    assert 'if mode in ("specific", "generic"):' not in src
    assert 'kw["skip_tool_search_assembly"] = True' in src
    assert '"--research-holdout-service", "worker-api"' in src
