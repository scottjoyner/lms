"""Offline regression tests for fail-closed two-condition evidence scoring."""
import json
from pathlib import Path
from runpy import run_path

import pytest

SCRIPT = run_path(str(Path(__file__).resolve().parents[1] /
                      "scripts/hermes_native_preamble_ablation_audit.py"))
score_pair = SCRIPT["score_pair"]


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc) + "\n")


@pytest.fixture
def pair(tmp_path):
    root = tmp_path / "pair"
    ids = ["a" * 64, "b" * 64]
    write(root / "native-research-summary.json", {
        "research_only": True, "release_authorized": False,
        "qualification_authorized": False, "research_pair_only": True,
        "run_count_expected": 2, "research_modes": ["control", "minimal"],
        "model_server_removed": True, "owned_container_ids": ids,
    })
    for index, mode in enumerate(("control", "minimal"), 1):
        trial = root / "trial-roots/mcp_tool_distractor_selection" / f"trial-{index:02d}"
        base = trial / "fixture-state"
        write(trial / "trial-report.json", {
            "timed_out": True, "valid": False, "final_response": "",
        })
        write(trial / "native-independent-provenance.json", {
            "scratch_container_id": ids[index-1],
            "teardown_verified": True, "exact_container_removed": True,
            "native_transcript_http_correlated": False,
            "native_complete_source_provenance": False,
        })
        write(base / "calls.jsonl", {
            "tool": "graph_query", "is_error": True,
            "result": {"error": "denied"},
        })
        write(base / "model-request-ablation.jsonl", {
            "mode": mode,
            "original_system_chars": 7300 + index,
            "effective_system_chars": 109 if index == 2 else 7300 + index,
            "original_system_sha256": str(index) * 64,
            "effective_system_sha256": str(index) * 64,
        })
        write(base / "model-http-receipts.jsonl", {
            "wire_surface": {
                "generation": {"temperature": 0.0, "seed": 43,
                               "max_tokens": 640, "stream": True},
                "model_message_roles": ["system", "user"],
                "advertised_tool_contracts": [
                    {"name": x, "sha": "fixed"}
                    for x in ("tool_search", "tool_describe", "tool_call")
                ],
                "user_content_sha256": ["c" * 64],
                "all_other_options_sha256": "d" * 64,
                "system_content_sha256": [str(index) * 64],
            },
        })
    return root


def test_first_wire_matches_only_system_and_never_qualifies(pair):
    result = score_pair(pair)
    assert result["actual_first_wire_only_system_differs"] is True
    assert result["original_system_prompt_drifted"] is True
    assert result["native_verified_success_count"] == 0
    assert result["engine_verified_success_count"] == 0
    assert result["native_three_trial_gate_passed"] is False
    assert result["causal_preamble_effect_established"] is False


@pytest.mark.parametrize("relative,corrupt", [
    ("native-research-summary.json", lambda x: x.update(model_server_removed=False)),
    ("native-research-summary.json", lambda x: x.update(
        owned_container_ids=["a" * 64, "a" * 64])),
    ("trial-roots/mcp_tool_distractor_selection/trial-02/fixture-state/model-http-receipts.jsonl",
     lambda x: x["wire_surface"].update(generation={"temperature": 0.8})),
    ("trial-roots/mcp_tool_distractor_selection/trial-02/fixture-state/model-http-receipts.jsonl",
     lambda x: x["wire_surface"].update(advertised_tool_contracts=[])),
    ("trial-roots/mcp_tool_distractor_selection/trial-01/native-independent-provenance.json",
     lambda x: x.update(teardown_verified=False)),
    ("trial-roots/mcp_tool_distractor_selection/trial-01/fixture-state/calls.jsonl",
     lambda x: x.update(tool="send_email")),
])
def test_missing_model_custody_schema_and_safety_fail_closed(pair, relative, corrupt):
    path = pair / relative
    document = json.loads(path.read_text())
    corrupt(document)
    write(path, document)
    with pytest.raises(ValueError):
        score_pair(pair)


def test_tampering_with_user_hash_is_detected_without_leaking_text(pair):
    p = pair / "trial-roots/mcp_tool_distractor_selection/trial-02/fixture-state/model-http-receipts.jsonl"
    doc = json.loads(p.read_text())
    doc["wire_surface"]["user_content_sha256"] = ["e" * 64]
    write(p, doc)
    result = score_pair(pair)
    assert result["actual_first_wire_only_system_differs"] is False
    assert result["causal_preamble_effect_established"] is False


def test_non_hex_exact_resource_identity_denied(pair):
    path = pair / "native-research-summary.json"
    state = json.loads(path.read_text())
    state["owned_container_ids"][1] = "z" * 64
    write(path, state)
    with pytest.raises(ValueError, match="scoped research Docker IDs"):
        score_pair(pair)
