"""Offline fail-closed request-ablation tests, no model or graph access."""
import copy

import pytest

from lms_agent_bench.hermes_native_request_ablation import (
    GENERATION, build_ablation_request,
)

SYSTEM = "You are a precise tool-using agent. Select only tools necessary for the task and avoid external side effects."


def fixture():
    return {
        "model": "liquid-24b-parity-candidate",
        "messages": [{"role": "system", "content": "Hermes private preamble" * 150},
                     {"role": "user", "content": "Find router-api owner"}],
        "tools": [{"type": "function", "function": {"name": "tool_call"}}],
        "stream": True,
    }


def test_exactly_one_request_field_changes_between_conditions():
    base = fixture()
    frozen = copy.deepcopy(base)
    control, a = build_ablation_request(base, mode="control", canonical_system=SYSTEM)
    minimal, b = build_ablation_request(base, mode="minimal", canonical_system=SYSTEM)
    assert base == frozen
    assert control["messages"][0]["content"] == frozen["messages"][0]["content"]
    assert minimal["messages"][0]["content"] == SYSTEM
    modified = copy.deepcopy(minimal)
    modified["messages"][0] = dict(control["messages"][0])
    assert modified == control, "only initial system content can differ"
    assert a["tool_list_sha256"] == b["tool_list_sha256"]
    assert a["user_message_sha256"] == b["user_message_sha256"]
    assert a["effective_system_sha256"] != b["effective_system_sha256"]
    assert a["generation"] == b["generation"] == GENERATION
    assert all("prompt" not in k or k == "raw_prompts_saved" for k in a)


@pytest.mark.parametrize("change", [
    lambda req: req.update(messages=[]),
    lambda req: req["messages"][0].update(role="user"),
    lambda req: req["messages"][0].update(content={"bad": "type"}),
    lambda req: req["messages"].pop(),
    lambda req: req.update(tools=None),
    lambda req: req.update(max_tokens=100),
    lambda req: req.update(seed=42),
    lambda req: req.update(temperature=0.8),
])
def test_incorrect_research_wire_conditions_fail_closed(change):
    req = fixture()
    change(req)
    with pytest.raises(ValueError):
        build_ablation_request(req, mode="minimal", canonical_system=SYSTEM)


def test_invalid_control_modes_and_user_defined_system_denied():
    for mode, system in (("unknown", SYSTEM), ("minimal", "short")):
        with pytest.raises(ValueError):
            build_ablation_request(fixture(), mode=mode, canonical_system=system)


def test_later_native_adaptive_sampling_is_preserved_not_denied():
    base = fixture()
    base["messages"].extend([
        {"role": "assistant", "tool_calls": [{"id": "id"}]},
        {"role": "tool", "tool_call_id": "id", "content": "denied"},
    ])
    base.update({"temperature": 0.55, "max_tokens": 256, "seed": 44})
    original = copy.deepcopy(base)
    request, receipt = build_ablation_request(base, mode="minimal", canonical_system=SYSTEM)
    assert base == original
    assert request["messages"][0]["content"] == SYSTEM
    assert request["messages"][1:] == base["messages"][1:]
    assert request["temperature"] == 0.55
    assert request["max_tokens"] == 256
    assert request["seed"] == 44
    assert receipt["initial_turn"] is False
    assert receipt["later_turn_native_adaptations"] == ["max_tokens", "seed", "temperature"]


def test_later_absent_sampling_fields_remain_fixed_for_both_modes():
    base = fixture()
    base["messages"].append({"role": "tool", "content": "trial"})
    a, a_receipt = build_ablation_request(base, mode="control", canonical_system=SYSTEM)
    b, b_receipt = build_ablation_request(base, mode="minimal", canonical_system=SYSTEM)
    assert a_receipt["initial_turn"] is b_receipt["initial_turn"] is False
    assert a_receipt["later_turn_native_adaptations"] == []
    assert b_receipt["later_turn_native_adaptations"] == []
    for k, v in GENERATION.items():
        assert a[k] == b[k] == v
