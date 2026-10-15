"""Offline strict one-field native graph tool description control."""
from copy import deepcopy
import pytest
from lms_agent_bench.hermes_native_query_description import (
    EXAMPLE, ORIGINAL, adapt_description,
)
from lms_agent_bench.hermes_native_tool_order import CANONICAL


def input_envelope():
    return {"model": "liquid-24b-parity-candidate",
        "messages": [{"role": "system", "content": "native instruction " * 60},
                     {"role": "user", "content": "Find owner of router-api"}],
        "tools": [{"type": "function", "function": {
            "name": name,
            "description": ORIGINAL if name.endswith("__graph_query") else name,
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
        }} for name in CANONICAL],
        "stream": True}


def test_example_only_changes_query_tool_description(tmp_path):
    original = input_envelope()
    untouched = deepcopy(original)
    pinned = tmp_path / "private-preamble"
    a, rec_a = adapt_description(original, mode="baseline", pinned=pinned)
    b, rec_b = adapt_description(original, mode="example", pinned=pinned)
    assert original == untouched
    assert a["messages"] == b["messages"]
    assert rec_a["pinned_system_sha256"] == rec_b["pinned_system_sha256"]
    assert [x["function"]["name"] for x in a["tools"]] == list(CANONICAL)
    assert a["tools"][:1] == b["tools"][:1]
    assert a["tools"][2:] == b["tools"][2:]
    assert a["tools"][1]["function"]["description"] == ORIGINAL
    assert b["tools"][1]["function"]["description"] == EXAMPLE
    assert a["tools"][1]["function"]["parameters"] == b["tools"][1]["function"]["parameters"]
    assert "team-atlas" not in EXAMPLE
    assert pinned.stat().st_mode & 0o077 == 0
    assert a["seed"] == b["seed"] == 43
    assert a["max_tokens"] == b["max_tokens"] == 640


@pytest.mark.parametrize("mode", ["production", "", "tool_order"])
def test_unexpected_mode_fail_closed(tmp_path, mode):
    with pytest.raises(ValueError, match="research-only"):
        adapt_description(input_envelope(), mode=mode, pinned=tmp_path / "p")


def test_original_description_or_tool_scope_must_match(tmp_path):
    payload = input_envelope()
    payload["tools"][1]["function"]["description"] = "unknown"
    with pytest.raises(ValueError, match="unrecognized"):
        adapt_description(payload, mode="baseline", pinned=tmp_path / "p")
    payload = input_envelope()
    payload["tools"].pop()
    with pytest.raises(ValueError, match="nine-tool"):
        adapt_description(payload, mode="baseline", pinned=tmp_path / "p")


def test_baseline_preamble_drift_denied(tmp_path):
    p = tmp_path / "p"
    payload = input_envelope()
    adapt_description(payload, mode="baseline", pinned=p)
    payload["messages"][0]["content"] += " changed"
    with pytest.raises(ValueError, match="drift"):
        adapt_description(payload, mode="baseline", pinned=p)
