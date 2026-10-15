"""Generic query-scaffolding A/B negative controls: strict one-field variation."""
from copy import deepcopy

import pytest

from lms_agent_bench.hermes_native_generic_query import (
    EXAMPLE, GENERIC, adapt_generic,
)
from lms_agent_bench.hermes_native_query_description import ORIGINAL
from lms_agent_bench.hermes_native_tool_order import CANONICAL


def envelope():
    return {"model": "liquid-24b-parity-candidate",
            "messages": [
                {"role": "system", "content": "native system instructions " * 44},
                {"role": "user", "content": "Who owns router-api?"},
            ],
            "tools": [{"type": "function", "function": {
                "name": name,
                "description": ORIGINAL if name.endswith("__graph_query") else name,
                "parameters": {"type": "object",
                               "properties": {"query": {"type": "string"}}},
            }} for name in CANONICAL],
            "stream": True}


def test_only_description_specificity_changes(tmp_path):
    p = tmp_path / "private-native-preamble"
    initial = envelope()
    unchanged = deepcopy(initial)
    a, ra = adapt_generic(initial, mode="specific", pinned=p)
    b, rb = adapt_generic(initial, mode="generic", pinned=p)
    assert initial == unchanged
    assert a["messages"] == b["messages"]
    assert ra["pinned_system_sha256"] == rb["pinned_system_sha256"]
    assert a["tools"][:1] == b["tools"][:1]
    assert a["tools"][2:] == b["tools"][2:]
    assert a["tools"][1]["function"]["parameters"] == b["tools"][1]["function"]["parameters"]
    assert a["tools"][1]["function"]["description"] == EXAMPLE
    assert b["tools"][1]["function"]["description"] == GENERIC
    assert "router-api" in EXAMPLE
    assert "router-api" not in GENERIC
    assert "team-atlas" not in GENERIC
    assert "<SERVICE_NAME>" in GENERIC
    assert p.stat().st_mode & 0o077 == 0
    assert a["temperature"] == b["temperature"] == 0.0
    assert a["seed"] == b["seed"] == 43
    assert a["max_tokens"] == b["max_tokens"] == 640


@pytest.mark.parametrize("bad", ["", "production", "baseline", "other"])
def test_out_of_scope_denied(tmp_path, bad):
    with pytest.raises(ValueError, match="unsupported"):
        adapt_generic(envelope(), mode=bad, pinned=tmp_path / "p")


def test_description_or_scope_drift_denied(tmp_path):
    broken = envelope()
    broken["tools"][1]["function"]["description"] = "weakened read-only policy"
    with pytest.raises(ValueError, match="unrecognized original"):
        adapt_generic(broken, mode="specific", pinned=tmp_path / "p")
    missing = envelope()
    missing["tools"].pop()
    with pytest.raises(ValueError, match="nine-tool"):
        adapt_generic(missing, mode="specific", pinned=tmp_path / "p")


def test_preamble_is_immutable_within_specific_arm(tmp_path):
    p = tmp_path / "p"
    original = envelope()
    adapt_generic(original, mode="specific", pinned=p)
    original["messages"][0]["content"] += " drift"
    with pytest.raises(ValueError, match="drift"):
        adapt_generic(original, mode="specific", pinned=p)
