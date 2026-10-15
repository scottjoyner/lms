"""No network, Docker, model or live graph: tool ordering isolation guards."""
from copy import deepcopy

import pytest

from lms_agent_bench.hermes_native_tool_order import (
    ALPHABETICAL, CANONICAL, adapt_order,
)


def input_envelope():
    tools = [{"type": "function", "function": {"name": x, "description": x}}
             for x in reversed(CANONICAL)]
    return {
        "model": "liquid-24b-parity-candidate",
        "tools": tools, "messages": [
            {"role": "system", "content": "isolated Hermes instruction " * 45},
            {"role": "user", "content": "Who owns router-api?"},
        ], "stream": True,
    }


def test_only_order_changes(tmp_path):
    pinned = tmp_path / "private-preamble.txt"
    body = input_envelope()
    original = deepcopy(body)
    canonical, a = adapt_order(body, mode="canonical", pinned=pinned)
    alphabetical, b = adapt_order(body, mode="alphabetical", pinned=pinned)
    assert tuple(x["function"]["name"] for x in canonical["tools"]) == CANONICAL
    assert tuple(x["function"]["name"] for x in alphabetical["tools"]) == ALPHABETICAL
    assert canonical["messages"] == alphabetical["messages"]
    assert a["pinned_system_sha256"] == b["pinned_system_sha256"]
    assert a["unordered_contract_hash"] == b["unordered_contract_hash"]
    assert a["ordered_contract_hash"] != b["ordered_contract_hash"]
    assert canonical["temperature"] == alphabetical["temperature"] == 0.0
    assert canonical["seed"] == alphabetical["seed"] == 43
    assert canonical["max_tokens"] == alphabetical["max_tokens"] == 640
    assert body == original
    assert pinned.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("bad_mode", ["production", "bridge", "", "direct"])
def test_unexpected_modes_denied(tmp_path, bad_mode):
    with pytest.raises(ValueError, match="unsupported"):
        adapt_order(input_envelope(), mode=bad_mode, pinned=tmp_path / "p")


def test_duplicate_or_missing_tools_denied(tmp_path):
    body = input_envelope()
    body["tools"].append(deepcopy(body["tools"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        adapt_order(body, mode="canonical", pinned=tmp_path / "p")
    body["tools"] = body["tools"][:-2]
    with pytest.raises(ValueError, match="nine exact"):
        adapt_order(body, mode="canonical", pinned=tmp_path / "p")


def test_private_preamble_drift_fails_closed(tmp_path):
    pinned = tmp_path / "p"
    body = input_envelope()
    adapt_order(body, mode="canonical", pinned=pinned)
    body["messages"][0]["content"] += " drift"
    with pytest.raises(ValueError, match="changed"):
        adapt_order(body, mode="canonical", pinned=pinned)
