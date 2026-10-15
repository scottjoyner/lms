"""Fail-closed test of pinned native Hermes preamble across tool interfaces."""
import copy
import pytest

from lms_agent_bench.hermes_native_tool_presentation import adapt_request


def envelope(tools):
    return {"model": "liquid-24b-parity-candidate", "messages": [
        {"role": "system", "content": "Native Hermes research instructions " * 50},
        {"role": "user", "content": "Use graph to report router-api owner"},
    ], "tools": [{"type": "function", "function": {"name": name}} for name in tools],
    "stream": True}


def test_arms_preserve_exact_system_but_change_tool_surface(tmp_path):
    file = tmp_path / "protected.txt"
    a = envelope(["tool_search", "tool_describe", "tool_call"])
    b = envelope(["mcp__lms_benchmark__graph_schema",
                  "mcp__lms_benchmark__graph_query"])
    first, a_receipt = adapt_request(a, mode="bridge", pinned=file)
    second, b_receipt = adapt_request(b, mode="direct", pinned=file)
    assert first["messages"] == second["messages"]
    assert first["tools"] != second["tools"]
    assert a_receipt["pinned_effective_system_sha256"] == b_receipt["pinned_effective_system_sha256"]
    assert file.stat().st_mode & 0o077 == 0
    assert first["temperature"] == second["temperature"] == 0.0
    assert first["max_tokens"] == second["max_tokens"] == 640


def test_bridge_preamble_drift_denied(tmp_path):
    file = tmp_path / "protected.txt"
    body = envelope(["tool_search"])
    adapt_request(body, mode="bridge", pinned=file)
    body = copy.deepcopy(body)
    body["messages"][0]["content"] += " drift"
    with pytest.raises(ValueError, match="drift"):
        adapt_request(body, mode="bridge", pinned=file)


def test_unqualified_mode_or_short_native_preamble_denied(tmp_path):
    with pytest.raises(ValueError, match="unexpected mode"):
        adapt_request(envelope([]), mode="production", pinned=tmp_path / "p")
    body = envelope([])
    body["messages"][0]["content"] = "short"
    with pytest.raises(ValueError, match="full native"):
        adapt_request(body, mode="bridge", pinned=tmp_path / "p")
