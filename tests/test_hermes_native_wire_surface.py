"""Verify real HTTPX serialization is recorded without prompt/header leakage."""
import json

import httpx

from lms_agent_bench.hermes_native_http_probe import observe_loopback_http


def test_metadata_only_model_surface_and_tool_contracts(tmp_path):
    path = tmp_path / "witness.jsonl"
    input_data = {
        "model": "research-only",
        "temperature": 0.0, "seed": 43, "max_tokens": 640, "stream": True,
        "messages": [
            {"role": "system", "content": "PRIVATE SYSTEM PROTOCOL"},
            {"role": "user", "content": "PRIVATE USER LOCATION"},
        ],
        "tools": [{"type": "function", "function": {
            "name": "mcp__lms_benchmark__graph_query",
            "description": "PRIVATE GRAPH TOOL DOC",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "read_only": {"type": "boolean"},
            }},
        }}],
    }
    def backend(request):
        assert request.headers.get("Authorization") == "Bearer PRIVATE_KEY"
        return httpx.Response(200, json={"choices": []})
    with httpx.Client(transport=httpx.MockTransport(backend)) as client:
        with observe_loopback_http(path, endpoint="http://127.0.0.1:12345/v1"):
            r = client.post("http://127.0.0.1:12345/v1/chat/completions",
                            json=input_data, headers={"Authorization": "Bearer PRIVATE_KEY"})
            assert r.status_code == 200
    x = json.loads(path.read_text())
    surface = x["wire_surface"]
    assert surface["model_message_roles"] == ["system", "user"]
    assert surface["advertised_tool_count"] == 1
    assert surface["advertised_tool_contracts"][0]["property_keys"] == ["query", "read_only"]
    assert surface["generation"] == {"max_tokens": 640, "seed": 43,
                                      "stream": True, "temperature": 0.0}
    assert len(surface["all_other_options_sha256"]) == 64
    for forbidden in ("PRIVATE SYSTEM", "PRIVATE USER", "PRIVATE GRAPH", "PRIVATE_KEY"):
        assert forbidden not in path.read_text()


def test_only_explicit_loopback_url_can_be_observed(tmp_path):
    path = tmp_path / "witness.jsonl"
    with httpx.Client(transport=httpx.MockTransport(lambda q: httpx.Response(200))) as c:
        with observe_loopback_http(path, endpoint="http://127.0.0.1:12345/v1"):
            c.post("http://127.0.0.1:12345/v2/chat/completions", json={"messages": []})
    assert not path.exists()
