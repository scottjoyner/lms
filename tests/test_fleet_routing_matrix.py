from __future__ import annotations

import json

from lms_agent_bench.fleet_routing_matrix_public import build_routing_matrix


def _status() -> dict:
    return {
        "Self": {
            "HostName": "x1-370",
            "DNSName": "x1-370.example.ts.net.",
            "TailscaleIPs": ["100.64.0.10"],
            "OS": "linux",
            "Online": True,
            "Active": True,
        },
        "Peer": {
            "peer-a": {
                "HostName": "scott-optiplex-9030-aio",
                "DNSName": "scott-optiplex-9030-aio.example.ts.net.",
                "TailscaleIPs": ["100.64.0.11"],
                "OS": "linux",
                "Online": True,
            },
            "peer-b": {
                "HostName": "iphone-12-pro-max",
                "DNSName": "iphone-12-pro-max.example.ts.net.",
                "TailscaleIPs": ["100.64.0.12"],
                "OS": "ios",
                "Online": True,
            },
            "peer-c": {
                "HostName": "joyner",
                "DNSName": "joyner.example.ts.net.",
                "TailscaleIPs": ["100.64.0.13"],
                "OS": "linux",
                "Online": False,
            },
        },
    }


def _policy() -> dict:
    return {
        "nodes": {
            "x1-370": {
                "roles": ["full_agent"],
                "worker_mode": "agent",
                "allow_agent_runtime": True,
                "max_concurrent": 2,
            },
            "scott-optiplex-9030-aio": {
                "roles": ["auxiliary_llm", "compression", "summarization"],
                "worker_mode": "auxiliary",
                "allow_agent_runtime": False,
                "max_concurrent": 1,
            },
            "iphone-12-pro-max": {
                "roles": ["observer"],
                "worker_mode": "observer_only",
                "benchmark_policy": "excluded",
            },
            "joyner": {
                "roles": ["benchmark_only"],
                "worker_mode": "benchmark_only",
                "benchmark_policy": "benchmark_deferred",
            },
        }
    }


def _comparison() -> dict:
    return {
        "schema_version": "model_loadout_comparison.v1",
        "rows": [
            {
                "node_id": "x1-370",
                "model_id": "qwen-agent",
                "loadout_fingerprint": "sha256:agent",
                "qualified": True,
                "overall_task_pass_rate": 0.91,
                "effect_checkpoint_rate": 0.95,
                "quality_confidence": 1.0,
                "completion_tokens_per_second_end_to_end": 14.0,
                "task_families": ["coding", "reasoning", "tool_use"],
            },
            {
                "node_id": "scott-optiplex-9030-aio",
                "model_id": "small-summary",
                "loadout_fingerprint": "sha256:summary",
                "qualified": True,
                "overall_task_pass_rate": 0.72,
                "effect_checkpoint_rate": 0.92,
                "quality_confidence": 1.0,
                "completion_tokens_per_second_end_to_end": 28.0,
                "task_families": ["summarization", "compression", "extraction"],
            },
        ],
    }


def test_every_tailnet_node_is_visible_without_becoming_routable() -> None:
    matrix = build_routing_matrix(
        _status(),
        role_policy=_policy(),
        benchmark_documents=[_comparison()],
    )

    nodes = {row["node_id"]: row for row in matrix["nodes"]}
    assert set(nodes) == {
        "x1-370",
        "scott-optiplex-9030-aio",
        "iphone-12-pro-max",
        "joyner",
    }
    assert nodes["iphone-12-pro-max"]["tailnet_discovered"] is True
    assert nodes["iphone-12-pro-max"]["worker_mode"] == "observer_only"
    assert nodes["iphone-12-pro-max"]["allow_agent_runtime"] is False
    assert matrix["summary"]["tailnet_nodes"] == 4
    assert json.loads(json.dumps(matrix))["schema_version"] == "fleet_routing_matrix.v1"


def test_auxiliary_node_routes_summary_but_not_full_coding_agent() -> None:
    matrix = build_routing_matrix(
        _status(),
        role_policy=_policy(),
        benchmark_documents=[_comparison()],
    )

    summary = matrix["rankings"]["summarization"]
    coding = matrix["rankings"]["coding"]
    assert summary[0]["node_id"] == "scott-optiplex-9030-aio"
    assert all(row["node_id"] != "scott-optiplex-9030-aio" for row in coding)
    assert coding[0]["node_id"] == "x1-370"


def test_quality_floor_applies_before_speed() -> None:
    comparison = _comparison()
    comparison["rows"].append(
        {
            "node_id": "scott-optiplex-9030-aio",
            "model_id": "fast-bad-summary",
            "qualified": True,
            "overall_task_pass_rate": 0.20,
            "effect_checkpoint_rate": 0.95,
            "quality_confidence": 1.0,
            "completion_tokens_per_second_end_to_end": 100.0,
            "task_families": ["summarization"],
        }
    )
    matrix = build_routing_matrix(
        _status(),
        role_policy=_policy(),
        benchmark_documents=[comparison],
    )

    assert all(
        row["model_id"] != "fast-bad-summary"
        for row in matrix["rankings"]["summarization"]
    )
