from __future__ import annotations

import csv
from pathlib import Path

from lms_agent_bench.fleet_loadout import build_plan, normalize_tailscale_csv, select_loadout


def observation(backends=("vulkan", "cpu"), total=32 * 1024**3):
    return {
        "observation_fingerprint": "sha256:test",
        "hardware": {
            "supported_backends": list(backends),
            "cpu": {"logical_processors": 8, "cores_per_socket": 4},
            "memory": {"total_bytes": total},
            "accelerators": {},
        },
    }


def test_plan_generates_accelerated_and_cpu_candidates():
    models = [{"id": "tiny-q4", "size_bytes": 2 * 1024**3, "parameter_billions": 3, "max_context": 8192}]
    plan = build_plan(observation(), models, contexts=[4096], max_candidates=20)
    assert plan["artifact_type"] == "benchmark_plan"
    assert {item["backend"] for item in plan["candidates"]} == {"vulkan", "cpu"}
    assert all(item["launch"]["bind_host"] == "127.0.0.1" for item in plan["candidates"])
    assert len({item["candidate_id"] for item in plan["candidates"]}) == len(plan["candidates"])


def test_plan_rejects_model_that_exceeds_memory_budget():
    models = [{"id": "oversize", "size_bytes": 20 * 1024**3, "parameter_billions": 35, "max_context": 32768}]
    plan = build_plan(observation(total=16 * 1024**3), models, contexts=[32768], max_candidates=20)
    assert not plan["candidates"]
    assert plan["rejected_candidates"]
    assert plan["rejected_candidates"][0]["reason"] == "estimated_memory_exceeds_budget"


def test_selection_prefers_stable_candidate_over_faster_crashing_candidate():
    models = [{"id": "tiny", "size_bytes": 1024**3, "parameter_billions": 1, "max_context": 4096}]
    plan = build_plan(observation(backends=("cpu",)), models, contexts=[4096], max_candidates=2)
    first, second = plan["candidates"][:2]
    results = [
        {
            "candidate_id": first["candidate_id"],
            "ok_rate": "1",
            "eval_score_avg": "0.9",
            "tps_med": "40",
            "ttft_med": "0.5",
            "memory_headroom_ratio": "0.30",
            "concurrency_ok": "true",
            "streaming_ok": "true",
            "cancellation_ok": "true",
            "crash_count": "1",
        },
        {
            "candidate_id": second["candidate_id"],
            "ok_rate": "1",
            "eval_score_avg": "0.9",
            "tps_med": "20",
            "ttft_med": "1.0",
            "memory_headroom_ratio": "0.30",
            "concurrency_ok": "true",
            "streaming_ok": "true",
            "cancellation_ok": "true",
            "crash_count": "0",
        },
    ]
    selected = select_loadout(plan, results)
    assert selected["selected"]["candidate_id"] == second["candidate_id"]
    assert selected["admission"]["admitted"] is False


def test_tailscale_inventory_redacts_and_filters(tmp_path: Path):
    path = tmp_path / "nodes.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Device name", "Device ID", "Managed by", "Creator", "OS", "OS Version", "Domain", "Tailscale version", "Created", "Last seen", "Key expiry", "Tailscale IPs", "Tailscale SSH", "Funnel"])
        writer.writeheader()
        writer.writerow({"Device name": "x1-370", "Device ID": "secret", "Managed by": "mail", "Creator": "mail", "OS": "linux", "Tailscale IPs": "100.64.43.123,fd00::1"})
        writer.writerow({"Device name": "iphone-12", "Device ID": "secret2", "OS": "iOS", "Tailscale IPs": "100.1.2.3"})
    artifact = normalize_tailscale_csv(str(path))
    assert [node["node_id"] for node in artifact["nodes"]] == ["x1-370"]
    assert "Device ID" not in artifact["nodes"][0]
    assert artifact["redaction"]["creator_emails_removed"] is True
