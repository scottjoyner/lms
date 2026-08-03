import csv
import json

from lms_agent_bench import fleet, fleet_orchestrator as orch


def seed_benchmarked_fleet(tmp_path, monkeypatch):
    """Create one deterministic benchmark artifact without touching the network."""
    runs = tmp_path / "runs"
    node_dir = runs / "ci-node"
    node_dir.mkdir(parents=True)
    with (node_dir / "run_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_key", "tps_med", "ttft_med", "eval_score_avg"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_key": "ci/model-q4",
                "tps_med": "20.0",
                "ttft_med": "0.25",
                "eval_score_avg": "0.90",
            }
        )

    monkeypatch.setattr(fleet, "RUNS", runs)
    monkeypatch.setattr(fleet, "STATE_JSON", tmp_path / "fleet_state.json")
    monkeypatch.setattr(fleet, "ROUTES_JSON", tmp_path / "routing_rules.json")
    monkeypatch.setattr(fleet, "discover", lambda: [])
    monkeypatch.setattr(fleet, "all_aliases", lambda _nodes: {})
    monkeypatch.setattr(fleet, "live_nodes", lambda _nodes: [])
    return node_dir


def test_build_state_no_hang_and_counts(tmp_path, monkeypatch):
    """Only benchmarked artifacts are indexed, without probing unrelated peers."""
    node_dir = seed_benchmarked_fleet(tmp_path, monkeypatch)
    state = fleet.build_state()

    assert state["node_count"] == 1
    assert state["nodes"][0]["name"] == "ci-node"
    assert (node_dir / "run_summary.csv").exists()
    assert state["model_best_node"]["ci/model-q4"]["best_node"] == "ci-node"


def test_routing_rules_shape(tmp_path, monkeypatch):
    seed_benchmarked_fleet(tmp_path, monkeypatch)
    fleet.cmd_routes()
    rules = json.loads(fleet.ROUTES_JSON.read_text(encoding="utf-8"))

    assert "nodes" in rules and "routing" in rules
    assert rules["routing"]["ci/model-q4"]["node"] == "ci-node"
    for node_rule in rules["nodes"]:
        assert "name" in node_rule and "max_concurrency" in node_rule


def test_loadout_command_reads_fleet_loadout(tmp_path, monkeypatch):
    """cmd_loadout consumes fleet_loadout.json and builds node-keyed plan items."""
    loadout = {
        "generated_at": "2026-07-16T00:00:00+00:00",
        "demand": "balanced",
        "nodes": {
            "x1-370": {
                "url": "http://127.0.0.1:1234/v1",
                "max_concurrency": 2,
                "mount": ["liquid/lfm2.5-1.2b"],
                "ram_gib": 91.9,
                "vram_gib": None,
            },
        },
        "routing": {},
    }
    path = tmp_path / "fleet_loadout.json"
    path.write_text(json.dumps(loadout))
    monkeypatch.setattr(orch, "FLEET_LOADOUT", path)

    captured = {}

    def fake_apply(plan, dry_run=True):
        captured["plan"] = plan

    monkeypatch.setattr(orch, "apply_loadouts", fake_apply)
    monkeypatch.setattr(
        orch,
        "discover_nodes",
        lambda: [
            {
                "slug": "x1-370",
                "hostname": "x1-370",
                "ip": "127.0.0.1",
                "online": True,
            }
        ],
    )
    monkeypatch.setattr(
        orch,
        "probe_node",
        lambda _node: {
            "reachable": True,
            "loaded_models": ["liquid/lfm2.5-1.2b"],
        },
    )
    monkeypatch.setattr(orch, "load_busy_map", lambda: {})

    class Args:
        apply = False
        only = None

    orch.cmd_loadout(Args())
    assert "plan" in captured
    assert any(item["node"] == "x1-370" for item in captured["plan"])
