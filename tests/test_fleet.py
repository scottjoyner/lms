import json

from lms_agent_bench import fleet, fleet_orchestrator as orch


def test_build_state_no_hang_and_counts():
    """build_state must not probe the whole tailscale peer set (which used to
    hang for minutes on unreachable peers) and must include all benchmarked
    nodes from runs/."""
    state = fleet.build_state()
    assert state["node_count"] >= 1
    # every node in state must correspond to a runs/<node>/run_summary.csv
    for n in state["nodes"]:
        assert (fleet.RUNS / n["name"] / "run_summary.csv").exists()
    # model index should be non-empty when artifacts exist
    assert len(state["model_best_node"]) >= 1


def test_routing_rules_shape():
    fleet.cmd_routes()
    rules = json.loads(fleet.ROUTES_JSON.read_text(encoding="utf-8"))
    assert "nodes" in rules and "routing" in rules
    for nr in rules["nodes"]:
        assert "name" in nr and "max_concurrency" in nr


def test_loadout_command_reads_fleet_loadout(tmp_path, monkeypatch):
    """cmd_loadout must consume fleet_loadout.json and produce plan items keyed
    by node slug, reusing apply_loadouts' shape (node/ip/actions)."""
    loadout = {
        "generated_at": "2026-07-16T00:00:00+00:00",
        "demand": "balanced",
        "nodes": {
            "x1-370": {"url": "http://127.0.0.1:1234/v1", "max_concurrency": 2,
                       "mount": ["liquid/lfm2.5-1.2b"], "ram_gib": 91.9, "vram_gib": None},
        },
        "routing": {},
    }
    p = tmp_path / "fleet_loadout.json"
    p.write_text(json.dumps(loadout))
    monkeypatch.setattr(orch, "FLEET_LOADOUT", p)

    # capture the plan cmd_loadout would build by stubbing apply_loadouts
    captured = {}
    def fake_apply(plan, dry_run=True):
        captured["plan"] = plan
    monkeypatch.setattr(orch, "apply_loadouts", fake_apply)
    # discover_nodes may reach the network; stub to return our one node
    monkeypatch.setattr(orch, "discover_nodes", lambda: [
        {"slug": "x1-370", "hostname": "x1-370", "ip": "127.0.0.1", "online": True}])
    monkeypatch.setattr(orch, "probe_node", lambda n: {
        "reachable": True, "loaded_models": ["liquid/lfm2.5-1.2b"]})
    monkeypatch.setattr(orch, "load_busy_map", lambda: {})

    class Args:
        apply = False
        only = None
    orch.cmd_loadout(Args())
    assert "plan" in captured
    assert any(item["node"] == "x1-370" for item in captured["plan"])
