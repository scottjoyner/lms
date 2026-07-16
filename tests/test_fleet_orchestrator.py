"""Smoke tests for lms_agent_bench.fleet_orchestrator (no live fleet/neo4j)."""

from lms_agent_bench import fleet_orchestrator as orch


def test_slug_normalizes():
    assert orch._slug("Scott's MacBook Air") == "scott-s-macbook-air"
    assert orch._slug("x1-370") == "x1-370"


def test_match_cap_exact_and_normalized():
    caps = {"deathstar": object(), "lenovo-ideapad": object()}
    assert orch._match_cap(caps, "deathstar") is caps["deathstar"]
    # -/_ normalization must still join (no substring hack)
    assert orch._match_cap(caps, "lenovo_ideapad") is caps["lenovo-ideapad"]
    assert orch._match_cap(caps, "death") is None  # substring hack removed


def test_params_b_parses():
    assert orch._params_b({"params": "9B"}) == 9.0
    assert orch._params_b({"params": "35B-A3B"}) == 35.0
    assert orch._params_b({"params": "128x2.6B"}) == 2.6  # multi-part parsed


def test_conservative_from_library_skips_diffusion_and_picks_tiers():
    lib = [
        {"key": "emb/model", "type": "embedding"},
        {"key": "small/a", "params": "1.2B"},
        {"key": "mid/b", "params": "9B"},
        {"key": "diff/c", "params": "128x2.6B"},
    ]
    want = orch.conservative_from_library(lib, big_node=False)
    assert "emb/model" in want
    assert "small/a" in want
    assert "diff/c" not in want


def test_discover_nodes_no_crash_without_fleet(monkeypatch):
    """discover_nodes must return a list even when fleet.toml is absent and
    tailscale is unavailable (no live fleet required)."""
    monkeypatch.setattr(orch, "_discover_fleet_nodes", lambda: [])
    nodes = orch.discover_nodes()
    assert isinstance(nodes, list)
