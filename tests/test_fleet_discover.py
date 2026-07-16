"""Smoke tests for lms_agent_bench.fleet_discover (no live tailscale/fleet)."""

from pathlib import Path


from lms_agent_bench import fleet_discover


def test_discover_returns_list(monkeypatch):
    """With no fleet.toml and no tailscale, discover() must not raise and must
    return a list (possibly empty)."""
    monkeypatch.setattr(fleet_discover, "FLEET_TOML", Path("/nonexistent/fleet.toml"))
    # tailscale unavailable -> _discover_tailscale returns []
    nodes = fleet_discover.discover()
    assert isinstance(nodes, list)


def test_reachable_retries_and_fails_closed():
    """_reachable must return False (not raise) when the endpoint is down."""
    assert fleet_discover._reachable("http://127.0.0.1:1/v1", retries=1, backoff=0.01) is False


def test_all_aliases_maps_names_and_aliases():
    n = fleet_discover.Node(name="x1", url="http://127.0.0.1:1234/v1", aliases=["x-one"])
    m = fleet_discover.all_aliases([n])
    assert m["x1"] == "x1"
    assert m["x-one"] == "x1"
