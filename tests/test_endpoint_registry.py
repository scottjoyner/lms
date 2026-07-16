import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lms_endpoint_registry as reg


def test_registry_add_and_select(tmp_path):
    path = tmp_path / "endpoints.json"
    data = reg.load_registry(path)
    data["endpoints"] = [
        {"name": "local", "base_url": "http://127.0.0.1:1234/v1", "enabled": True, "tags": ["local"]},
        {"name": "gpu", "base_url": "http://100.64.0.10:1234/v1", "enabled": False, "tags": ["gpu"]},
    ]
    reg.save_registry(path, data)
    loaded = reg.load_registry(path)
    selected = reg.select_endpoints(loaded, names=None, tags=["local"], enabled_only=True)
    assert len(selected) == 1
    assert selected[0]["name"] == "local"
    disabled = reg.select_endpoints(loaded, names=None, tags=["gpu"], enabled_only=True)
    assert disabled == []


def test_normalize_base_url():
    assert reg.normalize_base_url("127.0.0.1:1234") == "http://127.0.0.1:1234/v1"
    assert reg.normalize_base_url("http://x:1234/v1") == "http://x:1234/v1"


def test_export_inventory_with_mocked_probe(tmp_path, monkeypatch):
    registry = tmp_path / "endpoints.json"
    data = reg.load_registry(registry)
    data["endpoints"] = [{"name": "local", "base_url": "http://127.0.0.1:1234/v1", "enabled": True, "tags": ["local"]}]
    reg.save_registry(registry, data)

    monkeypatch.setattr(reg, "probe_endpoint", lambda base_url, timeout=8: {"reachable": True, "models": ["qwen-7b-q4"], "model_count": 1, "error": None})
    out = tmp_path / "inventory.csv"
    rc = reg.main(["--registry", str(registry), "export-inventory", "--out", str(out)])
    assert rc == 0
    assert "qwen-7b-q4" in out.read_text()


def test_build_tailscale_candidates_and_refresh(tmp_path, monkeypatch):
    status = {
        "Self": {
            "HostName": "x1-370",
            "TailscaleIPs": ["100.64.43.123"],
        },
        "Peer": {
            "node-key-1": {
                "HostName": "joyner",
                "TailscaleIPs": ["100.83.215.83"],
            },
            "node-key-2": {
                "HostName": "other-node",
                "TailscaleIPs": ["100.99.99.99"],
            },
        },
    }
    candidates = reg.build_tailscale_endpoint_candidates(status, include_self=True)
    assert any(c["name"] == "tailscale-joyner" for c in candidates)
    assert any(c["base_url"] == "http://100.83.215.83:1234/v1" for c in candidates)

    registry = tmp_path / "endpoints.json"
    monkeypatch.setattr(reg, "tailscale_status", lambda timeout=8: status)
    monkeypatch.setattr(reg, "probe_endpoint", lambda base_url, timeout=8: {
        "reachable": base_url == "http://100.83.215.83:1234/v1",
        "models": ["refinedtoolcallv5-3b"] if base_url == "http://100.83.215.83:1234/v1" else [],
        "model_count": 1 if base_url == "http://100.83.215.83:1234/v1" else 0,
        "error": None if base_url == "http://100.83.215.83:1234/v1" else "unreachable",
    })
    result = reg.refresh_tailscale_registry(registry, include_self=True)
    assert result["reachable"] == 1
    loaded = reg.load_registry(registry)
    names = [e["name"] for e in loaded["endpoints"]]
    assert "tailscale-joyner" in names
    joyner = next(e for e in loaded["endpoints"] if e["name"] == "tailscale-joyner")
    assert joyner["base_url"] == "http://100.83.215.83:1234/v1"
    assert joyner["enabled"] is True
