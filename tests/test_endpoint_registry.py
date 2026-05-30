import json
from pathlib import Path

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
