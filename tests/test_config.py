import json

import lms_config


def test_effective_config_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("LMS_BENCH_ENDPOINT", raising=False)
    cfg = lms_config.effective_config(tmp_path / "missing.json")
    assert cfg["values"]["default_endpoint"] == "http://127.0.0.1:1234/v1"
    assert cfg["sources"]["default_endpoint"] == "default"


def test_effective_config_file_and_env_precedence(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"default_endpoint": "http://file:1234/v1", "repeats": 2}))
    monkeypatch.setenv("LMS_BENCH_ENDPOINT", "http://env:1234/v1")
    cfg = lms_config.effective_config(path)
    assert cfg["values"]["default_endpoint"] == "http://env:1234/v1"
    assert cfg["values"]["repeats"] == 2
    assert cfg["sources"]["default_endpoint"] == "env:LMS_BENCH_ENDPOINT"
    assert cfg["sources"]["repeats"] == str(path)


def test_validate_config_rejects_bad_values(tmp_path):
    cfg = {"values": dict(lms_config.DEFAULTS)}
    cfg["values"]["repeats"] = 0
    result = lms_config.validate_config(cfg)
    assert result["ok"] is False
    assert any("repeats" in err for err in result["errors"])


def test_init_config(tmp_path):
    path = tmp_path / "config.json"
    rc = lms_config.main(["--config", str(path), "init"])
    assert rc == 0
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["schema_version"] == "lms_bench_config.v1"
