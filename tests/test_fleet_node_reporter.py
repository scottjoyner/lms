from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fleet_node_reporter",
    ROOT / "fleet_node_reporter.py",
)
assert SPEC and SPEC.loader
reporter = importlib.util.module_from_spec(SPEC)
sys.modules["fleet_node_reporter"] = reporter
SPEC.loader.exec_module(reporter)


def test_runtime_observations_cover_native_and_openai_compatible(monkeypatch):
    def fake_get(url: str, timeout: float = 5.0):
        if url == "http://localhost:1234/api/v1/models":
            return {
                "models": [
                    {
                        "key": "lm/native",
                        "loaded_instances": [{"id": "instance"}],
                    }
                ]
            }
        if url == "http://localhost:1234/api/v0/models":
            return {"data": [{"id": "lm/native", "path": "/models/native.gguf"}]}
        if url == "http://localhost:1234/v1/models":
            return {"data": [{"id": "lm/native"}]}

        if url == "http://localhost:1235/api/v1/models":
            return None
        if url == "http://localhost:1235/v1/models":
            return {"data": [{"id": "k2-36b"}]}

        if url == "http://localhost:38898/api/v1/models":
            return None
        if url == "http://localhost:38898/v1/models":
            return {"data": [{"id": "ternary-bonsai-2"}]}
        return None

    monkeypatch.setattr(reporter, "_http_get_json", fake_get)

    observations = reporter._runtime_observations(
        "http://localhost:1234",
        "destroyer",
        [
            "http://localhost:1235/v1",
            "http://localhost:38898",
            "http://localhost:1235",
        ],
    )

    assert len(observations) == 3
    by_url = {item["base_url"]: item for item in observations}

    native = by_url["http://localhost:1234"]
    assert native["runtime_kind"] == "lmstudio"
    assert native["protocol"] == "lmstudio-native"
    assert native["models"] == ["lm/native"]

    k2 = by_url["http://localhost:1235"]
    assert k2["runtime_kind"] == "openai_compatible"
    assert k2["models"] == ["k2-36b"]

    bonsai = by_url["http://localhost:38898"]
    assert bonsai["runtime_kind"] == "openai_compatible"
    assert bonsai["models"] == ["ternary-bonsai-2"]

    assert all(item["admitted"] is False for item in observations)
    assert all(item["ready"] is True for item in observations)
    assert all(item["observed_model_count"] == 1 for item in observations)
    assert all(
        item["runtime_observation_id"].startswith("runtime-observation:")
        for item in observations
    )


def test_configured_runtime_urls_merge_env_and_cli_without_duplicates(monkeypatch):
    monkeypatch.setenv(
        "FLEET_RUNTIME_URLS",
        "http://localhost:1236/v1,http://localhost:38898",
    )

    urls = reporter._configured_runtime_urls(
        "http://localhost:1234",
        [
            "http://localhost:1236",
            "localhost:1235",
        ],
    )

    assert urls == [
        "http://localhost:1234",
        "http://localhost:1236",
        "http://localhost:1235",
        "http://localhost:38898",
    ]


def test_build_report_keeps_legacy_fields_and_adds_non_admitting_runtime_evidence(
    monkeypatch,
):
    monkeypatch.setattr(reporter.socket, "gethostname", lambda: "x1-370")
    monkeypatch.setattr(
        reporter,
        "_specs",
        lambda: {"hostname": "x1-370", "system_ram_gib": 96.0},
    )
    monkeypatch.setattr(
        reporter,
        "_library",
        lambda _url: [{"id": "legacy-library-model", "path": "/model.gguf"}],
    )
    monkeypatch.setattr(
        reporter,
        "_loaded_models",
        lambda _url: ["legacy-loaded-model"],
    )
    monkeypatch.setattr(
        reporter,
        "_runtime_observations",
        lambda _url, _hostname, _extra, _witnesses=None: [
            {
                "observation_schema": "fleet-runtime-observation.v1",
                "runtime_observation_id": "runtime-observation:abc",
                "runtime_kind": "openai_compatible",
                "protocol": "openai-compatible",
                "base_url": "http://localhost:1235",
                "models": ["k2-36b"],
                "ready": True,
                "observed_at": 1,
                "admitted": False,
            }
        ],
    )

    report = reporter.build_report(
        "http://localhost:1234",
        ["http://localhost:1235"],
    )

    assert report["hostname"] == "x1-370"
    assert report["library"][0]["id"] == "legacy-library-model"
    assert report["loaded"] == ["legacy-loaded-model"]
    assert report["runtimes"][0]["models"] == ["k2-36b"]
    assert report["runtimes"][0]["admitted"] is False


def test_empty_model_list_is_visible_but_not_ready(monkeypatch):
    def fake_get(url: str, timeout: float = 5.0):
        if url == "http://localhost:1235/api/v1/models":
            return None
        if url == "http://localhost:1235/v1/models":
            return {"data": []}
        return None

    monkeypatch.setattr(reporter, "_http_get_json", fake_get)

    observation = reporter._runtime_observation(
        "http://localhost:1235",
        "destroyer",
    )

    assert observation is not None
    assert observation["models"] == []
    assert observation["observed_model_count"] == 0
    assert observation["ready"] is False
    assert observation["admitted"] is False


def test_runtime_observation_attaches_signed_witness_evidence(monkeypatch):
    def fake_get(url: str, timeout: float = 5.0):
        if url == "http://localhost:1235/api/v1/models":
            return None
        if url == "http://localhost:1235/v1/models":
            return {"data": [{"id": "k2-36b"}]}
        return None

    monkeypatch.setattr(reporter, "_http_get_json", fake_get)
    monkeypatch.setattr(
        reporter._runtime_witness,
        "observe_process_continuity",
        lambda _witness: {
            "valid": True,
            "reason": "match",
            "checked_at": 123,
            "pid": 42,
            "boot_id": "boot",
            "process_start_ticks": 99,
            "executable_basename": "server",
        },
    )
    witness = {
        "schema_version": "fleet-runtime-identity-witness.v1",
        "runtime_url": "http://localhost:1235",
        "provider_model": "k2-36b",
    }
    payload = json.dumps(
        witness,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ) + "\n"

    observation = reporter._runtime_observation(
        "http://localhost:1235",
        "destroyer",
        {
            "witness": witness,
            "payload": payload,
            "signature": "-----BEGIN SSH SIGNATURE-----\nabc\n-----END SSH SIGNATURE-----\n",
        },
    )

    assert observation is not None
    assert observation["runtime_identity_witness_json"] == payload
    assert "BEGIN SSH SIGNATURE" in observation["runtime_identity_witness_signature"]
    assert observation["runtime_identity_continuity"]["valid"] is True
    assert observation["admitted"] is False
