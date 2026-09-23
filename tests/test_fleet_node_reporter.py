from __future__ import annotations

import fleet_node_reporter as reporter


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
        lambda _url, _hostname, _extra: [
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
