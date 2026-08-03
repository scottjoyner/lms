from __future__ import annotations

import json
from pathlib import Path

import pytest

from lms_agent_bench.fleet_coverage import (
    enforce_rollout_coverage,
    validate_census,
    validate_rollout_coverage,
)


def write_json(path: Path, value) -> str:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return str(path)


def census(devices):
    return {
        "schema_version": "fleet_benchmark_census.v1",
        "devices": devices,
    }


def device(node_id, policy="benchmark_required", reason=None):
    value = {
        "node_id": node_id,
        "os_family": "linux" if node_id != "phone" else "ios",
        "benchmark_policy": policy,
        "benchmark_class": "test",
    }
    if reason is not None:
        value["reason"] = reason
    return value


def rollout(census_name, nodes, mode="full"):
    return {
        "schema_version": "fleet_rollout.v1",
        "coverage_mode": mode,
        "census_file": census_name,
        "nodes": [{"node_id": node_id} for node_id in nodes],
    }


def test_full_coverage_accounts_for_required_and_unsupported_devices(tmp_path):
    census_path = tmp_path / "census.json"
    write_json(
        census_path,
        census(
            [
                device("node-a"),
                device("node-b"),
                device("phone", "unsupported", "no remote runner"),
            ]
        ),
    )
    config_path = tmp_path / "rollout.json"
    config = rollout(census_path.name, ["node-a", "node-b"])
    write_json(config_path, config)

    report = validate_rollout_coverage(config, str(config_path))
    assert report["ready"] is True
    assert report["coverage_complete"] is True
    assert report["fleet_device_count"] == 3
    assert report["benchmark_required_count"] == 2
    assert report["configured_benchmark_count"] == 2
    assert report["accounted_device_count"] == 3
    assert report["unsupported_node_ids"] == ["phone"]


def test_full_coverage_fails_when_required_node_is_missing(tmp_path):
    census_path = tmp_path / "census.json"
    write_json(census_path, census([device("node-a"), device("node-b")]))
    config_path = tmp_path / "rollout.json"
    config = rollout(census_path.name, ["node-a"])
    write_json(config_path, config)

    report = validate_rollout_coverage(config, str(config_path))
    assert report["ready"] is False
    assert report["missing_required_node_ids"] == ["node-b"]
    with pytest.raises(ValueError, match="missing required benchmark nodes"):
        enforce_rollout_coverage(config, str(config_path))


def test_partial_coverage_is_explicit_but_not_complete(tmp_path):
    census_path = tmp_path / "census.json"
    write_json(census_path, census([device("node-a"), device("node-b")]))
    config_path = tmp_path / "rollout.json"
    config = rollout(census_path.name, ["node-a"], mode="partial")
    write_json(config_path, config)

    report = validate_rollout_coverage(config, str(config_path))
    assert report["ready"] is True
    assert report["coverage_complete"] is False
    assert report["missing_required_node_ids"] == ["node-b"]


def test_unsupported_device_cannot_be_benchmarked_as_a_rollout_node(tmp_path):
    census_path = tmp_path / "census.json"
    write_json(
        census_path,
        census([device("phone", "unsupported", "no remote runner")]),
    )
    config_path = tmp_path / "rollout.json"
    config = rollout(census_path.name, ["phone"])
    write_json(config_path, config)

    report = validate_rollout_coverage(config, str(config_path))
    assert report["ready"] is False
    assert report["unsupported_configured_node_ids"] == ["phone"]


def test_unsupported_policy_requires_a_reason():
    with pytest.raises(ValueError, match="requires a reason"):
        validate_census(census([device("phone", "unsupported")]))


def test_canonical_full_fleet_template_covers_current_census():
    config_path = Path("examples/fleet-rollout.full-fleet.template.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = validate_rollout_coverage(config, str(config_path))

    assert report["ready"] is True
    assert report["coverage_complete"] is True
    assert report["fleet_device_count"] == 11
    assert report["benchmark_required_count"] == 10
    assert report["configured_benchmark_count"] == 10
    assert report["accounted_device_count"] == 11
    assert report["unsupported_node_ids"] == ["iphone-12-pro-max"]
    assert set(report["configured_node_ids"]) == {
        "destroyer",
        "raspberrypi",
        "beelink-ryzen-7-mini-pc",
        "deathstar-xps-8920",
        "scott-lenovo-ideapad-330s-15ikb",
        "scott-optiplex-9030-aio",
        "scotts-macbook-air",
        "scotts-macbook-pro-2",
        "x1-370",
        "xwing",
    }


def test_tier1_template_is_marked_partial_and_reports_deferred_nodes():
    config_path = Path("examples/fleet-rollout.tier1.template.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = validate_rollout_coverage(config, str(config_path))

    assert report["coverage_mode"] == "partial"
    assert report["ready"] is True
    assert report["coverage_complete"] is False
    assert report["configured_benchmark_count"] == 3
    assert len(report["missing_required_node_ids"]) == 7
