"""Fail-closed fleet census coverage for physical benchmark rollouts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

CENSUS_SCHEMA_VERSION = "fleet_benchmark_census.v1"
_ALLOWED_POLICIES = {
    "benchmark_required",
    "benchmark_deferred",
    "adapter_required",
    "unsupported",
}
_ALLOWED_MODES = {"full", "partial"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_census_path(config_path: str, census_value: str) -> Path:
    path = Path(census_value)
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve()


def validate_census(census: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if census.get("schema_version") != CENSUS_SCHEMA_VERSION:
        raise ValueError(f"fleet census schema must be {CENSUS_SCHEMA_VERSION}")
    devices = census.get("devices")
    if not isinstance(devices, list) or not devices:
        raise ValueError("fleet census requires a non-empty devices list")
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, raw in enumerate(devices):
        if not isinstance(raw, Mapping):
            raise ValueError(f"fleet census device {index} must be an object")
        node_id = str(raw.get("node_id") or "").strip()
        policy = str(raw.get("benchmark_policy") or "").strip()
        os_family = str(raw.get("os_family") or "").strip()
        if not node_id:
            raise ValueError(f"fleet census device {index} has no node_id")
        if node_id in by_id:
            raise ValueError(f"duplicate fleet census node_id: {node_id}")
        if policy not in _ALLOWED_POLICIES:
            raise ValueError(
                f"fleet census node {node_id} has invalid benchmark_policy"
            )
        if not os_family:
            raise ValueError(f"fleet census node {node_id} has no os_family")
        reason = str(raw.get("reason") or "").strip()
        if policy != "benchmark_required" and not reason:
            raise ValueError(f"{policy} fleet census node {node_id} requires a reason")
        by_id[node_id] = dict(raw)
    return by_id


def validate_rollout_coverage(
    config: Mapping[str, Any], config_path: str
) -> Dict[str, Any]:
    mode = str(config.get("coverage_mode") or "unmanaged")
    census_value = str(config.get("census_file") or "").strip()
    nodes = config.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("rollout configuration requires nodes before coverage")
    configured = {
        str(item.get("node_id") or "")
        for item in nodes
        if isinstance(item, Mapping)
    }

    if mode == "unmanaged" and not census_value:
        return {
            "schema_version": "fleet_rollout_coverage.v1",
            "coverage_mode": "unmanaged",
            "coverage_enforced": False,
            "coverage_complete": False,
            "benchmark_interface_complete": False,
            "current_execution_scope_complete": False,
            "configured_node_ids": sorted(configured),
            "benchmark_required_node_ids": [],
            "benchmark_deferred_node_ids": [],
            "adapter_required_node_ids": [],
            "unsupported_node_ids": [],
            "missing_required_node_ids": [],
            "unexpected_node_ids": [],
            "non_rollout_configured_node_ids": [],
            "fleet_device_count": None,
            "benchmark_required_count": None,
            "benchmark_deferred_count": None,
            "adapter_required_count": None,
            "unsupported_count": None,
            "configured_benchmark_count": len(configured),
            "accounted_device_count": None,
            "ready": True,
            "admission": {"admitted": False},
        }
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            "coverage_mode must be full or partial when census_file is set"
        )
    if not census_value:
        raise ValueError("coverage-managed rollout requires census_file")

    census_path = _resolve_census_path(config_path, census_value)
    census = load_json(census_path)
    if not isinstance(census, Mapping):
        raise ValueError("fleet census must be a JSON object")
    by_id = validate_census(census)
    all_ids = set(by_id)
    required = {
        node_id
        for node_id, item in by_id.items()
        if item["benchmark_policy"] == "benchmark_required"
    }
    deferred = {
        node_id
        for node_id, item in by_id.items()
        if item["benchmark_policy"] == "benchmark_deferred"
    }
    adapter_required = {
        node_id
        for node_id, item in by_id.items()
        if item["benchmark_policy"] == "adapter_required"
    }
    unsupported = {
        node_id
        for node_id, item in by_id.items()
        if item["benchmark_policy"] == "unsupported"
    }
    non_rollout = deferred | adapter_required | unsupported
    missing = required - configured
    unexpected = configured - all_ids
    non_rollout_configured = configured & non_rollout
    accounted = (configured & required) | non_rollout
    complete = not missing and not unexpected and not non_rollout_configured
    ready = not unexpected and not non_rollout_configured
    if mode == "full":
        ready = ready and complete

    report = {
        "schema_version": "fleet_rollout_coverage.v1",
        "coverage_mode": mode,
        "coverage_enforced": True,
        "coverage_complete": complete,
        "benchmark_interface_complete": not adapter_required,
        "current_execution_scope_complete": not deferred,
        "census_file": str(census_path),
        "configured_node_ids": sorted(configured),
        "benchmark_required_node_ids": sorted(required),
        "benchmark_deferred_node_ids": sorted(deferred),
        "adapter_required_node_ids": sorted(adapter_required),
        "unsupported_node_ids": sorted(unsupported),
        "missing_required_node_ids": sorted(missing),
        "unexpected_node_ids": sorted(unexpected),
        "non_rollout_configured_node_ids": sorted(non_rollout_configured),
        "fleet_device_count": len(all_ids),
        "benchmark_required_count": len(required),
        "benchmark_deferred_count": len(deferred),
        "adapter_required_count": len(adapter_required),
        "unsupported_count": len(unsupported),
        "configured_benchmark_count": len(configured & required),
        "accounted_device_count": len(accounted),
        "ready": ready,
        "admission": {"admitted": False},
    }
    blockers = []
    if deferred:
        blockers.append(
            "benchmark deferred until node returns online: "
            + ", ".join(sorted(deferred))
        )
    if adapter_required:
        blockers.append(
            "benchmark adapter required for: "
            + ", ".join(sorted(adapter_required))
        )
    if blockers:
        report["qualification_blockers"] = blockers
    if not ready:
        reasons = []
        if missing and mode == "full":
            reasons.append(
                "missing required benchmark nodes: " + ", ".join(sorted(missing))
            )
        if unexpected:
            reasons.append(
                "rollout nodes absent from census: "
                + ", ".join(sorted(unexpected))
            )
        if non_rollout_configured:
            reasons.append(
                "non-runnable census devices were included as SSH rollout nodes: "
                + ", ".join(sorted(non_rollout_configured))
            )
        report["errors"] = reasons
    return report


def enforce_rollout_coverage(
    config: Mapping[str, Any], config_path: str
) -> Dict[str, Any]:
    report = validate_rollout_coverage(config, config_path)
    if not report["ready"]:
        raise ValueError("; ".join(report.get("errors") or ["fleet coverage failed"]))
    return report
