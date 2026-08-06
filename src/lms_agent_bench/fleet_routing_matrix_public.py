from __future__ import annotations

import datetime as _datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# datetime.UTC was added in Python 3.11. The project supports Python 3.10, so
# provide the equivalent module attribute only while loading the routing module.
if not hasattr(_datetime, "UTC"):
    setattr(_datetime, "UTC", _datetime.timezone.utc)

from . import fleet_routing_matrix as _implementation  # noqa: E402
from .fleet_routing_serialization_policy import _json_safe  # noqa: E402

_raw_builder = _implementation.build_routing_matrix


def _expanded_role_policy(
    tailnet_status: Mapping[str, Any],
    role_policy: Any | None,
) -> Any | None:
    if not isinstance(role_policy, Mapping):
        return role_policy
    default = role_policy.get("default_policy")
    configured = role_policy.get("nodes")
    if not isinstance(default, Mapping) or not isinstance(configured, Mapping):
        return role_policy
    expanded: dict[str, dict[str, Any]] = {}
    for node in _implementation._tailnet_rows(tailnet_status):
        node_id = str(node["node_id"])
        specific = configured.get(node_id)
        expanded[node_id] = {
            **dict(default),
            **(dict(specific) if isinstance(specific, Mapping) else {}),
        }
    return {**dict(role_policy), "nodes": expanded}


def _qualified_only(document: dict[str, Any]) -> dict[str, Any]:
    profiles = [
        profile
        for profile in document.get("profiles") or []
        if profile.get("qualified") is True
    ]
    eligible = {
        (
            str(profile.get("task_family") or ""),
            str(profile.get("node_id") or ""),
            str(profile.get("model_id") or ""),
            str(profile.get("loadout_fingerprint") or ""),
        )
        for profile in profiles
    }
    rankings: dict[str, list[dict[str, Any]]] = {}
    for family, rows in (document.get("rankings") or {}).items():
        rankings[str(family)] = [
            row
            for row in rows
            if (
                str(family),
                str(row.get("node_id") or ""),
                str(row.get("model_id") or ""),
                str(row.get("loadout_fingerprint") or ""),
            )
            in eligible
        ]
    document["profiles"] = profiles
    document["rankings"] = rankings
    summary = document.setdefault("summary", {})
    summary["routing_profiles"] = len(profiles)
    summary["qualified_benchmark_rows"] = len(
        {
            (
                str(profile.get("node_id") or ""),
                str(profile.get("model_id") or ""),
                str(profile.get("loadout_fingerprint") or ""),
            )
            for profile in profiles
        }
    )
    return document


def build_routing_matrix(
    tailnet_status: Mapping[str, Any],
    *,
    role_policy: Any | None = None,
    benchmark_documents: Iterable[Any] = (),
) -> dict[str, Any]:
    document = _raw_builder(
        tailnet_status,
        role_policy=_expanded_role_policy(tailnet_status, role_policy),
        benchmark_documents=benchmark_documents,
    )
    return _json_safe(_qualified_only(document))


def main(argv: Sequence[str] | None = None) -> int:
    original = _implementation.build_routing_matrix
    _implementation.build_routing_matrix = build_routing_matrix
    try:
        return _implementation.main(argv)
    finally:
        _implementation.build_routing_matrix = original


__all__ = ["build_routing_matrix", "main"]
