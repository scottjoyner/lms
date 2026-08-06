#!/usr/bin/env python3
"""Build a non-admitting fleet capability and benchmark routing matrix.

The matrix deliberately separates three questions:

1. Is a machine visible on the tailnet?
2. Which bounded worker roles may it perform?
3. Which loaded model/loadout is best for a specific task family?

Tailnet discovery never admits a runtime, loads a model, or grants an agent shell.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests

SCHEMA_VERSION = "fleet_routing_matrix.v1"

TASK_POLICIES: dict[str, dict[str, Any]] = {
    "coding": {
        "quality_floor": 0.68,
        "weights": {"quality": 0.58, "speed": 0.17, "reliability": 0.20, "confidence": 0.05},
        "required_roles": {"full_agent", "code_agent"},
    },
    "reasoning": {
        "quality_floor": 0.70,
        "weights": {"quality": 0.62, "speed": 0.13, "reliability": 0.20, "confidence": 0.05},
        "required_roles": {"full_agent", "reasoning"},
    },
    "tool_use": {
        "quality_floor": 0.70,
        "weights": {"quality": 0.58, "speed": 0.12, "reliability": 0.25, "confidence": 0.05},
        "required_roles": {"full_agent", "tool_agent"},
    },
    "long_context": {
        "quality_floor": 0.65,
        "weights": {"quality": 0.55, "speed": 0.10, "reliability": 0.20, "confidence": 0.15},
        "required_roles": {"full_agent", "long_context"},
    },
    "summarization": {
        "quality_floor": 0.52,
        "weights": {"quality": 0.32, "speed": 0.38, "reliability": 0.25, "confidence": 0.05},
        "required_roles": {"full_agent", "summarization", "auxiliary_llm"},
    },
    "compression": {
        "quality_floor": 0.48,
        "weights": {"quality": 0.27, "speed": 0.43, "reliability": 0.25, "confidence": 0.05},
        "required_roles": {"full_agent", "compression", "auxiliary_llm"},
    },
    "extraction": {
        "quality_floor": 0.58,
        "weights": {"quality": 0.42, "speed": 0.28, "reliability": 0.25, "confidence": 0.05},
        "required_roles": {"full_agent", "extraction", "auxiliary_llm"},
    },
}

ROLE_CAPABILITIES: dict[str, set[str]] = {
    "observer": {"tailscale_reachable", "inventory"},
    "benchmark_only": {"tailscale_reachable", "inventory", "benchmark"},
    "auxiliary_llm": {"llm", "summarization", "compression", "extraction"},
    "summarization": {"llm", "summarization"},
    "compression": {"llm", "compression"},
    "extraction": {"llm", "extraction"},
    "reasoning": {"llm", "reasoning"},
    "long_context": {"llm", "long_context"},
    "tool_agent": {"llm", "tool_use"},
    "code_agent": {"llm", "coding", "code_execution"},
    "full_agent": {
        "llm",
        "coding",
        "reasoning",
        "tool_use",
        "long_context",
        "summarization",
        "compression",
        "extraction",
        "code_execution",
        "agent_runtime",
    },
}


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    return text.lower() or "unknown-node"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tailnet_status(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        value = _load_json(path)
    else:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            timeout=30,
        )
        value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("tailscale status must be a JSON object")
    return value


def _tailnet_rows(status: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    self_row = status.get("Self")
    if isinstance(self_row, Mapping):
        candidates.append(("self", self_row))
    peers = status.get("Peer")
    if isinstance(peers, Mapping):
        for peer_key, peer in peers.items():
            if isinstance(peer, Mapping):
                candidates.append((str(peer_key), peer))
    elif isinstance(peers, list):
        for index, peer in enumerate(peers):
            if isinstance(peer, Mapping):
                candidates.append((f"peer-{index}", peer))

    rows: dict[str, dict[str, Any]] = {}
    for peer_key, peer in candidates:
        dns_name = str(peer.get("DNSName") or peer.get("dns_name") or "").rstrip(".")
        host_name = str(peer.get("HostName") or peer.get("hostname") or "").strip()
        node_id = _slug(host_name or (dns_name.split(".", 1)[0] if dns_name else peer_key))
        ips = peer.get("TailscaleIPs") or peer.get("tailscale_ips") or []
        if isinstance(ips, str):
            ips = [ips]
        tags = peer.get("Tags") or peer.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        row = {
            "node_id": node_id,
            "display_name": host_name or dns_name or node_id,
            "dns_name": dns_name or None,
            "tailscale_ips": sorted({str(item) for item in ips if str(item).strip()}),
            "os_family": str(peer.get("OS") or peer.get("os") or "unknown").lower(),
            "online": bool(peer.get("Online", peer.get("online", False))),
            "active": bool(peer.get("Active", peer.get("active", False))),
            "last_seen": peer.get("LastSeen") or peer.get("last_seen"),
            "tags": sorted({str(item) for item in tags if str(item).strip()}),
            "tailnet_discovered": True,
            "discovery_source": "tailscale-status-json",
        }
        existing = rows.get(node_id)
        if existing is None or (row["online"] and not existing["online"]):
            rows[node_id] = row
    return sorted(rows.values(), key=lambda item: item["node_id"])


def _policy_nodes(document: Any) -> dict[str, dict[str, Any]]:
    if document is None:
        return {}
    if not isinstance(document, Mapping):
        raise ValueError("fleet role policy must be an object")
    raw = document.get("nodes", document)
    if isinstance(raw, list):
        result = {}
        for item in raw:
            if isinstance(item, Mapping) and item.get("node_id"):
                result[_slug(item["node_id"])] = dict(item)
        return result
    if isinstance(raw, Mapping):
        return {_slug(key): dict(value) for key, value in raw.items() if isinstance(value, Mapping)}
    raise ValueError("fleet role policy nodes must be an object or array")


def apply_role_policy(
    nodes: Sequence[dict[str, Any]],
    policy_document: Any | None,
) -> list[dict[str, Any]]:
    policies = _policy_nodes(policy_document)
    result: list[dict[str, Any]] = []
    for raw in nodes:
        node = dict(raw)
        policy = policies.get(str(node["node_id"]), {})
        roles = policy.get("roles") or ["observer"]
        if isinstance(roles, str):
            roles = [roles]
        normalized_roles = sorted({_slug(item).replace("-", "_") for item in roles})
        capabilities = {"tailscale_reachable", "inventory"}
        for role in normalized_roles:
            capabilities.update(ROLE_CAPABILITIES.get(role, set()))
        configured = policy.get("capabilities") or []
        if isinstance(configured, str):
            configured = [configured]
        capabilities.update(str(item).strip().lower() for item in configured if str(item).strip())
        worker_mode = str(policy.get("worker_mode") or "observer_only").strip().lower()
        allow_agent_runtime = bool(policy.get("allow_agent_runtime", "full_agent" in normalized_roles))
        allow_code_execution = bool(policy.get("allow_code_execution", "code_agent" in normalized_roles or "full_agent" in normalized_roles))
        node.update(
            {
                "roles": normalized_roles,
                "capabilities": sorted(capabilities),
                "worker_mode": worker_mode,
                "allow_agent_runtime": allow_agent_runtime,
                "allow_code_execution": allow_code_execution,
                "benchmark_policy": str(policy.get("benchmark_policy") or "benchmark_required"),
                "max_concurrent": max(0, int(policy.get("max_concurrent") or 0)),
                "energy_class": str(policy.get("energy_class") or "unknown"),
                "notes": str(policy.get("notes") or ""),
            }
        )
        result.append(node)
    return result


def _normalized_benchmark_rows(documents: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        schema = str(document.get("schema_version") or "")
        source_rows = document.get("rows") if schema == "model_loadout_comparison.v1" else document.get("entries")
        if not isinstance(source_rows, list):
            continue
        for raw in source_rows:
            if not isinstance(raw, Mapping):
                continue
            node_id = _slug(raw.get("node_id"))
            model_id = str(raw.get("model_id") or raw.get("provider_model") or "").strip()
            if not model_id or node_id == "unknown-node":
                continue
            quality = raw.get("overall_task_pass_rate", raw.get("quality_score"))
            confidence = raw.get("quality_confidence", raw.get("confidence", 0.0))
            reliability = raw.get("success_rate", raw.get("effect_checkpoint_rate", quality))
            tps = raw.get("completion_tokens_per_second_end_to_end", raw.get("tokens_per_second"))
            families = raw.get("task_families") or []
            if isinstance(families, str):
                families = [families]
            rows.append(
                {
                    "node_id": node_id,
                    "model_id": model_id,
                    "loadout_fingerprint": raw.get("loadout_fingerprint"),
                    "qualified": bool(raw.get("qualified", raw.get("loaded", False))),
                    "quality_score": _bounded(quality, 0.5),
                    "quality_confidence": _bounded(confidence, 0.0),
                    "reliability": _bounded(reliability, 0.5),
                    "tokens_per_second": _positive_float(tps),
                    "task_families": sorted({str(item).strip().lower() for item in families if str(item).strip()}),
                    "source_schema": schema or "unknown",
                }
            )
    return rows


def _bounded(value: Any, default: float) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 6)
    except (TypeError, ValueError):
        return default


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _speed_score(tps: float | None) -> float:
    if tps is None:
        return 0.15
    return max(0.0, min(1.0, tps / (tps + 20.0)))


def _eligible_for_family(node: Mapping[str, Any], family: str) -> bool:
    if not node.get("online"):
        return False
    if str(node.get("worker_mode") or "observer_only") == "observer_only":
        return False
    roles = {str(item) for item in node.get("roles") or []}
    return bool(roles & set(TASK_POLICIES[family]["required_roles"]))


def build_routing_matrix(
    tailnet_status: Mapping[str, Any],
    *,
    role_policy: Any | None = None,
    benchmark_documents: Iterable[Any] = (),
) -> dict[str, Any]:
    nodes = apply_role_policy(_tailnet_rows(tailnet_status), role_policy)
    node_index = {str(node["node_id"]): node for node in nodes}
    benchmark_rows = _normalized_benchmark_rows(benchmark_documents)
    profiles: list[dict[str, Any]] = []
    rankings: dict[str, list[dict[str, Any]]] = {}

    for family, policy in TASK_POLICIES.items():
        candidates: list[dict[str, Any]] = []
        for row in benchmark_rows:
            node = node_index.get(str(row["node_id"]))
            if node is None or not _eligible_for_family(node, family):
                continue
            family_evidence = set(row.get("task_families") or [])
            quality = float(row["quality_score"])
            confidence = float(row["quality_confidence"])
            quality_floor = float(policy["quality_floor"])
            below_floor = confidence >= 0.5 and quality < quality_floor
            weights = policy["weights"]
            speed = _speed_score(row.get("tokens_per_second"))
            utility = (
                quality * float(weights["quality"])
                + speed * float(weights["speed"])
                + float(row["reliability"]) * float(weights["reliability"])
                + confidence * float(weights["confidence"])
            )
            if family_evidence and family not in family_evidence:
                utility *= 0.85
            profile = {
                **row,
                "task_family": family,
                "quality_floor": quality_floor,
                "quality_floor_passed": not below_floor,
                "speed_score": round(speed, 6),
                "utility_score": round(utility, 6),
                "worker_mode": node["worker_mode"],
                "roles": node["roles"],
            }
            profiles.append(profile)
            if not below_floor:
                candidates.append(profile)
        candidates.sort(
            key=lambda item: (
                -float(item["utility_score"]),
                -float(item["quality_score"]),
                -(float(item["tokens_per_second"]) if item.get("tokens_per_second") is not None else -1.0),
                str(item["node_id"]),
                str(item["model_id"]),
            )
        )
        rankings[family] = [
            {
                "rank": index,
                "node_id": item["node_id"],
                "model_id": item["model_id"],
                "loadout_fingerprint": item.get("loadout_fingerprint"),
                "utility_score": item["utility_score"],
                "quality_score": item["quality_score"],
                "tokens_per_second": item.get("tokens_per_second"),
                "reliability": item["reliability"],
                "confidence": item["quality_confidence"],
            }
            for index, item in enumerate(candidates, start=1)
        ]

    generated = datetime.now(UTC).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "fleet_capability_and_routing_matrix",
        "generated_at_utc": generated,
        "source": {
            "tailnet": "tailscale status --json",
            "benchmarks": sorted({row["source_schema"] for row in benchmark_rows}),
        },
        "policy": {
            "discovery_is_not_admission": True,
            "observer_nodes_are_visible_but_not_routable": True,
            "quality_floor_precedes_speed_ranking": True,
            "task_policies": TASK_POLICIES,
        },
        "summary": {
            "tailnet_nodes": len(nodes),
            "online_nodes": sum(bool(node["online"]) for node in nodes),
            "observer_only_nodes": sum(node["worker_mode"] == "observer_only" for node in nodes),
            "agent_runtime_nodes": sum(bool(node["allow_agent_runtime"]) for node in nodes),
            "benchmark_rows": len(benchmark_rows),
            "routing_profiles": len(profiles),
        },
        "nodes": nodes,
        "profiles": profiles,
        "rankings": rankings,
        "admission": {"admitted": False},
    }


def publish_matrix(
    document: Mapping[str, Any],
    *,
    base_url: str,
    user_env: str,
    password_env: str,
) -> dict[str, Any]:
    user = os.getenv(user_env, "").strip()
    password = os.getenv(password_env, "")
    if not user or not password:
        raise ValueError("AssistX Basic Auth environment variables are required")
    response = requests.post(
        f"{base_url.rstrip('/')}/api/fleet/routing-matrix/import",
        json=dict(document),
        auth=(user, password),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("AssistX routing-matrix response must be an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a tailnet-wide capability and benchmark routing matrix")
    parser.add_argument("--tailscale-json", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--comparison", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assistx-url")
    parser.add_argument("--assistx-user-env", default="BASIC_AUTH_USER")
    parser.add_argument("--assistx-password-env", default="BASIC_AUTH_PASS")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status = load_tailnet_status(args.tailscale_json)
        policy = _load_json(args.policy) if args.policy else None
        benchmarks = [_load_json(path) for path in args.comparison]
        document = build_routing_matrix(status, role_policy=policy, benchmark_documents=benchmarks)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.assistx_url:
            publish_matrix(
                document,
                base_url=args.assistx_url,
                user_env=args.assistx_user_env,
                password_env=args.assistx_password_env,
            )
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError, requests.RequestException) as exc:
        print(f"fleet routing matrix failed: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
