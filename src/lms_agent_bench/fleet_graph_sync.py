#!/usr/bin/env python3
"""fleet_graph_sync.py — publish live LM Studio fleet health into Neo4j.

This is the maintained replacement for the broken fleet-task-dispatcher.service.
It is a *writer only*: it observes the fleet (via the auto-router HTTP API and the
local fleet_state.json produced by fleet.py) and mirrors that into the knowledge
graph using the existing FleetSnapshot / FleetNodeState / FleetModelState /
FleetLoadout node schema already present in Neo4j.

Why this matters (per the fleet reliability work):
  * Other agents can query fleet status PROGRAMMATICALLY from the graph instead
    of scraping router health or stale JSON.
  * Health, loaded models, and the current loadout plan are all versioned as a
    snapshot chain (previous_snapshot_id), so any agent can answer "what was the
    fleet doing an hour ago" as well as "what is it doing now".

Schema (must match what already exists in the graph):
  (FleetSnapshot {snapshot_id, captured_at, ...}) -[:HAS_NODE_STATE]-> (FleetNodeState)
  (FleetSnapshot) -[:HAS_MODEL_STATE]-> (FleetModelState)
  (FleetSnapshot) -[:HAS_LOADOUT]-> (FleetLoadout)
  (FleetTaskProfile) standalone, keyed by task_profile_id

Run:
  python3 fleet_graph_sync.py                      # one-shot publish
  python3 fleet_graph_sync.py --watch --sleep 60   # loop
  python3 fleet_graph_sync.py --router-url http://localhost:8088

Env: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DB are read from the
environment (no hardcoded secret). The driver is provided by
:mod:`lms_agent_bench.neo4j`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from lms_agent_bench.neo4j import NEO4J_DB, get_driver

HERE = Path(__file__).resolve().parents[2]
STATE_JSON = HERE / "fleet_state.json"
ROUTES_JSON = HERE / "routing_rules.json"

DEFAULT_ROUTER = os.environ.get("FLEET_ROUTER_URL", "http://localhost:8088")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(ts: Optional[dt.datetime] = None) -> str:
    return (ts or utc_now()).isoformat()


def _http_get_json(url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] GET {url} failed: {exc}", file=sys.stderr)
        return None


def _router_nodes(router_url: str) -> List[Dict[str, Any]]:
    data = _http_get_json(f"{router_url.rstrip('/')}/api/fleet/nodes", timeout=12)
    if not isinstance(data, dict):
        return []
    nodes = data.get("nodes") or []
    return [n for n in nodes if isinstance(n, dict)]


def _router_health(router_url: str) -> Dict[str, Any]:
    data = _http_get_json(f"{router_url.rstrip('/')}/health", timeout=12)
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------
# Snapshot assembly
# --------------------------------------------------------------------------
def build_snapshot(router_url: str) -> Dict[str, Any]:
    """Assemble a fleet snapshot from the live router + local fleet_state.json.

    Prefers the router's reporter-fed live view (real loaded models, real specs)
    and enriches with the benchmarked capacity from fleet_state.json when present.
    """
    captured_at = utc_now()
    snapshot_id = str(uuid.uuid4())

    nodes_live = _router_nodes(router_url)
    health = _router_health(router_url)

    # benchmarked capacity index: model_key -> {best_node, tps, ...}
    capacity: Dict[str, Any] = {}
    state: Dict[str, Any] = {}
    if STATE_JSON.exists():
        try:
            state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
            capacity = state.get("model_best_node", {})
        except Exception:
            state = {}

    # Build node states from the live router report.
    node_states: List[Dict[str, Any]] = []
    model_states: List[Dict[str, Any]] = []
    online = 0
    loaded_total = 0
    for n in nodes_live:
        name = str(n.get("host_name") or n.get("hostname") or n.get("name") or "unknown")
        loaded = n.get("loaded") or []
        specs = n.get("specs") or {}
        ok = bool(n.get("health", {}).get("ok", True)) if isinstance(n.get("health"), dict) else True
        ip = n.get("ip") or ""
        if ok and loaded:
            online += 1
        loaded_total += len(loaded)
        node_states.append({
            "node_name": name,
            "ip": ip,
            "online": ok,
            "error": "" if ok else (str(n.get("error") or "unreachable")),
            "latency_ms": float(n.get("latency_ms") or 0.0),
            "loaded_models": list(loaded),
            "all_models": list(loaded),  # live reporter only sees loaded; catalog not needed for health
            "power_watts": float(specs.get("power_watts") or 0.0),
            "ram_gib": float(specs.get("system_ram_gib") or 0.0) or None,
            "vram_gib": float(specs.get("vram_gib") or 0.0) or None,
            "cpu": str(specs.get("cpu_model") or specs.get("cpu") or ""),
        })
        for mk in loaded:
            model_states.append({
                "node_name": name,
                "model_id": str(mk),
                "online": ok,
                "loaded": True,
                "latency_ms": float(n.get("latency_ms") or 0.0),
            })

    # Loadout plan: derive a simple primary/fallback per task family from capacity.
    loadouts = _derive_loadouts(state, capacity)

    summary = {
        "online_nodes": online,
        "loaded_models_total": loaded_total,
        "node_count": len(node_states),
        "router_health_ok": bool(health.get("ok")),
        "router_status": str(health.get("status", "unknown")),
        "open_circuits": int(health.get("open_circuits", 0)),
        "outbox_level": str((health.get("assistx_outbox_pressure") or {}).get("level", "unknown")),
    }

    return {
        "snapshot_id": snapshot_id,
        "captured_at": _iso(captured_at),
        "captured_at_ms": int(captured_at.timestamp() * 1000),
        "source": "fleet_graph_sync.py",
        "node_count": len(node_states),
        "model_count": len(model_states),
        "loadout_count": len(loadouts),
        "task_profile_count": len(loadouts),
        "summary_json": json.dumps(summary),
        "raw_json": json.dumps({
            "health": health,
            "nodes": node_states,
            "models": model_states,
            "loadouts": loadouts,
        }),
        "node_states": node_states,
        "model_states": model_states,
        "loadouts": loadouts,
    }


def _derive_loadouts(state: Dict[str, Any], capacity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn the benchmarked model_best_node map into FleetLoadout rows.

    One loadout per (task_family-ish) model: the best node is primary, the
    next-best live node is fallback. This gives other agents a queryable
    primary/fallback routing plan straight from the graph.
    """
    loadouts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for mk, info in capacity.items():
        if mk in seen:
            continue
        seen.add(mk)
        best = info.get("best_node")
        if not best:
            continue
        fallbacks = [x for x in info.get("available_on", []) if x != best]
        fallback = fallbacks[0] if fallbacks else None
        loadouts.append({
            "task_profile_id": f"model:{mk}",
            "task_profile_name": mk,
            "primary_node": best,
            "primary_model": mk,
            "fallback_node": fallback,
            "fallback_model": mk if fallback else None,
            "reviewer_node": best,
            "reviewer_model": mk,
            "score": round(float(info.get("best_tps", 0.0)), 3),
            "rationale": f"highest tps among available nodes; fallback={fallback or 'none'}",
        })
    return loadouts


# --------------------------------------------------------------------------
# Neo4j write
# --------------------------------------------------------------------------
def ensure_schema(driver, db: str) -> None:
    stmts = [
        "CREATE CONSTRAINT fleet_snapshot_id IF NOT EXISTS FOR (n:FleetSnapshot) REQUIRE n.snapshot_id IS UNIQUE",
        "CREATE CONSTRAINT fleet_node_state IF NOT EXISTS FOR (n:FleetNodeState) REQUIRE (n.snapshot_id, n.node_name) IS UNIQUE",
        "CREATE CONSTRAINT fleet_model_state IF NOT EXISTS FOR (n:FleetModelState) REQUIRE (n.snapshot_id, n.node_name, n.model_id) IS UNIQUE",
        "CREATE CONSTRAINT fleet_loadout_id IF NOT EXISTS FOR (n:FleetLoadout) REQUIRE n.loadout_id IS UNIQUE",
        "CREATE CONSTRAINT fleet_task_profile_id IF NOT EXISTS FOR (n:FleetTaskProfile) REQUIRE n.task_profile_id IS UNIQUE",
    ]
    with driver.session(database=db) as s:
        for stmt in stmts:
            try:
                s.run(stmt)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] schema stmt failed: {exc}", file=sys.stderr)


def publish_snapshot(driver, db: str, snap: Dict[str, Any], previous_snapshot_id: Optional[str]) -> None:
    snap_props = {k: v for k, v in snap.items() if k not in ("node_states", "model_states", "loadouts")}
    if previous_snapshot_id:
        snap_props["previous_snapshot_id"] = previous_snapshot_id

    with driver.session(database=db) as s:
        # Snapshot node + chain link to previous.
        s.run(
            """
            MERGE (snap:FleetSnapshot {snapshot_id: $snapshot_id})
            SET snap += $props
            WITH snap
            OPTIONAL MATCH (prev:FleetSnapshot {snapshot_id: $previous_id})
            WHERE prev IS NOT NULL
            MERGE (prev)-[:NEXT_SNAPSHOT]->(snap)
            """,
            snapshot_id=snap["snapshot_id"],
            props=snap_props,
            previous_id=previous_snapshot_id or "",
        )

        # Node states.
        for ns in snap["node_states"]:
            s.run(
                """
                MERGE (n:FleetNodeState {snapshot_id: $snapshot_id, node_name: $node_name})
                SET n += $props
                WITH n
                MATCH (snap:FleetSnapshot {snapshot_id: $snapshot_id})
                MERGE (snap)-[:HAS_NODE_STATE]->(n)
                """,
                snapshot_id=snap["snapshot_id"],
                node_name=ns["node_name"],
                props=ns,
            )

        # Model states.
        for ms in snap["model_states"]:
            s.run(
                """
                MERGE (m:FleetModelState {snapshot_id: $snapshot_id, node_name: $node_name, model_id: $model_id})
                SET m += $props
                WITH m
                MATCH (snap:FleetSnapshot {snapshot_id: $snapshot_id})
                MERGE (snap)-[:HAS_MODEL_STATE]->(m)
                """,
                snapshot_id=snap["snapshot_id"],
                node_name=ms["node_name"],
                model_id=ms["model_id"],
                props=ms,
            )

        # Loadouts.
        for lo in snap["loadouts"]:
            loadout_id = f"{snap['snapshot_id']}:{lo['task_profile_id']}"
            lo_full = dict(lo)
            lo_full["loadout_id"] = loadout_id
            s.run(
                """
                MERGE (l:FleetLoadout {loadout_id: $loadout_id})
                SET l += $props
                WITH l
                MATCH (snap:FleetSnapshot {snapshot_id: $snapshot_id})
                MERGE (snap)-[:HAS_LOADOUT]->(l)
                """,
                loadout_id=loadout_id,
                snapshot_id=snap["snapshot_id"],
                props=lo_full,
            )


def latest_snapshot_id(driver, db: str) -> Optional[str]:
    with driver.session(database=db) as s:
        rec = s.run(
            "MATCH (n:FleetSnapshot) RETURN n.snapshot_id AS id ORDER BY n.captured_at DESC LIMIT 1"
        ).single()
        return rec["id"] if rec else None


def cmd_publish(args: argparse.Namespace) -> int:

    snap = build_snapshot(args.router_url)
    if not snap["node_states"]:
        print("[error] no live fleet nodes retrieved from router; aborting.", file=sys.stderr)
        return 2

    driver = get_driver()
    try:
        ensure_schema(driver, NEO4J_DB)
        previous = latest_snapshot_id(driver, NEO4J_DB)
        publish_snapshot(driver, NEO4J_DB, snap, previous)
    finally:
        driver.close()

    print(
        f"Published FleetSnapshot {snap['snapshot_id'][:8]} | "
        f"nodes={snap['node_count']} models={snap['model_count']} loadouts={snap['loadout_count']} "
        f"(prev={ (previous[:8] if previous else 'none')})"
    )
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    while True:
        rc = cmd_publish(args)
        if rc != 0:
            print("[watch] publish failed; retrying", file=sys.stderr)
        print(f"[watch] next publish in {args.sleep}s", flush=True)
        time.sleep(args.sleep)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Publish LM Studio fleet health into Neo4j.")
    ap.add_argument("--router-url", default=DEFAULT_ROUTER)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish", help="publish one snapshot")
    p.add_argument("--router-url", default=DEFAULT_ROUTER, help="router URL (default env FLEET_ROUTER_URL)")
    p.set_defaults(func=cmd_publish)
    w = sub.add_parser("watch", help="publish on an interval")
    w.add_argument("--router-url", default=DEFAULT_ROUTER, help="router URL (default env FLEET_ROUTER_URL)")
    w.add_argument("--sleep", type=int, default=60)
    w.set_defaults(func=cmd_watch)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
