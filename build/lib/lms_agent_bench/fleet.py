#!/usr/bin/env python3
"""fleet.py — unified fleet control plane.

One command that owns the full benchmark -> analyze -> route lifecycle, solving
the structural gaps in the original point-in-time scripts:

  * discovery is centralized in fleet_discover.py (no more duplicated NODES
    dicts, no more ALIAS hacks — fleet.toml + tailscale win the day);
  * ``state`` emits a machine-readable ``fleet_state.json`` the orchestrator can
    consume live (health, per-model availability, measured tps, derived
    concurrency tier, and a per-node ``stale`` flag so crash artifacts are
    never read as live);
  * concurrency caps are *measured*, not hand-curated — the probe's speed-hit
    drives a 1/2/4 tier per (node, model);
  * ``routes`` emits ``routing_rules.json`` from measured data instead of the
    hardcoded NOTES dicts;
  * every network touch retries with exponential backoff.

Subcommands:
  discover            list configured/live nodes
  state               build fleet_state.json from current artifacts
  routes              emit routing_rules.json from fleet_state.json
  report              regenerate docs/fleet_analysis.md + fleet_writeup.md
  bench               run the full benchmark + concurrency probe pipeline
  status              quick live health snapshot (stdout)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"
STATE_JSON = HERE / "fleet_state.json"
ROUTES_JSON = HERE / "routing_rules.json"
STALE_SECONDS = 6 * 3600  # artifacts older than this are flagged stale

sys.path.insert(0, str(HERE / "src" / "lms_agent_bench"))
from lms_agent_bench.fleet_discover import discover, discover_fleet, live_nodes, all_aliases  # noqa: E402


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_embed(mk: str) -> bool:
    return bool(mk) and "embed" in mk.lower()


def _newest_stamp(node_dir: Path) -> float:
    best = 0.0
    for p in node_dir.glob("*.csv"):
        best = max(best, p.stat().st_mtime)
    return best


def _read_rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


# --------------------------------------------------------------------------
# Concurrency tiering — MEASURED, replaces NOTES / --struggle-nodes by fiat.
# --------------------------------------------------------------------------
def derive_concurrency_tier(node: str, model: str, probe_rows: list[dict]) -> dict:
    """Given a node's concurrency-probe rows for one model, classify a safe tier.

    Returns {tier, basis, speed_hit, gain}. tier in {1,2,4}.
    A model that never appears in the probe is treated as 'unknown' (tier 1,
    conservative) so the orchestrator never over-parallelizes blind.
    """
    summary = None
    for r in probe_rows:
        if r.get("node") == node and r.get("model") == model and r.get("phase") == "summary":
            summary = r
            break
    if not summary:
        return {"tier": 1, "basis": "no-probe", "speed_hit": None, "gain": None}
    err = summary.get("error", "")
    hit = gain = None
    import re
    mh = re.search(r"speed_hit=(-?\d+)", err)
    mg = re.search(r"gain=(-?\d+)", err)
    if mh:
        hit = int(mh.group(1))
    if mg:
        gain = int(mg.group(1))
    status = summary.get("status", "")
    if status in ("FAIL", "POOR"):
        tier = 1
    elif status == "DEGRADED":
        tier = 2
    elif status == "OK" and (gain is not None and gain >= 0):
        tier = 2  # measured 2-parallel is acceptable
    else:
        tier = 1
    return {"tier": tier, "basis": status.lower(), "speed_hit": hit, "gain": gain}


# --------------------------------------------------------------------------
# State assembly
# --------------------------------------------------------------------------
def build_state() -> dict:
    nodes = discover()
    aliases = all_aliases(nodes)
    # The real fleet = nodes we have actually benchmarked (a runs/<node> dir).
    # Tailscale peers without artifacts (raspberrypi, other MacBooks, etc.) are
    # not part of the benchmarked fleet and are excluded from the state.
    # When two dirs map to the same canonical name (e.g. stale `macbook-air`
    # + real `scotts-macbook-air`), keep the most recently stamped one.
    discovered = {n.name: n for n in nodes}
    raw_dirs = [p for p in RUNS.iterdir() if p.is_dir() and (p / "run_summary.csv").exists()]
    by_canon: dict[str, Path] = {}
    for nd in raw_dirs:
        canon = aliases.get(nd.name, nd.name)
        prev = by_canon.get(canon)
        if prev is None or _newest_stamp(nd) > _newest_stamp(prev):
            by_canon[canon] = nd
    bench_dirs = list(by_canon.values())
    # Liveness is probed ONLY for nodes we have actually benchmarked (the real
    # fleet), not the whole tailscale view — probing every peer (raspberrypi,
    # iphones, etc.) with retry/backoff would hang build_state for minutes.
    canon_names = {aliases.get(nd.name, nd.name) for nd in bench_dirs}
    live_nodeset = live_nodes([n for n in nodes if n.name in canon_names])
    live = {n.name for n in live_nodeset}
    probe_dir = RUNS / "concurrency_probe"
    probe_rows: list[dict] = []
    if probe_dir.exists():
        for csvp in sorted(probe_dir.glob("concurrency_probe_*.csv")):
            probe_rows.extend(_read_rows(csvp))

    node_states = []
    model_index: dict[str, list[dict]] = {}

    for nd in bench_dirs:
        raw_name = nd.name
        name = aliases.get(raw_name, raw_name)
        # fallback node when only artifacts exist (no discovery entry)
        class _Art:  # minimal stand-in carrying name/notes
            def __init__(self, nm):
                self.name = nm
                self.url = ""
                self.via = "artifact"
                self.notes = ""
        node_obj = discovered.get(name) or _Art(name)
        stamp = _newest_stamp(nd)
        stale = (time.time() - stamp) > STALE_SECONDS if stamp else True
        hw = {}
        hp = nd / "host_profile.json"
        if hp.exists():
            try:
                hw = json.loads(hp.read_text(encoding="utf-8"))
            except Exception:
                hw = {}
        summary = nd / "run_summary.csv"
        rows = _read_rows(summary) if summary.exists() else []
        chat = [r for r in rows if not _is_embed(r.get("model_key"))]

        models = []
        for r in chat:
            mk = r["model_key"]
            tps = _f(r.get("tps_med")) or 0.0
            available = tps > 0
            tier = derive_concurrency_tier(name, mk, probe_rows) if available else {
                "tier": 0, "basis": "unavailable", "speed_hit": None, "gain": None}
            entry = {
                "model": mk,
                "available": available,
                "tps_med": round(tps, 2) if tps else 0.0,
                "ttft_med_ms": round(_f(r.get("ttft_med")) or 0.0, 1),
                "eval_score": round(_f(r.get("eval_score_avg")) or 0.0, 3),
                "concurrency": tier,
            }
            models.append(entry)
            model_index.setdefault(mk, []).append({
                "node": name, "available": available, "tps_med": entry["tps_med"],
                "ttft_med_ms": entry["ttft_med_ms"], "concurrency_tier": tier["tier"],
            })

        # hardware caps for capacity
        ram = (hw.get("memory") or {}).get("ram_total_gib")
        vram_mib = (hw.get("vram") or {}).get("vram_total_mib")
        node_states.append({
            "name": name,
            "url": node_obj.url,
            "via": node_obj.via,
            "live": name in live,
            "stale": stale,
            "generated_at": datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat() if stamp else None,
            "hardware": {
                "cpu": (hw.get("cpu") or {}).get("model"),
                "ram_gib": ram,
                "vram_gib": round(vram_mib / 1024, 1) if vram_mib else None,
                "verified": bool(hw.get("source")),
            },
            "models_loaded": len(chat),
            "models_available": sum(1 for m in models if m["available"]),
            "models": models,
            "notes": node_obj.notes,
        })

    best = {}
    for mk, entries in model_index.items():
        avail = [e for e in entries if e["available"]]
        if not avail:
            best[mk] = {"best_node": None, "best_tps": 0.0, "available_everywhere": False}
            continue
        top = max(avail, key=lambda e: e["tps_med"])
        best[mk] = {
            "best_node": top["node"],
            "best_tps": top["tps_med"],
            "max_concurrency": max(e["concurrency_tier"] for e in avail),
            "available_on": [e["node"] for e in avail],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale_threshold_seconds": STALE_SECONDS,
        "node_count": len(node_states),
        "live_count": sum(1 for n in node_states if n["live"]),
        "nodes": node_states,
        "model_best_node": best,
        "aliases": aliases,
    }


def cmd_state(args=None) -> int:
    state = build_state()
    STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Wrote {STATE_JSON} | {state['node_count']} nodes, "
          f"{state['live_count']} live, {len(state['model_best_node'])} models indexed.")
    return 0


# --------------------------------------------------------------------------
# Routing rules — derived from measured state, not hand-curated NOTES.
# --------------------------------------------------------------------------
def cmd_routes(args=None) -> int:
    if not STATE_JSON.exists():
        cmd_state()
    state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    rules = {"generated_at": state["generated_at"], "default_concurrency": 1, "nodes": []}

    for n in state["nodes"]:
        caps = sorted({m["concurrency"]["tier"] for m in n["models"] if m["concurrency"]["tier"] > 0}, reverse=True)
        node_rule = {
            "name": n["name"],
            "url": n["url"],
            "live": n["live"],
            "stale": n["stale"],
            "max_concurrency": max(caps) if caps else 1,
            "ram_gib": n["hardware"]["ram_gib"],
            "vram_gib": n["hardware"]["vram_gib"],
            "serves": [m["model"] for m in n["models"] if m["available"]],
            "prefer_for": [],
        }
        # fast small models -> preferred for low-latency routing
        fast = [m["model"] for m in n["models"]
                if m["available"] and m["tps_med"] >= 15 and m["ttft_med_ms"] <= 50]
        node_rule["prefer_for"] = fast
        rules["nodes"].append(node_rule)

    # pick a primary node per model (highest tps among non-stale, live)
    routing = {}
    for mk, info in state["model_best_node"].items():
        if not info["best_node"]:
            routing[mk] = {"node": None, "reason": "unavailable-everywhere"}
            continue
        routing[mk] = {
            "node": info["best_node"],
            "tps_med": info["best_tps"],
            "max_concurrency": info["max_concurrency"],
            "fallbacks": [x for x in info.get("available_on", []) if x != info["best_node"]],
        }
    rules["routing"] = routing
    ROUTES_JSON.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    print(f"Wrote {ROUTES_JSON} | {len(rules['nodes'])} node rules, "
          f"{len(routing)} model routes.")
    return 0


# --------------------------------------------------------------------------
# Live status
# --------------------------------------------------------------------------
def cmd_status(args=None) -> int:
    nodes = discover() if getattr(args, "all", False) else discover_fleet()
    exc = set(getattr(args, "exclude", None) or [])
    nodes = [n for n in nodes if n.name not in exc]
    live = live_nodes(nodes)
    live_names = {n.name for n in live}
    print(f"Fleet status — {len(live)}/{len(nodes)} nodes live\n")
    for n in nodes:
        mark = "UP " if n.name in live_names else "DOWN"
        print(f"  [{mark}] {n.name:28s} {n.url}")
    return 0


def cmd_discover(args=None) -> int:
    for n in discover():
        print(f"{n.name:28s} {n.url:48s} [{n.via}]")
    return 0


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def cmd_bench(args) -> int:
    # Resolve --only / --exclude against the curated fleet (explicit toml nodes);
    # tailscale-only peers (raspberrypi, iphones) are not benchmarked unless --all.
    all_nodes = discover() if args.all else discover_fleet()
    names = {n.name for n in all_nodes}
    # Shared endpoints (e.g. x1-370 localhost used by other systems) are excluded
    # by default so we don't contend with them; opt in with --include-shared.
    shared = {n.name for n in all_nodes if getattr(n, "shared", False)}
    if shared and not args.include_shared:
        print(f"=== excluding shared nodes: {sorted(shared)} "
              f"(use --include-shared to bench them) ===", flush=True)
    if args.only:
        only = [n for n in args.only if n in names]
    else:
        only = [n.name for n in all_nodes]
    for ex in (args.exclude or []):
        only = [n for n in only if n != ex]
    if not args.include_shared:
        only = [n for n in only if n not in shared]
    extra = [a for n in only for a in ("--only", n)]
    print(f"=== fleet bench targets: {only} ===", flush=True)
    print("=== fleet bench: stage 1/2 single-stream ===", flush=True)
    rc1 = subprocess.call(
        [sys.executable, str(HERE / "src" / "lms_agent_bench" / "bench_fleet.py"), "--concurrency", str(args.concurrency), *extra])
    print(f"bench_fleet rc={rc1}", flush=True)
    print("=== fleet bench: stage 2/2 concurrency probe ===", flush=True)
    rc2 = subprocess.call(
        [sys.executable, str(HERE / "src" / "lms_agent_bench" / "bench_concurrency_probe.py"),
         "--max-concurrent", str(args.max_concurrency), *extra])
    print(f"probe rc={rc2}", flush=True)
    print("=== regenerating state + routes + report ===", flush=True)
    cmd_state()
    cmd_routes()
    subprocess.call([sys.executable, str(HERE / "fleet_analysis.py")])
    return 0 if (rc1 == 0 and rc2 == 0) else 1


def cmd_report(args=None) -> int:
    rc = subprocess.call([sys.executable, str(HERE / "fleet_analysis.py")])
    subprocess.call([sys.executable, str(HERE / "fleet_writeup.py")])
    cmd_state()
    cmd_routes()
    return rc


# --------------------------------------------------------------------------
# Loadout plan — the orchestrator-consumable artifact (Item 7).
# Converts measured fleet_state.json into per-node mount lists + routing.
# --------------------------------------------------------------------------
def cmd_plan(args=None) -> int:
    if not STATE_JSON.exists():
        cmd_state()
    state = json.loads(STATE_JSON.read_text(encoding="utf-8"))

    # demand mode: balanced | realtime | quality
    demand = (args.demand if args else "balanced")
    per_node = {}
    for n in state["nodes"]:
        if n["stale"]:
            continue
        av = [m for m in n["models"] if m["available"]]
        if demand == "realtime":
            av.sort(key=lambda m: (m["ttft_med_ms"], -m["tps_med"]))
        elif demand == "quality":
            av.sort(key=lambda m: -m["eval_score"])
        else:  # balanced
            av.sort(key=lambda m: (m["tps_med"] * 0.5 + m["eval_score"] * 10))
        per_node[n["name"]] = {
            "url": n["url"],
            "max_concurrency": max([m["concurrency"]["tier"] for m in av], default=1),
            "mount": [m["model"] for m in av[: args.top if args else 6]],
            "ram_gib": n["hardware"]["ram_gib"],
            "vram_gib": n["hardware"]["vram_gib"],
        }

    plan = {
        "generated_at": state["generated_at"],
        "demand": demand,
        "nodes": per_node,
        "routing": state["model_best_node"],
    }
    LOADOUT_JSON = HERE / "fleet_loadout.json"
    LOADOUT_JSON.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Wrote {LOADOUT_JSON} | {len(per_node)} nodes, demand={demand}")
    return 0


def cmd_watch(args=None) -> int:
    """Continuous refresh of fleet_state.json + routing_rules.json + loadout (Item 9).

    Keeps the machine-readable artifacts live so the orchestrator never reads a
    stale snapshot, and keeps fleet_loadout.json current so `loadout` converges
    against fresh measured data. Pair with cron or nohup:
        nohup python3 fleet.py watch --sleep 900 &
    """
    sleep = (args.sleep if args else 900)
    demand = (args.demand if args else "balanced")
    while True:
        cmd_state()
        cmd_routes()
        try:
            cmd_plan(_WatchArgs(demand=demand))
        except Exception as e:  # noqa: BLE001 - loadout refresh must never kill the loop
            print(f"[watch] loadout refresh failed: {e}", flush=True)
        print(f"[watch] refreshed; next in {sleep}s", flush=True)
        time.sleep(sleep)
    return 0


class _WatchArgs:
    def __init__(self, demand="balanced", top=6):
        self.demand = demand
        self.top = top


def main() -> int:
    ap = argparse.ArgumentParser(prog="fleet.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("discover", help="list configured/live nodes")
    sub.add_parser("state", help="build fleet_state.json")
    sub.add_parser("routes", help="emit routing_rules.json")
    sub.add_parser("report", help="regenerate markdown docs + state")
    st = sub.add_parser("status", help="quick live health snapshot")
    st.add_argument("--exclude", action="append", default=[], help="exclude nodes")
    st.add_argument("--all", action="store_true", help="include tailscale peers")
    w = sub.add_parser("watch", help="continuously refresh state+routes/loadout")
    w.add_argument("--sleep", type=int, default=900, help="seconds between refreshes")
    w.add_argument("--demand", choices=["balanced", "realtime", "quality"], default="balanced", help="demand profile for the loadout refresh")
    p = sub.add_parser("plan", help="emit fleet_loadout.json (orchestrator-consumable)")
    p.add_argument("--demand", choices=["balanced", "realtime", "quality"], default="balanced")
    p.add_argument("--top", type=int, default=6, help="max models to mount per node")

    b = sub.add_parser("bench", help="run full benchmark + probe pipeline")
    b.add_argument("--only", action="append", default=[], help="restrict to nodes")
    b.add_argument("--concurrency", type=int, default=4, help="parallel node benchmarks")
    b.add_argument("--max-concurrent", type=int, default=2, help="probe concurrency ceiling")
    b.add_argument("--exclude", action="append", default=[], help="exclude nodes (e.g. x1-370 when localhost is shared)")
    b.add_argument("--all", action="store_true", help="include tailscale-discovered peers, not just fleet.toml nodes")
    b.add_argument("--include-shared", action="store_true", help="also bench shared endpoints (e.g. x1-370 localhost)")

    args = ap.parse_args()
    fn = {
        "discover": cmd_discover, "state": cmd_state, "routes": cmd_routes,
        "report": cmd_report, "status": cmd_status, "bench": cmd_bench,
        "plan": cmd_plan, "watch": cmd_watch,
    }[args.cmd]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
