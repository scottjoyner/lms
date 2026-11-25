#!/usr/bin/env python3
"""Fleet orchestrator: discover tailscale nodes, probe each LM Studio :1234, and
dynamically mount/unmount models per node to match demand — spec-aware and
tokens/sec-aware. See docs/fleet_orchestrator.md for the design.

This is the control loop that keeps the fleet's *mounted* models aligned with what
the router actually needs. It does NOT benchmark (that is bench_fleet.py /
lms_model_fit.py); it *consumes* those artifacts (runs/<node>/*.csv) as the
capability matrix.

Examples
--------
  python3 fleet_orchestrator.py discover          # list tailscale nodes + :1234 state
  python3 fleet_orchestrator.py status            # nodes, loaded models, busy, capability
  python3 fleet_orchestrator.py plan --demand realtime
  python3 fleet_orchestrator.py apply --demand realtime --apply   # actually mount/unmount
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Canonical fleet identity/discovery now lives in lms_agent_bench.fleet_discover
# (fleet.toml + tailscale, one hostname scheme). The orchestrator consumes it
# instead of maintaining its own device list. See docs/LLD_UNIFIED_FLEET.md W-70.
from lms_agent_bench.discovery import discover_tailscale_nodes
from lms_agent_bench.fleet_discover import discover as _discover_fleet_nodes

HERE = Path(__file__).resolve().parents[2]
RUNS = HERE / "runs"

LM_PORT = int(os.environ.get("LM_PORT", "1234"))
# Router fleet state (per-owner in-flight) used for busy detection.
FLEET_JSON = Path(
    __import__("os").environ.get(
        "FLEET_JSON",
        "/media/scott/SSD_4TB/knowledge/.metrics/fleet.json",
    )
)
HTTP_TIMEOUT = 8.0

# Conservative bootstrap loadout: an embedding model + a tiny model everywhere,
# plus a small/medium model on the big unified-memory nodes (the user confirmed
# xwing / x1-370 / deathstar run multiple models well; beelink can run a few).
# Keys that are not present in a node's local LM Studio library simply fail to
# load (we skip them), so this is safe to run fleet-wide.
LMSTUDIO_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
TINY_MODEL = "liquid/lfm2.5-1.2b"
SMALL_MODEL = "orinth-1.0-9b"
MEDIUM_MODEL = "google/gemma-4-12b-qat"
BIG_NODES = {"xwing", "x1-370", "deathstar", "beelink-ryzen-7-mini-pc"}

# Where the router publishes aggregated node self-reports (libraries + loaded).
FLEET_NODES_URL = os.environ.get("FLEET_NODES_URL", "http://localhost:8088/api/fleet/nodes")

# Authoritative fleet device inventory (from the tailscale device list): stable node
# identity (device id + hostname + IP + OS) so discovery is reproducible.
FLEET_BASELINE = Path(os.environ.get("FLEET_BASELINE", HERE / "fleet_baseline.csv"))

# Measured loadout produced by fleet.py plan (per-node mount lists + routing).
FLEET_LOADOUT = Path(os.environ.get("FLEET_LOADOUT", HERE / "fleet_loadout.json"))


def _params_b(model: dict | None) -> float:
    """Best-effort parameter count in billions from a library entry."""
    if not model:
        return 0.0
    raw = str(model.get("params") or model.get("paramsString") or "")
    # Handle "35B-A3B" (MoE total), "0.8B", "128x2.6B" (diffusion -> skip), "9B".
    head = raw.split("-")[0].split("x")[-1]
    try:
        return float(head.replace("B", "").replace("b", "").strip()) if "B" in raw.upper() else 0.0
    except ValueError:
        return 0.0


def conservative_from_library(library: list[dict], big_node: bool = False) -> list[str]:
    """Pick a small, representative, *routable* loadout from a node's downloaded
    library: one embedding + one fast small chat model + one mid + (big nodes) one
    large. Excludes diffusion / diffusion-distilled models (params containing 'x',
    e.g. '128x2.6b') and any model we cannot size."""
    embeds: list[tuple[float, str]] = []
    small: list[tuple[float, str]] = []
    mid: list[tuple[float, str]] = []
    large: list[tuple[float, str]] = []
    for m in library:
        key = m.get("key") or m.get("identifier") or ""
        if not key:
            continue
        params = (m.get("params") or "").lower()
        if "x" in params:  # diffusion / diffusion-distilled
            continue
        is_embed = (m.get("type") == "embedding") or ("embedding" in key.lower())
        if is_embed:
            embeds.append((0.0, key))
            continue
        b = _params_b(m)
        if b <= 0:
            continue
        if b <= 3:
            small.append((b, key))
        elif b <= 14:
            mid.append((b, key))
        else:
            large.append((b, key))
    embeds.sort(); small.sort(); mid.sort(); large.sort()
    want: list[str] = []
    if embeds:
        want.append(embeds[0][1])            # one embedding
    if small:
        want.append(small[0][1])             # fastest small
    if mid:
        want.append(mid[len(mid) // 2][1])   # a mid-size model
    if big_node and large:
        want.append(large[0][1])             # one large model for big nodes
    return want


def fetch_fleet_reports() -> dict[str, dict]:
    """Pull aggregated node self-reports from the router (libraries + loaded)."""
    try:
        with urlopen(FLEET_NODES_URL, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {n["hostname"]: n for n in data.get("nodes", [])}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _slug(hostname: str) -> str:
    out = []
    for ch in hostname.lower().strip():
        out.append(ch if ch.isalnum() else "-")
    return "".join(out).strip("-") or "node"


def discover_nodes() -> list[dict]:
    """Enumerate the fleet using the canonical discovery module
    (lms_agent_bench.fleet_discover: fleet.toml + tailscale). This is the single
    source of fleet identity now — one hostname scheme, no duplicated device
    lists. Liveness is confirmed later by probe_node()."""
    nodes: list[dict] = []
    for n in _discover_fleet_nodes():
        ip = n.url.rsplit(":", 1)[0].rsplit("://", 1)[-1]
        nodes.append({
            "hostname": n.name,
            "slug": _slug(n.name),
            "device_id": None,
            "ip": ip,
            "online": True,  # confirmed later by probe_node
            "os": "",
            "base_url": f"http://{ip}:{LM_PORT}/v1",
            "native_url": f"http://{ip}:{LM_PORT}/api/v1/models",
        })
    if nodes:
        return nodes
    # Fallback: tailscale-only discovery (no fleet.toml).
    return _discover_tailscale()


def _discover_tailscale() -> list[dict]:
    """Fallback discovery via `tailscale status --json`, delegated to the shared
    lms_agent_bench.discovery module."""
    nodes: list[dict] = []
    for n in discover_tailscale_nodes(port=LM_PORT):
        nodes.append({
            "hostname": n["name"],
            "slug": _slug(n["name"]),
            "ip": n["ip"],
            "online": bool(n.get("online", False)),
            "os": n.get("os", ""),
            "base_url": f"http://{n['ip']}:{LM_PORT}/v1",
            "native_url": f"http://{n['ip']}:{LM_PORT}/api/v1/models",
        })
    return nodes


def _load_baseline() -> list[dict]:
    """Load the fleet device inventory CSV (device_id, hostname, os, tailscale_ip).

    The committed ``fleet_baseline.csv`` is a *cache* of tailscale discovery, not a
    hardcoded source of truth: if the file is absent it is regenerated from live
    tailscale discovery so the repo no longer ships a stale device list."""
    if not FLEET_BASELINE.exists():
        _regenerate_baseline()
    if not FLEET_BASELINE.exists():
        return []
    out: list[dict] = []
    with FLEET_BASELINE.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("hostname"):
                continue
            out.append({
                "device_id": row.get("device_id"),
                "hostname": row["hostname"],
                "os": row.get("os", ""),
                "ip": row.get("tailscale_ip"),
            })
    return out


def _regenerate_baseline() -> None:
    """Write fleet_baseline.csv from live tailscale discovery (best-effort)."""
    try:
        rows = discover_tailscale_nodes(port=LM_PORT)
    except Exception:
        return
    if not rows:
        return
    try:
        FLEET_BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with FLEET_BASELINE.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["device_id", "hostname", "os", "tailscale_ip"])
            w.writeheader()
            for n in rows:
                w.writerow({
                    "device_id": "",
                    "hostname": n["name"],
                    "os": n.get("os", ""),
                    "tailscale_ip": n["ip"],
                })
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
def _http_get_json(url: str) -> Any | None:
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def probe_node(node: dict) -> dict:
    """Return live LM Studio state for a node: loaded models + reachability.

    Loaded-model detection uses the OpenAI-compatible ``/v1/models`` endpoint so it
    matches what the router and the benchmark observe. (The native ``/api/v1/models``
    can diverge on some nodes.)
    """
    info = dict(node)
    models_url = node["base_url"].rstrip("/") + "/models"
    data = _http_get_json(models_url)
    if data is None:
        info["reachable"] = False
        info["loaded_models"] = []
        info["loaded_detail"] = []
        return info
    info["reachable"] = True
    models = data.get("data", []) if isinstance(data, dict) else []
    info["loaded_models"] = [m.get("id") for m in models if isinstance(m, dict)]
    info["loaded_detail"] = models
    return info


def load_busy_map() -> dict[str, set[str]]:
    """Map of node-ip -> set of model keys currently serving (from router fleet.json)."""
    busy: dict[str, set[str]] = {}
    if not FLEET_JSON.exists():
        return busy
    try:
        data = json.loads(FLEET_JSON.read_text(encoding="utf-8"))
    except Exception:
        return busy
    # fleet.json shape: per-model entries with call counts / inflight per owner.
    for entry in data.get("models", data if isinstance(data, list) else []):
        owner = entry.get("owner") or entry.get("provider")
        inflight = entry.get("inflight", 0) or entry.get("active", 0)
        if owner and inflight:
            if "/" in owner:
                ip_part = owner.split("/")[0]
            else:
                ip_part = owner
            busy.setdefault(ip_part, set()).add(entry.get("model", ""))
    return busy


# --------------------------------------------------------------------------- #
# Capability store (consumes bench_fleet.py / lms_model_fit.py artifacts)
# --------------------------------------------------------------------------- #
@dataclass
class Capability:
    specs: dict = field(default_factory=dict)          # node-level RAM/VRAM
    models: dict = field(default_factory=dict)         # model_key -> perf + fit


def load_capability() -> dict[str, Capability]:
    """Ingest runs/<node>/{model_fit.csv, run_summary.csv} into per-node capability."""
    caps: dict[str, Capability] = {}
    if not RUNS.exists():
        return caps
    for node_dir in RUNS.iterdir():
        if not node_dir.is_dir():
            continue
        slug = node_dir.name
        cap = Capability()
        fit = node_dir / "model_fit.csv"
        if fit.exists():
            with fit.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    mk = row.get("model_key")
                    if mk:
                        cap.models.setdefault(mk, {})["fit"] = dict(row)
                    if "system_ram_gib" in row:
                        cap.specs = dict(row)
        summ = node_dir / "run_summary.csv"
        if summ.exists():
            with summ.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    mk = row.get("model_key")
                    if mk:
                        cap.models.setdefault(mk, {}).setdefault("perf", []).append(dict(row))
        # Curated, task-aware signal (reliability_grade / recommended_use per model).
        # This is the highest-quality capability data and was previously ignored by
        # the planner, which only looked at the RAM-based fit_grade.
        matrix = node_dir / "capability_matrix.csv"
        if matrix.exists():
            with matrix.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    mk = row.get("model_key")
                    if mk:
                        cap.models.setdefault(mk, {}).setdefault("matrix", []).append(dict(row))
        caps[slug] = cap
    return caps


def _merge_reporter_specs(caps: dict[str, Capability], nodes: list[dict] | None = None) -> dict[str, Capability]:
    """Override per-node specs with the REAL hardware reported by the node
    reporter (orchestrator-side fix for the benchmark spec-contamination gap).

    Benchmark runs execute on whichever host ran ``bench_fleet.py`` and only stamp
    ``host_name`` -- so every node's ``model_fit.csv`` carries that runner's specs.
    The node reporter, however, publishes this node's own ``/proc/meminfo`` /
    ``sysctl`` specs. When present and non-zero, prefer those for the RAM budget so
    a 16 GiB MacBook is not budgeted like x1-370's 91 GiB. Nodes with no reporter
    data keep their (contaminated) model_fit specs as before.

    We iterate the authoritative fleet nodes (baseline), resolve each node's report
    by hostname or tailscale IP (the reporter's source IP as seen by the router --
    a node's actual hostname need not equal its baseline slug, e.g. the MacBook Air
    reports as ``kipnerter``), then attach the specs to the cap via the tolerant
    ``_match_cap`` (runs-dir keys like ``deathstar`` / ``lenovo-ideapad-...`` do not
    match reporter hostnames directly)."""
    reports = fetch_fleet_reports()
    if not reports:
        return caps
    by_ip = {r.get("ip"): r for r in reports.values() if r.get("ip")}
    if nodes is None:
        nodes = discover_nodes()
    for n in nodes:
        slug = n["slug"]
        ip = n.get("ip")
        rep = reports.get(slug) or by_ip.get(ip)
        if rep is None:
            rep = next(
                (r for r in reports.values()
                 if slug in (r.get("hostname", ""), _slug(r.get("hostname", "")))),
                None,
            )
        if not rep:
            continue
        cap = _match_cap(caps, slug)
        if cap is None:
            continue
        rs = rep.get("specs") or {}
        ram = rs.get("available_ram_gib") or rs.get("system_ram_gib")
        try:
            ram = float(ram)
        except (TypeError, ValueError):
            ram = 0.0
        if ram <= 0:
            continue
        cap.specs = {
            "system_ram_gib": float(rs.get("system_ram_gib") or ram),
            "available_ram_gib": float(rs.get("available_ram_gib") or ram),
            "cpu_model": rs.get("cpu_model", ""),
            "cpu_cores": rs.get("cpu_cores", 0),
            "gpu": rs.get("gpu", ""),
            "source": "node-reporter",
        }
    return caps


# --------------------------------------------------------------------------- #
# Planner — the "rigid harness"
# --------------------------------------------------------------------------- #
def _match_cap(caps: dict[str, Capability], slug: str) -> Capability | None:
    """Capability lookup by node slug.

    With fleet identity unified on ``fleet_discover`` (W-70), benchmark run dirs
    (``runs/<node>``) are keyed by the same canonical node name as the slug, so an
    exact match is sufficient. We still normalize ``-``/``_`` so a benchmark dir
    stamped with one separator joins a node named with the other. The old
    prefix/suffix substring hack is removed — it silently joined unrelated nodes'
    capability data."""
    if slug in caps:
        return caps[slug]
    norm = slug.replace("_", "-").lower()
    for k, v in caps.items():
        if k.replace("_", "-").lower() == norm:
            return v
    return None


def _fit_score(cap: Capability, model_key: str) -> float:
    """Higher = better fit for this node. Uses fit_grade + available RAM headroom."""
    m = cap.models.get(model_key, {})
    fit = m.get("fit", {})
    grade = str(fit.get("fit_grade", "")).lower()
    grade_w = {"good": 1.0, "ok": 0.6, "tight": 0.3, "poor": 0.0}.get(grade, 0.5)
    try:
        avail = float(fit.get("available_ram_gib") or 0)
        need = float(fit.get("estimated_model_memory_gib") or 0)
        head = (avail - need) / avail if avail > 0 else 0.0
    except (TypeError, ValueError):
        head = 0.0
    return 0.7 * grade_w + 0.3 * max(0.0, min(1.0, head))


def _perf(cap: Capability, model_key: str, field_name: str) -> float:
    perfs = cap.models.get(model_key, {}).get("perf", [])
    if not perfs:
        return 0.0
    vals = []
    for p in perfs:
        try:
            vals.append(float(p.get(field_name, 0) or 0))
        except (TypeError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


# Reliability grade -> numeric weight. "poor"/"F" mean the model is not safe to
# mount on this node; "unknown" (no curated data) is treated as neutral so the
# raw benchmark tps/eval still apply.
_GRADE_W = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2,
            "F": 0.0, "POOR": 0.0, "UNKNOWN": 0.5, "": 0.5}


def _ok_rate(cap: Capability, model_key: str) -> float:
    """Mean benchmark success rate for this node×model. 0 means it never actually
    loaded/ran here — the real source of truth, vs the RAM-only fit_grade."""
    perfs = cap.models.get(model_key, {}).get("perf", [])
    if not perfs:
        return 0.0
    vals = []
    for p in perfs:
        try:
            vals.append(float(p.get("ok_rate", 0) or 0))
        except (TypeError, ValueError):
            pass
    return sum(vals) / len(vals) if vals else 0.0


def _reliability_w(cap: Capability, model_key: str) -> float:
    """Most-conservative reliability across task families from capability_matrix."""
    rows = cap.models.get(model_key, {}).get("matrix", [])
    if not rows:
        return 0.5
    ws = []
    for r in rows:
        g = str(r.get("reliability_grade", "") or "").strip().upper()
        ws.append(_GRADE_W.get(g, 0.5))
    return min(ws) if ws else 0.5


def plan_loadouts(nodes: list[dict], caps: dict[str, Capability], demand: str) -> list[dict]:
    """Produce mount/unmount actions per node, spec- and tps-aware.

    demand in {realtime, quality, balanced}. Mounting prefers, in order:
      realtime  -> best tokens/sec (tps_med) + low ttft, must fit specs
      quality   -> best eval score, must fit specs
      balanced  -> blend of tps and quality
    A node mounts the top-N models it can fit; we never unmount a busy model.
    """
    plan: list[dict] = []
    for node in nodes:
        if not node.get("online"):
            continue
        slug = node["slug"]
        cap = _match_cap(caps, slug)
        if cap is None:
            # No benchmark data for this node yet -> leave as-is, just report.
            plan.append({"node": slug, "ip": node["ip"], "actions": [],
                         "note": "no capability data; run bench_fleet.py",
                         "spec_source": "none"})
            continue
        # Candidate models = everything we have a fit record for on this node.
        # Gate on REAL benchmark success: a model with ok_rate==0 never actually
        # loaded/ran here (the RAM-only fit_grade lies for 27-35B models), so it is
        # not a valid mount target regardless of how it "fits".
        raw = [mk for mk in cap.models if cap.models[mk].get("fit")]
        skipped_bad: list[str] = []
        candidates = []
        for mk in raw:
            # Require the model to actually succeed on this node. Benchmarks show
            # big models (27-35B) "fit" by RAM but fail to load/run (ok_rate~0.1);
            # mounting them would just waste capacity. <0.5 ok_rate = unreliable.
            if _ok_rate(cap, mk) < 0.5:
                skipped_bad.append(mk)
                continue
            candidates.append(mk)
        if not candidates:
            note = "no benchmarked-runnable models"
            if skipped_bad:
                note += f" (skipped {len(skipped_bad)} that failed/never-benched-ok)"
            plan.append({"node": slug, "ip": node["ip"], "actions": [],
                         "note": note, "skipped_failed": sorted(skipped_bad)})
            continue

        def rank(mk: str) -> float:
            if demand == "realtime":
                base = _perf(cap, mk, "tps_med") - 0.5 * _perf(cap, mk, "ttft_med")
            elif demand == "quality":
                base = _perf(cap, mk, "eval_score_avg")
            else:
                base = 0.5 * _perf(cap, mk, "tps_med") + 50.0 * _perf(cap, mk, "eval_score_avg")
            # Demote models that are unreliable / only partly benchmarked-ok on
            # this node. Reliability comes from capability_matrix grades; models
            # with no curated data keep neutral weight and rank on raw tps/eval.
            return base * (0.4 + 0.6 * _reliability_w(cap, mk))

        candidates.sort(key=rank, reverse=True)
        # Spec budget: sum of model memory of mounted set must stay under headroom.
        try:
            budget = float(cap.specs.get("available_ram_gib", 0) or 0)
        except (TypeError, ValueError):
            budget = 0.0
        mounted: list[str] = []
        used = 0.0
        for mk in candidates:
            need = 0.0
            try:
                need = float(cap.models[mk]["fit"].get("estimated_model_memory_gib", 0) or 0)
            except (TypeError, ValueError):
                need = 0.0
            if used + need <= budget * 0.9:
                mounted.append(mk)
                used += need
            if len(mounted) >= 4:  # "each machine can run a few models"
                break

        current = set(node.get("loaded_models", []))
        target = set(mounted)
        busy = node.get("_busy", set())
        actions: list[dict] = []
        for mk in target - current:
            actions.append({"op": "load", "model": mk})
        for mk in current - target:
            if mk in busy:
                actions.append({"op": "keep_busy", "model": mk,
                                "note": "busy; not unmounting"})
            else:
                actions.append({"op": "unload", "model": mk})
        plan.append({"node": slug, "ip": node["ip"], "actions": actions,
                     "mounted": sorted(mounted), "used_gib": round(used, 1),
                     "budget_gib": round(budget, 1),
                     "spec_source": ("reporter(real)" if cap.specs.get("source") == "node-reporter"
                                     else "model_fit(CONTAMINATED)"),
                     "skipped_failed": sorted(skipped_bad)})
    return plan


# --------------------------------------------------------------------------- #
# Actuator
# --------------------------------------------------------------------------- #
INSTANCE_CACHE = Path.home() / ".cache" / "fleet_orchestrator" / "instances.json"


def _load_instance_cache() -> dict:
    try:
        return json.loads(INSTANCE_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_instance_cache(cache: dict) -> None:
    try:
        INSTANCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INSTANCE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


def _ip_from_native(native_url: str) -> str:
    try:
        return native_url.split("/")[2].rsplit(":", 1)[0]
    except Exception:
        return native_url


def _loaded_instance_ids(native_url: str, model: str) -> list[str]:
    """Return the instance_ids of currently-loaded copies of ``model`` via the
    LM Studio native API. NOTE: in current LM Studio the listing omits
    ``loaded_instances``/``instance_id``, so this is a best-effort fallback; the
    authoritative id is captured at load time (see ``_load_unload`` cache)."""
    try:
        req = Request(native_url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    out: list[str] = []
    for m in data.get("models", []):
        if m.get("key") == model or m.get("id") == model:
            for inst in (m.get("loaded_instances") or []):
                iid = inst.get("instance_id")
                if iid:
                    out.append(iid)
    return out


def _load_unload(native_url: str, op: str, model: str, cache: dict | None = None) -> tuple[bool, str, bool]:
    """Returns (ok, detail, skipped). ``skipped`` is True when an unload could not
    be resolved (no captured instance_id for an externally-loaded model) so the
    caller can treat it as non-fatal rather than a hard failure."""
    base = native_url.rsplit("/api/v1/models", 1)[0] + "/api/v1/models"
    if op == "load":
        url = f"{base}/load"
        payload = json.dumps({"model": model}).encode()
        req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
            # Capture the instance_id for later unload.
            if cache is not None:
                try:
                    iid = json.loads(body).get("instance_id")
                    if iid:
                        cache[(_ip_from_native(native_url), model)] = iid
                except Exception:
                    pass
            return True, f"{resp.status}", False
        except Exception as exc:  # pragma: no cover - network dependent
            return False, str(exc)[:120], False
    else:  # unload requires the loaded instance_id, not the model key
        ip = _ip_from_native(native_url)
        iid = (cache or {}).get((ip, model)) if cache is not None else None
        ids = [iid] if iid else _loaded_instance_ids(native_url, model)
        if not ids:
            return False, f"skip:no instance_id for {model} (externally loaded?)", True
        last = ""
        for iid in ids:
            url = f"{base}/unload"
            payload = json.dumps({"instance_id": iid}).encode()
            req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    last = f"{resp.status}"
            except Exception as exc:
                return False, str(exc)[:120], False
        return True, last, False


def apply_loadouts(plan: list[dict], dry_run: bool = True) -> None:
    cache = _load_instance_cache()
    for item in plan:
        native = f"http://{item['ip']}:{LM_PORT}/api/v1/models"
        for action in item["actions"]:
            if action["op"] in ("load", "unload"):
                if dry_run:
                    print(f"  [dry-run] {item['node']}: {action['op']} {action['model']}")
                    continue
                ok, detail, skipped = _load_unload(native, action["op"], action["model"], cache)
                tag = "SKIP" if skipped else ("OK" if ok else "FAIL")
                print(f"  {tag} {item['node']}: {action['op']} {action['model']} -> {detail}")
            else:
                print(f"  keep {item['node']}: {action['model']} ({action.get('note','')})")
    _save_instance_cache(cache)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _local_ip() -> str | None:
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=10
        ).stdout.strip().splitlines()
        return out[0] if out else None
    except Exception:
        return None


def _filter_nodes(nodes: list[dict], only: str | None) -> list[dict]:
    """Restrict to a single node. `self` resolves to the node whose IP matches this
    machine's tailscale IP (or hostname 'self')."""
    if not only:
        return nodes
    o = only.lower()
    if o == "self":
        lip = _local_ip()
        return [n for n in nodes if n.get("ip") == lip or n.get("hostname") == "self"]
    return [n for n in nodes if o in n.get("slug", "").lower() or o in n.get("hostname", "").lower()]


def cmd_discover(args: argparse.Namespace) -> None:
    nodes = _filter_nodes(discover_nodes(), args.only)
    for n in nodes:
        print(f"{n['slug']:28} online={n['online']} ip={n['ip']}  {n['os']}")


def cmd_status(args: argparse.Namespace) -> None:
    nodes = _filter_nodes(discover_nodes(), args.only)
    caps = _merge_reporter_specs(load_capability())
    busy = load_busy_map()
    for n in nodes:
        p = probe_node(n)
        n["loaded_models"] = p.get("loaded_models", [])
        n["reachable"] = p.get("reachable", False)
        n["_busy"] = busy.get(n["ip"], set())
        cap = _match_cap(caps, n["slug"])
        n_caps = len(cap.models) if cap else 0
        if cap and cap.specs:
            real = cap.specs.get("source") == "node-reporter"
            sp = (f"ram={cap.specs.get('system_ram_gib')}GiB "
                  f"avail={cap.specs.get('available_ram_gib')}GiB "
                  f"[{'reporter(real)' if real else 'model_fit(CONTAMINATED)'}]")
        else:
            sp = "no specs"
        print(f"{n['slug']:28} online={n['online']} reachable={n['reachable']} "
              f"loaded={len(n['loaded_models'])} caps={n_caps} "
              f"busy_models={len(n['_busy'])}  {sp}")
        for mk in n["loaded_models"]:
            print(f"    loaded: {mk}")


def cmd_plan(args: argparse.Namespace) -> None:
    nodes = _filter_nodes(discover_nodes(), args.only)
    caps = _merge_reporter_specs(load_capability())
    busy = load_busy_map()
    for n in nodes:
        p = probe_node(n)
        n["loaded_models"] = p.get("loaded_models", [])
        n["_busy"] = busy.get(n["ip"], set())
    plan = plan_loadouts(nodes, caps, args.demand)
    for item in plan:
        print(f"{item['node']:28} mounted_target={item.get('mounted')} "
              f"used={item.get('used_gib')}/{item.get('budget_gib')}GiB "
              f"[{item.get('spec_source','')}] note={item.get('note','')}")
        for a in item["actions"]:
            print(f"    {a['op']:10} {a['model']}")
        if item.get("skipped_failed"):
            sf = item["skipped_failed"]
            print(f"    (skipped {len(sf)} that failed benchmarks: "
                  f"{', '.join(sf[:6])}{'...' if len(sf) > 6 else ''})")


def cmd_apply(args: argparse.Namespace) -> None:
    nodes = _filter_nodes(discover_nodes(), args.only)
    caps = _merge_reporter_specs(load_capability())
    busy = load_busy_map()
    for n in nodes:
        p = probe_node(n)
        n["loaded_models"] = p.get("loaded_models", [])
        n["_busy"] = busy.get(n["ip"], set())
    plan = plan_loadouts(nodes, caps, args.demand)
    print(f"{'DRY-RUN' if not args.apply else 'APPLYING'} loadout plan (demand={args.demand})")
    apply_loadouts(plan, dry_run=not args.apply)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Conservatively mount a bootstrap loadout on every online node so the fleet
    has *something* mounted to benchmark and route to. Uses each node's reported
    library (from the fleet pubsub layer) so we mount the exact identifiers it
    actually has; tolerant of missing keys."""
    nodes = _filter_nodes(discover_nodes(), args.only)
    reports = fetch_fleet_reports()
    print(f"{'DRY-RUN' if not args.apply else 'BOOTSTRAPPING'} conservative loadouts")
    for n in nodes:
        if not n.get("online"):
            continue
        p = probe_node(n)
        native = f"http://{n['ip']}:{LM_PORT}/api/v1/models"
        rep = reports.get(n["slug"]) or reports.get(n["hostname"]) or next(
            (r for r in reports.values() if r.get("ip") and r.get("ip") == n.get("ip")), None
        )
        if rep and rep.get("library"):
            want = conservative_from_library(rep["library"], big_node=n["slug"] in BIG_NODES)
        else:
            want = []  # no reported library -> cannot guess keys safely
        loaded = set(p.get("loaded_models", []))
        for mk in want:
            if mk in loaded:
                print(f"  {n['slug']}: {mk} already loaded")
                continue
            if args.apply:
                ok, detail = _load_unload(native, "load", mk)
                print(f"  {'OK' if ok else 'SKIP'} {n['slug']}: load {mk} -> {detail}")
            else:
                print(f"  [dry-run] {n['slug']}: load {mk}")


def cmd_loadout(args: argparse.Namespace) -> None:
    """Apply a measured loadout from fleet_loadout.json (produced by fleet.py plan).

    This is the payoff for the benchmark pipeline: instead of deriving mount
    targets from capability/Capability data (which can be stale or contaminated
    by RAM-only fit grades), we use the *measured* per-node mount lists and the
    best-node-per-model routing map. We still probe each node for its currently
    loaded models and respect busy models (never unmount a serving model).
    """
    if not FLEET_LOADOUT.exists():
        print(f"no measured loadout at {FLEET_LOADOUT}; run: python3 fleet.py plan",
              file=sys.stderr)
        return
    data = json.loads(FLEET_LOADOUT.read_text(encoding="utf-8"))
    loadout_nodes = data.get("nodes", {})
    routing = data.get("routing", {})

    nodes = _filter_nodes(discover_nodes(), args.only)
    busy = load_busy_map()
    plan: list[dict] = []
    for n in nodes:
        p = probe_node(n)
        if not p.get("reachable", False):
            print(f"{n['slug']:28} SKIP (unreachable)", file=sys.stderr)
            continue
        n["loaded_models"] = p.get("loaded_models", [])
        n["_busy"] = busy.get(n["ip"], set())

        entry = loadout_nodes.get(n["slug"]) or loadout_nodes.get(n["hostname"])
        if not entry:
            # slug not in measured loadout (e.g. tailscale alias mismatch) -> skip
            plan.append({"node": n["slug"], "ip": n["ip"], "actions": [],
                         "mounted": [], "used_gib": 0.0, "budget_gib": 0.0,
                         "spec_source": "measured-loadout",
                         "note": "node absent from fleet_loadout.json"})
            continue
        target = set(entry.get("mount", []))
        current = set(n["loaded_models"])
        actions: list[dict] = []
        for mk in target - current:
            actions.append({"op": "load", "model": mk})
        for mk in current - target:
            if mk in n["_busy"]:
                actions.append({"op": "keep_busy", "model": mk,
                                "note": "busy; not unmounting"})
            else:
                actions.append({"op": "unload", "model": mk})
        plan.append({"node": n["slug"], "ip": n["ip"], "actions": actions,
                     "mounted": sorted(target),
                     "used_gib": 0.0,
                     "budget_gib": round(entry.get("ram_gib") or 0.0, 1),
                     "spec_source": "measured-loadout",
                     "max_concurrency": entry.get("max_concurrency"),
                     "note": f"routing_best={routing.get(next(iter(target),''),{}).get('best_node') if target else None}"})

    header = f"{'DRY-RUN' if not args.apply else 'APPLYING'} measured loadout from {FLEET_LOADOUT.name} (demand={data.get('demand')})"
    print(header)
    apply_loadouts(plan, dry_run=not args.apply)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fleet model-loadout orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_disc = sub.add_parser("discover")
    p_disc.add_argument("--only", default=None, help="restrict to one node (slug/hostname, or 'self')")
    p_disc.set_defaults(func=cmd_discover)
    p_stat = sub.add_parser("status")
    p_stat.add_argument("--only", default=None, help="restrict to one node (slug/hostname, or 'self')")
    p_stat.set_defaults(func=cmd_status)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--demand", choices=["realtime", "quality", "balanced"], default="balanced")
    p_plan.add_argument("--only", default=None, help="restrict to one node (slug/hostname, or 'self')")
    p_plan.set_defaults(func=cmd_plan)
    p_apply = sub.add_parser("apply")
    p_apply.add_argument("--demand", choices=["realtime", "quality", "balanced"], default="balanced")
    p_apply.add_argument("--apply", action="store_true", help="actually mount/unmount (default dry-run)")
    p_apply.add_argument("--only", default=None, help="restrict to one node (slug/hostname, or 'self')")
    p_apply.set_defaults(func=cmd_apply)
    p_boot = sub.add_parser("bootstrap")
    p_boot.add_argument("--apply", action="store_true", help="actually mount (default dry-run)")
    p_boot.add_argument("--only", default=None, help="restrict to one node (slug/hostname, or 'self')")
    p_boot.set_defaults(func=cmd_bootstrap)
    p_load = sub.add_parser("loadout", help="apply measured loadout from fleet_loadout.json")
    p_load.add_argument("--apply", action="store_true", help="actually mount/unmount (default dry-run)")
    p_load.add_argument("--only", default=None, help="restrict to one node (slug/hostname, or 'self')")
    p_load.set_defaults(func=cmd_loadout)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
