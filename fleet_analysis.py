#!/usr/bin/env python3
"""Detailed fleet analysis -> docs/fleet_analysis.md.

Correctness notes:
- A model is AVAILABLE if it produced output (tps_med > 0). `ok_rate` is the
  fraction of benchmark *cases* that passed eval, NOT a binary - so it measures
  QUALITY, not availability. We keep the two separate.
- Node display = basename. Embeddings are excluded from chat-model routing.

Re-run: python3 fleet_analysis.py
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from datetime import datetime, timezone
from statistics import median

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "docs" / "fleet_analysis.md"

NOTES = {
    "x1-370": {"cap": 2, "concurrency": "Strongest concurrency node (96 GiB RAM, many co-resident "
        "services). refinedtoolcallv5-3b, orinth-1.0-35b, qwen3.5-0.8b run concurrently decently; "
        "per-stream throughput drops to ~2-8 tok/s under concurrent load. Good for multiplexing small models.",
        "constraints": "Numbers taken while it was also the orchestrator are depressed; re-run solo for clean figures."},
    "deathstar": {"cap": 1, "concurrency": "Force concurrency 1. CPU maxed by other jobs; cannot absorb parallel load.",
        "constraints": "System RAM ~20 GiB BUT only 8 GiB GPU VRAM allocated - models >~7 GiB exceed VRAM and spill to CPU (slow) or fail. CPU also maxed by other jobs."},
    "scott-optiplex-9030-aio": {"cap": 1, "concurrency": "Chokes under concurrency. Cap at 1.", "constraints": "Single-stream small models only."},
    "lenovo-ideapad-330s-15ikb": {"cap": 1, "concurrency": "Chokes under concurrency. Cap at 1.", "constraints": "Single-stream small models only."},
    "destroyer": {"cap": 1, "concurrency": "Chokes under concurrency. Cap at 1.", "constraints": "Single-stream small models only."},
    "xwing": {"cap": 2, "concurrency": "Small models benefit from concurrency. Large models (Hermes-class) degrade "
        "badly at 2 concurrent sessions. Test at 2, prefer 1 for big models.", "constraints": "Big-model concurrency is the danger zone."},
    "joyner": {"cap": 2, "concurrency": "Test at 2 concurrent.", "constraints": ""},
    "beelink-ryzen-7-mini-pc": {"cap": 2, "concurrency": "Test at 2 concurrent.", "constraints": ""},
    "macbook-air": {"cap": 2, "concurrency": "Test at 2 concurrent; mind unified-memory pressure.", "constraints": ""},
    "scotts-macbook-air": {"cap": 2, "concurrency": "Test at 2 concurrent; mind unified-memory pressure.", "constraints": ""},
}
# Canonical alias: the short name "macbook-air" actually has real data under
# the "scotts-macbook-air" dir; the bare "macbook-air" dir is a pre-SSH stale
# leftover with no host_profile.json. Redirect to the dir holding real data.
ALIAS = {"macbook-air": "scotts-macbook-air"}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def is_embed(mk):
    return mk and "embed" in mk.lower()


def load_node(name):
    name = ALIAS.get(name, name)
    d = RUNS / name
    info = {"name": name}
    # Prefer REAL per-node hardware (collect_node_profile.py run on the host).
    hw_src = "runner (unverified)"
    cpu = {}; mem = {}; lms = {}
    hp = d / "host_profile.json"
    if hp.exists():
        h = json.loads(hp.read_text(encoding="utf-8"))
        cpu = {"model": (h.get("cpu") or {}).get("model"),
               "logical_processors": (h.get("cpu") or {}).get("logical_processors")}
        mem = {"ram_total_gib": (h.get("memory") or {}).get("ram_total_gib"),
               "ram_avail_gib": (h.get("memory") or {}).get("ram_avail_gib")}
        info["gpu"] = (h.get("gpu") or [])[:2]
        info["arch"] = h.get("platform")
        info["vram_mib"] = (h.get("vram") or {}).get("vram_total_mib")
        hw_src = "per-node (verified)"
    prof = d / "machine_profile.json"
    if prof.exists():
        p = json.loads(prof.read_text(encoding="utf-8"))
        if not hp.exists():
            cpu = p.get("cpu", {}); mem = p.get("memory", {})
        lms = p.get("lmstudio", {})
    info["cpu"] = cpu.get("model"); info["cores"] = cpu.get("logical_processors")
    rt = mem.get("mem_total_bytes") or mem.get("ram_total_gib")
    ra = mem.get("mem_available_bytes") or mem.get("ram_avail_gib")
    info["ram_total"] = (round(rt / 1073741824, 1) if isinstance(rt, (int, float)) and rt > 1000 else rt)
    info["ram_avail"] = (round(ra / 1073741824, 1) if isinstance(ra, (int, float)) and ra > 1000 else ra)
    eps = (lms.get("endpoint_probes") or [{}])
    info["loaded"] = eps[0].get("model_count") if eps else None
    info["endpoint"] = eps[0].get("base_url") if eps else None
    info["hw_src"] = hw_src
    summary = d / "run_summary.csv"
    info["has_summary"] = summary.exists()
    chat = []; failed = []; scored = []; rows = []
    if summary.exists():
        rows = [r for r in csv.DictReader(summary.open(encoding="utf-8")) if not is_embed(r.get("model_key"))]
        info["n"] = len(rows)
        for r in rows:
            tps = f(r.get("tps_med")) or 0.0
            if tps > 0:
                chat.append(r)
                scored.append((r.get("model_key"), tps, f(r.get("ttft_med")) or 0.0, f(r.get("eval_score_avg")) or 0.0))
            else:
                failed.append((r.get("model_key"), (r.get("error") or r.get("eval_ok_rate") or "")[:80]))
        info["n_ran"] = len(chat); info["n_fail"] = len(failed)
        info["failed"] = failed[:10]
        scored.sort(key=lambda t: t[1], reverse=True)
        info["top"] = scored[:5]; info["bottom"] = scored[-5:][::-1] if scored else []
        if scored:
            info["tps_avg"] = sum(t[1] for t in scored) / len(scored)
            info["tps_med"] = median(t[1] for t in scored)
            qs = [t[3] for t in scored if t[3] > 0]
            info["qual_avg"] = sum(qs) / len(qs) if qs else None
    fit = d / "model_fit.csv"
    info["has_fit"] = fit.exists()
    if fit.exists():
        grades = {}; memmap = {}
        for r in csv.DictReader(fit.open(encoding="utf-8")):
            g = (r.get("fit_grade") or "unknown").strip(); grades[g] = grades.get(g, 0) + 1
            mk = r.get("model_key")
            mg = r.get("estimated_model_memory_gib") or r.get("model_memory_gib")
            if mk:
                memmap[mk] = f(mg)
        info["fit_grades"] = grades; info["memmap"] = memmap
    info["note"] = NOTES.get(ALIAS.get(name, name), NOTES.get(name))
    return info, rows


def main():
    seen = set()
    raw = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and (p / "machine_profile.json").exists())
    nodes = []
    for n in raw:
        canon = ALIAS.get(n, n)
        if canon in seen:
            continue
        seen.add(canon)
        nodes.append(n)
    infos = []; allrows = {}
    for n in nodes:
        info, rows = load_node(n)
        infos.append(info); allrows[info["name"]] = rows

    model_map = {}
    for info in infos:
        n = info["name"]
        if not info["has_summary"]:
            continue
        for r in allrows[n]:
            mk = r.get("model_key")
            if not mk or is_embed(mk):
                continue
            tps = f(r.get("tps_med")) or 0.0
            ran = tps > 0
            fit = (info.get("memmap") or {}).get(mk)
            model_map.setdefault(mk, []).append((n, tps, ran, fit))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done = [i for i in infos if i["has_summary"]]
    pending = [i for i in infos if not i["has_summary"]]

    L = []
    L.append("# Fleet Analysis - detailed capabilities & routing\n")
    L.append("> Generated: " + now)
    L.append("> **Status:** " + str(len(done)) + " nodes have a `run_summary.csv` from this pass; " + str(len(pending)) +
             " being re-benchmarked (destroyer, joyner, lenovo); concurrency probe pending. Re-run: `python3 fleet_analysis.py`.\n")
    L.append(">\n> **Data reliability - read first.** The high *Failed (no output)* counts in this pass are "
             "almost certainly **artifacts, not real model breakage**: (1) the fleet hit a disk-full condition "
             "mid-run (Docker filled the root volume), interrupting several node benchmarks and leaving zero-tps "
             "rows; (2) x1-370 was benchmarked *while also orchestrating* the other 8 nodes, heavily contending "
             "its own CPU/RAM. The initial pre-crash pass had x1-370's 22 models all producing tokens at 5-13 "
             "tok/s. **Treat per-node failure tallies as 'needs a clean solo re-run', not 'model is broken'.** "
             "Re-run contended/interrupted nodes solo (`bench_fleet.py --only <node>`) for final figures.\n")

    L.append("## Fleet overview\n")
    L.append("| Node | HW | CPU | RAM (GiB) | VRAM (GiB) | Loaded | Chat | Ran | Fail | Med tps | Cap | Status |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i in infos:
        loaded = i.get("loaded") if i.get("loaded") is not None else "-"
        tps = ("%.1f" % i["tps_med"]) if i.get("tps_med") is not None else "-"
        status = "has data" if i["has_summary"] else "REDO pending"
        cap = (i.get("note") or {}).get("cap", 2)
        cpu = (i.get("cpu") or "-")
        if len(cpu) > 26:
            cpu = cpu[:25] + "\u2026"
        ram = ("%.1f" % i["ram_total"]) if i.get("ram_total") else "-"
        vram = ("%.1f" % (i["vram_mib"] / 1024.0)) if i.get("vram_mib") else "-"
        src = "V" if i.get("hw_src") == "per-node (verified)" else "?"
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            i["name"], src, cpu, ram, vram, loaded, i.get("n", "-"), i.get("n_ran", "-"), i.get("n_fail", "-"), tps, cap, status))
    L.append("")
    L.append("_HW column: **V** = real per-node hardware (`host_profile.json`); **?** = runner profile, "
             "unverified. RAM/CPU are only meaningful where marked V._\n")
    L.append("## Hardware capture status\n")
    L.append("- Real per-node hardware collected via `collect_node_profile.py` (run on each host over SSH) "
             "for nodes marked **V** above.")
    L.append("- Nodes still on the runner fallback (**?**): SSH key / platform access not yet available. "
             "Deploy the runner's SSH key (or run `python3 collect_node_profile.py > runs/<node>/host_profile.json` "
             "locally on the node) to upgrade them to **V**.")
    L.append("")

    L.append("## Per-machine deep dive\n")
    for i in infos:
        L.append("### " + i["name"] + "\n")
        hw = []
        if i.get("cpu"):
            hw.append("**CPU:** " + str(i["cpu"]) + (" (%s logical)" % i["cores"] if i.get("cores") else ""))
        if i.get("ram_total"):
            hw.append("**RAM:** %.1f GiB (%.1f avail)" % (i["ram_total"], i.get("ram_avail") or 0))
        if i.get("gpu"):
            hw.append("**GPU:** " + "; ".join(i["gpu"]))
        if i.get("endpoint"):
            hw.append("**Endpoint:** `" + str(i["endpoint"]) + "`")
        if i.get("loaded") is not None:
            hw.append("**Models loaded at profile:** %s" % i["loaded"])
        L.append("  \n".join(hw))
        L.append("")
        if i.get("hw_src") == "per-node (verified)":
            L.append("_Hardware: real, captured on this host via `collect_node_profile.py`._")
        else:
            L.append("_Hardware: **unverified** - taken from the runner profile, not this node. Run "
                     "`collect_node_profile.py` on this host to upgrade._")
        L.append("")
        if not i["has_summary"]:
            L.append("_No validated benchmark data yet - unreachable during the last pass, being re-benchmarked now._\n")
        else:
            L.append("- **Chat models benchmarked:** %s | **Ran:** %s | **Failed (no output):** %s" %
                     (i.get("n"), i.get("n_ran"), i.get("n_fail")))
            if i.get("tps_med") is not None:
                q = ("%.2f" % i["qual_avg"]) if i.get("qual_avg") is not None else "-"
                L.append("- **Median throughput (ran, non-embed):** %.1f tok/s  |  **Avg eval score:** %s" % (i["tps_med"], q))
            if i.get("top"):
                L.append("- **Fastest (ran):** " + ", ".join("%s (%.1f tok/s, ttft %.0f ms)" % (m, t, tt) for m, t, tt, _ in i["top"]))
            if i.get("bottom"):
                L.append("- **Slowest (ran):** " + ", ".join("%s (%.1f tok/s)" % (m, t) for m, t, _, _ in i["bottom"]))
            if i.get("failed"):
                L.append("- **Failed (no output / crash):** " + "; ".join("%s%s" % (m, " (" + e + ")" if e else "") for m, e in i["failed"]))
            if i.get("has_fit") and i.get("fit_grades"):
                L.append("- **Fit grades:** " + str(i["fit_grades"]))
        note = i.get("note")
        if note:
            L.append("\n**Concurrency posture:** " + note.get("concurrency", ""))
            if note.get("constraints"):
                L.append("**Constraints:** " + note["constraints"])
        L.append("")

    L.append("## Cross-fleet model placement\n")
    L.append("For each chat model on >1 node, fastest validated home (produced output). Embeddings excluded.\n")
    L.append("| Model | Available on (tps) | Best home | Fit on best |")
    L.append("|---|---|---|---|")
    multi = {m: v for m, v in model_map.items() if len(v) > 1}
    for m in sorted(multi, key=lambda x: -len(multi[x])):
        entries = multi[m]
        ran_entries = [e for e in entries if e[2]]
        best = max(ran_entries, key=lambda e: e[1]) if ran_entries else None
        loc = ", ".join("%s (%.0f)" % (n, t) for n, t, ok, _ in sorted(entries, key=lambda e: -e[1]))
        if best:
            L.append("| %s | %s | %s | %s |" % (m, loc, best[0], ("%.2f" % best[3]) if best[3] else "-"))
        else:
            L.append("| %s | %s | _none ran_ | - |" % (m, loc))
    L.append("")

    # ---- per-node capacity (real RAM/VRAM vs model memory) ----
    model_mem = {}
    for info in infos:
        fit = RUNS / info["name"] / "model_fit.csv"
        if fit.exists():
            for r in csv.DictReader(fit.open(encoding="utf-8")):
                mk = r.get("model_key"); v = f(r.get("estimated_model_memory_gib"))
                if mk and v and mk not in model_mem:
                    model_mem[mk] = v
    L.append("## Per-node capacity (RAM / VRAM vs model memory)\n")
    L.append("Model memory is intrinsic (from `model_fit.estimated_model_memory_gib`). Effective limit = "
             "known VRAM where it is the binding constraint, else system RAM minus ~4 GiB OS headroom. "
             "Where VRAM is unknown the RAM figure may be optimistic for GPU-loaded models.\n")
    L.append("| Node | HW | Eff limit (GiB) | Basis | Models fit | Largest fit | Too-big |")
    L.append("|---|---|---:|---|---:|---|---:|")
    cap_details = []
    for i in infos:
        ram = i.get("ram_total")
        vram_gib = (i.get("vram_mib") / 1024.0) if i.get("vram_mib") else None
        if ram is None:
            L.append("| %s | ? | - | n/a (no real HW) | - | - | - |" % i["name"])
            continue
        ram_budget = max(ram - 4, ram * 0.8)
        if vram_gib and vram_gib < ram_budget:
            eff = vram_gib; basis = "VRAM"
        else:
            eff = ram_budget; basis = "RAM"
        fits = [(m, v) for m, v in model_mem.items() if v <= eff]
        big = [(m, v) for m, v in model_mem.items() if v > eff]
        largest = max(fits, key=lambda x: x[1])[0] if fits else "-"
        L.append("| %s | %s | %.1f | %s | %s | %s | %s |" % (
            i["name"], "V" if i.get("hw_src") == "per-node (verified)" else "?", eff, basis, len(fits), largest, len(big)))
        cap_details.append((i["name"], big, eff, basis))
    L.append("")
    for name, big, eff, basis in cap_details:
        if big:
            top = sorted(big, key=lambda x: -x[1])[:6]
            L.append("- **%s** (limit %.1f GiB, %s) - too big: %s" % (
                name, eff, basis, ", ".join("%s (%.1f)" % (m, v) for m, v in top)))
    L.append("")

    L.append("## Recommendations for agents & the orchestrator\n")
    L.append("### Concurrency\n")
    L.append("- **Small models (<~4B):** safe to multiplex. x1-370 is the best concurrency host; "
             "xwing/joyner/beelink/macbooks tolerate 2 concurrent sessions with acceptable latency.")
    L.append("- **Large models (>=~9B, esp. 30B+ Hermes-class):** keep **single-stream**. Two concurrent "
             "sessions to the same big model balloon latency or fail.")
    L.append("- **Cap-1 nodes (optiplex, lenovo, destroyer, deathstar):** never issue parallel requests; "
             "mount at most one model, single-stream.")
    L.append("- **deathstar:** also avoid models >7 GiB (CPU maxed by other jobs); if mounted, expect slow/unreliable.\n")
    L.append("### Loadout\n")
    L.append("- Mount **small fast tool/agent models** broadly (x1-370, xwing, beelink, macbooks, joyner) for low-latency routing.")
    L.append("- Concentrate **large quality models** on strongest RAM hosts (x1-370 96 GiB; deathstar only if <=7 GiB), single-stream.")
    L.append("- Treat cap-1 weak nodes as **single-model edge servers**.\n")
    L.append("### Data hygiene\n")
    L.append("- Track completion by `run_summary.csv`, NOT `capability_matrix.csv` (stale files from crashed runs give false 'done').")
    L.append("- `ok_rate`/`eval_ok_rate` measure QUALITY (cases passed), not availability; availability = model produced output (tps_med > 0).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("Wrote " + str(OUT) + " | " + str(len(done)) + " validated, " + str(len(pending)) + " pending, " + str(len(model_map)) + " chat models")


if __name__ == "__main__":
    raise SystemExit(main())
