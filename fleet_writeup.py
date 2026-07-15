#!/usr/bin/env python3
"""Generate docs/fleet_writeup.md: a per-machine fleet writeup for agents and
implementers. Aggregates machine_profile.json / run_summary.csv / model_fit.csv
under runs/<node>/ and folds in curated concurrency insights the raw benchmarks
can't capture (who multiplexes well, who chokes, >7GB limits).

Re-run any time:  python3 fleet_writeup.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = HERE / "docs" / "fleet_writeup.md"

NOTES = {
    "x1-370": {
        "concurrency": "Handles several models at once with decent concurrency. "
                       "refinedtoolcallv5-3b, orinth-1.0-35b, and qwen3.5-0.8b run concurrently "
                       "decently well; throughput drops to ~2-8 tok/s under concurrent load. "
                       "Good for multiplexing small models.",
        "cap": 2,
        "constraints": "Numbers taken while it was also the orchestrator are depressed; "
                       "re-run solo for clean figures. Tons of co-resident services compete for RAM/CPU.",
    },
    "deathstar": {
        "concurrency": "Force concurrency 1 in the probe. CPU cores are maxed by unrelated work, "
                       "so it cannot absorb parallel load.",
        "cap": 1,
        "constraints": "Models >7 GiB struggle / may stall or crash. Keep large models off this node "
                       "or accept very slow, unreliable throughput.",
    },
    "scott-optiplex-9030-aio": {"concurrency": "Chokes under concurrency. Cap at 1.", "cap": 1,
        "constraints": "Don't multiplex models here; single-stream small models only."},
    "lenovo-ideapad-330s-15ikb": {"concurrency": "Chokes under concurrency. Cap at 1.", "cap": 1,
        "constraints": "Single-stream small models only."},
    "destroyer": {"concurrency": "Chokes under concurrency. Cap at 1.", "cap": 1,
        "constraints": "Single-stream small models only."},
    "xwing": {
        "concurrency": "Small models benefit from concurrency (more sessions, acceptable latency). "
                       "Large models (e.g. Hermes-class) degrade badly at 2 concurrent sessions "
                       "(response times blow up / fail). Test at 2, prefer 1 for big models.",
        "cap": 2, "constraints": "Big-model concurrency is the danger zone here."},
    "joyner": {"concurrency": "Test at 2 concurrent.", "cap": 2, "constraints": ""},
    "beelink-ryzen-7-mini-pc": {"concurrency": "Test at 2 concurrent.", "cap": 2, "constraints": ""},
    "macbook-air": {"concurrency": "Test at 2 concurrent; mind unified-memory pressure.", "cap": 2, "constraints": ""},
    "scotts-macbook-air": {"concurrency": "Test at 2 concurrent; mind unified-memory pressure.", "cap": 2, "constraints": ""},
}
ALIAS = {"scotts-macbook-air": "macbook-air"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize_node(node_dir: Path) -> dict:
    name = node_dir.name
    info = {"name": name, "has_profile": False, "has_summary": False, "has_fit": False}
    prof = node_dir / "machine_profile.json"
    if prof.exists():
        info["has_profile"] = True
        try:
            p = json.loads(prof.read_text(encoding="utf-8"))
            cpu = p.get("cpu", {}); mem = p.get("memory", {}); gpu = p.get("gpu", {}); lms = p.get("lmstudio", {})
            info["cpu"] = cpu.get("model")
            info["cores"] = cpu.get("cores") or cpu.get("logical_processors")
            info["ram_total"] = mem.get("total_gib") or mem.get("total_ram_gib")
            info["ram_avail"] = mem.get("available_gib") or mem.get("available_ram_gib")
            info["gpu"] = gpu.get("devices") if isinstance(gpu, dict) else None
            if isinstance(lms, dict):
                info["loaded_at_profile"] = lms.get("models_loaded") or lms.get("model_count")
                info["endpoint"] = (lms.get("endpoints") or [{}])[0].get("base_url") if lms.get("endpoints") else None
        except Exception:
            pass
    summary = node_dir / "run_summary.csv"
    if summary.exists():
        info["has_summary"] = True
        rows = list(csv.DictReader(summary.open(encoding="utf-8")))
        info["n_models"] = len(rows)
        ok = [r for r in rows if _f(r.get("ok_rate")) is not None and _f(r.get("ok_rate")) >= 0.999]
        err = [r for r in rows if r not in ok]
        info["n_ok"] = len(ok); info["n_err"] = len(err)
        info["errors"] = [(r.get("model_key"), (r.get("error") or "")[:60]) for r in err][:6]
        scored = [(r.get("model_key"), _f(r.get("tps_med")) or 0.0, _f(r.get("ttft_med")) or 0.0)
                  for r in ok if r.get("model_key") and "embed" not in r.get("model_key", "").lower()]
        scored.sort(key=lambda t: t[1], reverse=True)
        info["top"] = scored[:3]; info["slow"] = scored[-3:] if scored else []
        if scored:
            info["tps_med_avg"] = sum(t[1] for t in scored) / len(scored)
    fit = node_dir / "model_fit.csv"
    if fit.exists():
        info["has_fit"] = True
        grades = {}
        for r in csv.DictReader(fit.open(encoding="utf-8")):
            g = (r.get("fit_grade") or "unknown").strip()
            grades[g] = grades.get(g, 0) + 1
        info["fit_grades"] = grades
    key = ALIAS.get(name, name)
    info["note"] = NOTES.get(key, NOTES.get(name))
    return info


def fmt_hw(info: dict) -> str:
    parts = []
    if info.get("cpu"):
        parts.append("**CPU:** " + str(info["cpu"]))
    if info.get("cores"):
        parts.append("(" + str(info["cores"]) + " logical)")
    if info.get("ram_total"):
        ram = str(info["ram_total"]) + " GiB"
        if info.get("ram_avail"):
            ram += " total / " + str(info["ram_avail"]) + " GiB avail"
        parts.append("**RAM:** " + ram)
    if info.get("gpu"):
        parts.append("**GPU:** " + str(info["gpu"]))
    if info.get("endpoint"):
        parts.append("**Endpoint:** `" + str(info["endpoint"]) + "`")
    return "  \n".join(parts) if parts else "_no profile captured_"


def main() -> int:
    nodes = sorted(p for p in RUNS.iterdir() if p.is_dir() and (p / "machine_profile.json").exists())
    infos = [summarize_node(n) for n in nodes]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_models = sum(i.get("n_models", 0) for i in infos)
    struggle = [i["name"] for i in infos if (i.get("note") or {}).get("cap") == 1]
    cap2 = [i["name"] for i in infos if (i.get("note") or {}).get("cap", 2) >= 2]

    L = []
    L.append("# Fleet Writeup - per-machine capabilities\n")
    L.append("> Generated: " + now + "  \n> Status: **preliminary** - crash-doc pass and concurrency "
             "probe may still be running. Regenerate with `python3 fleet_writeup.py`.\n")
    L.append("## Fleet overview\n")
    L.append("- **Machines profiled:** " + str(len(infos)))
    L.append("- **Model benchmark rows so far:** " + str(total_models))
    L.append("- **Concurrency-capable (tested at 2):** " + (", ".join(cap2) or "-"))
    L.append("- **Concurrency-limited (cap 1):** " + (", ".join(struggle) or "-"))
    L.append("\n### Concurrency principles (from fleet observation)\n")
    L.append("- Small models generally *benefit* from concurrency: more simultaneous sessions with "
             "acceptable latency, better aggregate throughput.")
    L.append("- Large models (Hermes-class, 30B+) degrade sharply at 2 concurrent sessions - "
             "response times balloon or requests fail. Keep big models single-stream.")
    L.append("- A handful of nodes (optiplex, lenovo, destroyer, deathstar) choke under any parallel "
             "load and are capped at 1 concurrent request in testing.")
    L.append("- deathstar additionally cannot run models >7 GiB reliably (CPU maxed by other jobs).")
    L.append("- x1-370 is the strongest concurrency node (96 GiB RAM) but co-resident services keep "
             "per-stream throughput low (~2-8 tok/s under concurrent load) - fine for multiplexing.\n")
    L.append("## Per-machine\n")
    for i in infos:
        L.append("### " + i["name"] + "\n")
        L.append(fmt_hw(i))
        L.append("")
        if i.get("has_summary"):
            L.append("- **Models benchmarked:** " + str(i.get("n_models")) +
                     " (ok: " + str(i.get("n_ok")) + ", errors: " + str(i.get("n_err")) + ")")
            if i.get("loaded_at_profile"):
                L.append("- **Models loaded at profile time:** " + str(i["loaded_at_profile"]))
            if i.get("tps_med_avg") is not None:
                L.append("- **Median tps (non-embedding, ok):** " + format(i["tps_med_avg"], ".1f"))
            if i.get("top"):
                L.append("- **Fastest:** " + ", ".join("%s (%.1f tok/s)" % (m, t) for m, t, _ in i["top"]))
            if i.get("slow"):
                L.append("- **Slowest:** " + ", ".join("%s (%.1f tok/s)" % (m, t) for m, t, _ in i["slow"]))
            if i.get("errors"):
                L.append("- **Errors/crashes:** " + "; ".join("%s%s" % (m, " (" + e + ")" if e else "") for m, e in i["errors"]))
        else:
            L.append("- _benchmark not yet complete for this node_")
        if i.get("has_fit") and i.get("fit_grades"):
            L.append("- **Fit grades:** " + str(i["fit_grades"]))
        note = i.get("note")
        if note:
            L.append("\n**Concurrency posture:** " + note.get("concurrency", ""))
            if note.get("constraints"):
                L.append("**Constraints:** " + note["constraints"])
        L.append("\n_Sources:_ `runs/" + i["name"] + "/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("Wrote " + str(OUT) + " (" + str(len(infos)) + " machines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
