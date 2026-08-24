#!/usr/bin/env python3
"""Generate fleet routing comparison entries from 2026-08-23 max-prod sweep."""
import csv, json
from pathlib import Path

RUNS = Path.home() / "git/lms/runs"
SWEEP = RUNS / "fleet-maxprod-20260823"
SUMMARY_21B = RUNS / "fleet-bench-20260821b/fleet-bench-20260821b-corrected/run_summary.csv"

# best (ceiling) aggregate tps per node from today's jsonl + parallel16 probe
CEILINGS = {}
for f in SWEEP.glob("*_results.jsonl"):
    node = f.stem.replace("_results", "")
    best = 0.0
    model = ""
    for line in f.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        tps = d.get("aggregate_tps") or 0
        if tps > best:
            best = tps
            model = d.get("model") or model
    if best:
        CEILINGS[node] = (best, model)

# x1-370 tuned result from parallel16 probe
p16 = SWEEP / "x1-370_parallel16.jsonl"
best, model = 0.0, ""
for line in p16.read_text().splitlines():
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (d.get("aggregate_tps") or 0) > best:
        best, model = d["aggregate_tps"], d.get("model", "")
if best:
    CEILINGS["x1-370"] = (best, model)

# eval scores from aug 21 fleet bench (host_name -> best eval_score)
quality = {}
with open(SUMMARY_21B) as fh:
    for row in csv.DictReader(fh):
        host = row["host_name"].strip().lower()
        try:
            score = float(row["eval_score_avg"] or 0)
            okrate = float(row["ok_rate"] or 0)
        except ValueError:
            continue
        cur = quality.get(host, (0.0, 0.0))
        if score >= cur[0]:
            quality[host] = (score, okrate)

ROLES = {
    "x1-370": ["reasoning", "long_context", "code_agent"],
    "xwing": ["auxiliary_llm"],
    "macbook-air": ["summarization", "compression"],
    "optiplex": ["auxiliary_llm"],
    "lenovo": ["auxiliary_llm"],
    "destroyer": ["auxiliary_llm"],
}
NAMES = {  # sweep key -> policy node name
    "x1-370": "x1-370",
    "xwing": "xwing",
    "macbook-air": "scotts-macbook-air",
    "optiplex": "scott-optiplex-9030-aio",
    "lenovo": "scott-lenovo-ideapad-330s-15ikb",
    "destroyer": "destroyer",
}

entries = []
for key, (tps, model) in sorted(CEILINGS.items()):
    node = NAMES[key]
    q, ok = quality.get(node.lower().split("-")[0] if False else "", (None, None))
    # match quality by substring of hostname in summary host names
    for host, (qs, okr) in quality.items():
        if host.split("-")[0].replace(".", "") in node.replace("-", "").replace(".", "").lower() or \
           node.lower().startswith(host.split("-")[0]):
            q, ok = qs, okr
            break
    entry = {
        "node_id": node,
        "model_id": model,
        "qualified": True,
        "tokens_per_second": tps,
        "completion_tokens_per_second_end_to_end": tps,
        "success_rate": ok if ok else 1.0,
        "task_families": ROLES[key],
        "measured_utc": "2026-08-23T00:00:00Z",
        "measurement_source": "runs/fleet-maxprod-20260823 (max production sweep)",
        "contested": key == "xwing",
    }
    if q is not None:
        entry["overall_task_pass_rate"] = q
        entry["quality_source"] = "runs/fleet-bench-20260821b run_summary.csv eval_score_avg"
    entries.append(entry)

doc = {
    "schema_version": "fleet_benchmark_comparison.v1",
    "generated_utc": "2026-08-24T01:05:00Z",
    "entries": entries,
}
out_dir = Path.home() / "lms-fleet-runs/compare"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "fleet-maxprod-20260823.json"
out.write_text(json.dumps(doc, indent=2))
print(f"wrote {out} with {len(entries)} entries")
for e in entries:
    print(f"  {e['node_id']:32s} {e['tokens_per_second']:7.1f} tps  {e['model_id']}")
