#!/usr/bin/env python3
"""
Generate:
  - Recommendation tables (Markdown + CSV)
  - routing.json

Inputs:
  - SQLite DB that contains benchmark results from benchmark_lmstudio_inventory.py

Outputs:
  - <out_dir>/recommendations.md
  - <out_dir>/recommendations.csv
  - <out_dir>/routing.json

Run:
  python3 recommend_and_route.py \
    --db lmstudio_inventory.sqlite \
    --run-id latest \
    --out-dir sidecar_bench/reco

Notes:
- Uses only "phase='run'" rows for performance/quality metrics.
- Uses "phase='load'" as the model load latency estimate.
- Prefers judge_quality when present, otherwise auto_quality.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def median(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.median(vals))


def mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def norm_minmax(x: Optional[float], lo: float, hi: float, invert: bool = False) -> float:
    """
    Normalize into [0,1] using fixed bounds. Out-of-range is clipped.
    invert=True means smaller is better (e.g., TTFT, load time).
    """
    if x is None:
        return 0.0
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    v = clamp01(v)
    return 1.0 - v if invert else v


@dataclass
class Agg:
    endpoint_id: int
    base_url: str
    host_name: Optional[str]
    host_ip: Optional[str]
    model_id: int
    model_key: str

    ok_rate: float
    ttft_med: Optional[float]
    tps_med: Optional[float]
    wall_med: Optional[float]
    load_s: Optional[float]

    auto_q_mean: Optional[float]
    judge_q_mean: Optional[float]

    # derived "quality" (judge preferred)
    q_mean: Optional[float]


# -----------------------------
# Task profiles / weights
# -----------------------------
# You can tune these weights to match your orchestration goals.
PROFILES: Dict[str, Dict[str, Any]] = {
    # interactive chat: TTFT matters most
    "interactive_chat": {
        "w_quality": 0.35,
        "w_ttft": 0.40,
        "w_tps": 0.15,
        "w_ok": 0.10,
        "w_load": 0.00,
        "task_types": ["reasoning", "summarize"],  # cases used to compute quality components
    },
    # batch summarization: throughput matters most
    "batch_summarize": {
        "w_quality": 0.30,
        "w_ttft": 0.10,
        "w_tps": 0.40,
        "w_ok": 0.10,
        "w_load": 0.10,
        "task_types": ["summarize"],
    },
    # extraction: JSON correctness dominates
    "extraction_json": {
        "w_quality": 0.55,
        "w_ttft": 0.10,
        "w_tps": 0.10,
        "w_ok": 0.25,
        "w_load": 0.00,
        "task_types": ["json"],
    },
    # coding assistant: quality + ok rate (shape correctness) + some speed
    "coding": {
        "w_quality": 0.45,
        "w_ttft": 0.15,
        "w_tps": 0.20,
        "w_ok": 0.20,
        "w_load": 0.00,
        "task_types": ["code"],
    },
    # harder reasoning/math: quality and ok rate matter most
    "reasoning_hard": {
        "w_quality": 0.55,
        "w_ttft": 0.15,
        "w_tps": 0.10,
        "w_ok": 0.20,
        "w_load": 0.00,
        "task_types": ["math", "reasoning"],
    },
}

# Fixed normalization bounds (tune as you see real data)
NORM_BOUNDS = {
    "ttft_s": (0.15, 8.0),     # smaller better
    "tps": (1.0, 120.0),       # bigger better
    "load_s": (0.2, 45.0),     # smaller better
    "quality": (0.0, 1.0),     # bigger better
}


def pick_run_id(conn: sqlite3.Connection, run_id_arg: str) -> int:
    if run_id_arg.lower() == "latest":
        row = conn.execute("SELECT run_id FROM bench_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        if not row:
            raise SystemExit("No bench_runs found. Run the benchmark script first.")
        return int(row["run_id"])
    try:
        return int(run_id_arg)
    except Exception:
        raise SystemExit("--run-id must be an integer or 'latest'.")


def load_aggs(conn: sqlite3.Connection, run_id: int) -> List[Agg]:
    """
    Aggregate benchmark rows per (endpoint, model).
    - Performance from phase='run' ok=1
    - Load from phase='load'
    - Quality from auto_quality/judge_quality on phase='run' ok=1
    """
    # endpoint/model identity + host (best-effort; host tables may have many-to-one)
    id_q = """
    SELECT
      e.endpoint_id,
      e.base_url,
      h.ts_name AS host_name,
      h.ts_ip AS host_ip,
      m.model_id,
      m.model_key
    FROM bench_results br
    JOIN endpoints e ON e.endpoint_id = br.endpoint_id
    JOIN models m ON m.model_id = br.model_id
    LEFT JOIN host_endpoints he ON he.endpoint_id = e.endpoint_id
    LEFT JOIN hosts h ON h.host_id = he.host_id
    WHERE br.run_id = ?
    GROUP BY e.endpoint_id, m.model_id
    ORDER BY e.endpoint_id, m.model_key
    """
    ids = [dict(r) for r in conn.execute(id_q, (run_id,)).fetchall()]

    aggs: List[Agg] = []

    for row in ids:
        endpoint_id = row["endpoint_id"]
        model_id = row["model_id"]

        # ok_rate across run rows (all cases/repeats) for this model+endpoint
        ok_rows = conn.execute(
            """
            SELECT ok FROM bench_results
            WHERE run_id=? AND endpoint_id=? AND model_id=? AND phase='run'
            """,
            (run_id, endpoint_id, model_id),
        ).fetchall()
        ok_rate = (sum(int(r["ok"]) for r in ok_rows) / len(ok_rows)) if ok_rows else 0.0

        # perf on successful runs only
        perf = conn.execute(
            """
            SELECT wall_s, ttft_s, tokens_per_sec, auto_quality, judge_quality
            FROM bench_results
            WHERE run_id=? AND endpoint_id=? AND model_id=? AND phase='run' AND ok=1
            """,
            (run_id, endpoint_id, model_id),
        ).fetchall()

        wall_med = median([r["wall_s"] for r in perf if r["wall_s"] is not None])
        ttft_med = median([r["ttft_s"] for r in perf if r["ttft_s"] is not None])
        tps_med = median([r["tokens_per_sec"] for r in perf if r["tokens_per_sec"] is not None])

        auto_q_mean = mean([r["auto_quality"] for r in perf if r["auto_quality"] is not None])
        judge_q_mean = mean([r["judge_quality"] for r in perf if r["judge_quality"] is not None])
        q_mean = judge_q_mean if judge_q_mean is not None else auto_q_mean

        # load time from phase='load' (best effort: take the first load row)
        load_row = conn.execute(
            """
            SELECT wall_s
            FROM bench_results
            WHERE run_id=? AND endpoint_id=? AND model_id=? AND phase='load'
            ORDER BY result_id ASC LIMIT 1
            """,
            (run_id, endpoint_id, model_id),
        ).fetchone()
        load_s = float(load_row["wall_s"]) if load_row and load_row["wall_s"] is not None else None

        aggs.append(
            Agg(
                endpoint_id=endpoint_id,
                base_url=row["base_url"],
                host_name=row.get("host_name"),
                host_ip=row.get("host_ip"),
                model_id=model_id,
                model_key=row["model_key"],
                ok_rate=float(ok_rate),
                ttft_med=ttft_med,
                tps_med=tps_med,
                wall_med=wall_med,
                load_s=load_s,
                auto_q_mean=auto_q_mean,
                judge_q_mean=judge_q_mean,
                q_mean=q_mean,
            )
        )

    return aggs


def case_task_type_map(conn: sqlite3.Connection, run_id: int) -> Dict[int, str]:
    """
    Map case_id -> task_type for cases referenced by this run.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT bc.case_id, bc.task_type
        FROM bench_results br
        JOIN bench_cases bc ON bc.case_id = br.case_id
        WHERE br.run_id=?
        """,
        (run_id,),
    ).fetchall()
    return {int(r["case_id"]): str(r["task_type"]) for r in rows}


def score_profile_for_endpoint_model(
    conn: sqlite3.Connection,
    run_id: int,
    endpoint_id: int,
    model_id: int,
    agg: Agg,
    profile: Dict[str, Any],
    caseid_to_tasktype: Dict[int, str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a weighted score in [0,1] for a given profile.
    For "quality", we recompute q_mean restricted to profile task_types when possible.
    """
    task_types = set(profile["task_types"])

    # Restrict quality to certain cases if we can:
    perf_rows = conn.execute(
        """
        SELECT br.case_id, br.auto_quality, br.judge_quality
        FROM bench_results br
        WHERE br.run_id=? AND br.endpoint_id=? AND br.model_id=? AND br.phase='run' AND br.ok=1
        """,
        (run_id, endpoint_id, model_id),
    ).fetchall()

    auto_vals = []
    judge_vals = []
    used = 0
    for r in perf_rows:
        tt = caseid_to_tasktype.get(int(r["case_id"]))
        if tt in task_types:
            used += 1
            if r["auto_quality"] is not None:
                auto_vals.append(float(r["auto_quality"]))
            if r["judge_quality"] is not None:
                judge_vals.append(float(r["judge_quality"]))

    auto_q = mean(auto_vals) if auto_vals else None
    judge_q = mean(judge_vals) if judge_vals else None
    q = judge_q if judge_q is not None else (auto_q if auto_q is not None else agg.q_mean)

    # Normalize components
    qn = norm_minmax(q, *NORM_BOUNDS["quality"], invert=False)
    ttftn = norm_minmax(agg.ttft_med, *NORM_BOUNDS["ttft_s"], invert=True)
    tpsn = norm_minmax(agg.tps_med, *NORM_BOUNDS["tps"], invert=False)
    loadn = norm_minmax(agg.load_s, *NORM_BOUNDS["load_s"], invert=True)
    okn = clamp01(agg.ok_rate)

    score = (
        profile["w_quality"] * qn
        + profile["w_ttft"] * ttftn
        + profile["w_tps"] * tpsn
        + profile["w_load"] * loadn
        + profile["w_ok"] * okn
    )

    details = {
        "score": round(score, 6),
        "ok_rate": round(okn, 4),
        "ttft_med_s": agg.ttft_med,
        "tps_med": agg.tps_med,
        "load_s": agg.load_s,
        "quality_mean": q,
        "quality_source": "judge" if judge_q is not None else ("auto" if auto_q is not None else "fallback"),
        "quality_cases_used": used,
        "norm": {
            "quality": round(qn, 4),
            "ttft": round(ttftn, 4),
            "tps": round(tpsn, 4),
            "load": round(loadn, 4),
            "ok": round(okn, 4),
        },
        "weights": {k: profile[k] for k in ["w_quality", "w_ttft", "w_tps", "w_load", "w_ok"]},
    }

    return score, details


def write_recommendations_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("# Model Recommendations")
    lines.append("")
    lines.append(f"- Generated (UTC): `{utc_now_iso()}`")
    lines.append("")
    lines.append("| Profile | Host | Endpoint | Model | Score | OK% | Load s | TTFT s | TPS | Quality |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['profile']}` | `{r.get('host_name') or ''}` | `{r['base_url']}` | `{r['model_key']}` | "
            f"{r['score']:.4f} | {r['ok_rate']*100:.1f} | {fmt(r.get('load_s'))} | {fmt(r.get('ttft_med_s'))} | "
            f"{fmt(r.get('tps_med'))} | {fmt(r.get('quality_mean'))} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_recommendations_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "profile",
        "host_name",
        "host_ip",
        "base_url",
        "model_key",
        "score",
        "ok_rate",
        "load_s",
        "ttft_med_s",
        "tps_med",
        "quality_mean",
        "quality_source",
        "quality_cases_used",
        "details_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in cols}
            out["details_json"] = json.dumps(r.get("details", {}))
            w.writerow(out)


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB containing benchmark + inventory tables")
    ap.add_argument("--run-id", default="latest", help="Run id integer or 'latest'")
    ap.add_argument("--out-dir", default="reco_out")
    ap.add_argument("--min-ok-rate", type=float, default=0.60, help="Filter out unstable models")
    ap.add_argument("--emit-per-endpoint-routing", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = db_connect(args.db)
    try:
        run_id = pick_run_id(conn, args.run_id)
        run_row = conn.execute("SELECT * FROM bench_runs WHERE run_id=?", (run_id,)).fetchone()
        if not run_row:
            raise SystemExit(f"Run id {run_id} not found.")

        started_at = str(run_row["started_at_utc"])
        config = json.loads(run_row["config_json"]) if run_row["config_json"] else {}

        aggs = load_aggs(conn, run_id)
        if not aggs:
            raise SystemExit("No aggregates found. Ensure bench_results exists for this run.")

        caseid_to_tasktype = case_task_type_map(conn, run_id)

        # Build routing decisions
        # For each profile, choose best model per endpoint; also compute a global best per profile.
        routing: Dict[str, Any] = {
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "run_started_at_utc": started_at,
            "db": os.path.abspath(args.db),
            "profiles": PROFILES,
            "norm_bounds": NORM_BOUNDS,
            "routing": {},
            "global_defaults": {},
        }

        reco_rows_md: List[Dict[str, Any]] = []

        # Index aggs by endpoint
        by_endpoint: Dict[int, List[Agg]] = {}
        for a in aggs:
            by_endpoint.setdefault(a.endpoint_id, []).append(a)

        for profile_name, profile in PROFILES.items():
            routing["routing"][profile_name] = {}

            # Global best across all endpoints for this profile
            global_candidates: List[Tuple[float, Agg, Dict[str, Any]]] = []

            for endpoint_id, models in by_endpoint.items():
                endpoint_candidates: List[Tuple[float, Agg, Dict[str, Any]]] = []

                for a in models:
                    if a.ok_rate < args.min_ok_rate:
                        continue
                    sc, details = score_profile_for_endpoint_model(
                        conn, run_id, a.endpoint_id, a.model_id, a, profile, caseid_to_tasktype
                    )
                    endpoint_candidates.append((sc, a, details))
                    global_candidates.append((sc, a, details))

                endpoint_candidates.sort(key=lambda x: x[0], reverse=True)
                if endpoint_candidates:
                    best_sc, best_a, best_details = endpoint_candidates[0]
                    routing["routing"][profile_name][best_a.base_url] = {
                        "host_name": best_a.host_name,
                        "host_ip": best_a.host_ip,
                        "model": best_a.model_key,
                        "score": best_sc,
                        "details": best_details,
                    }

                    reco_rows_md.append({
                        "profile": profile_name,
                        "host_name": best_a.host_name,
                        "host_ip": best_a.host_ip,
                        "base_url": best_a.base_url,
                        "model_key": best_a.model_key,
                        "score": best_sc,
                        "ok_rate": best_a.ok_rate,
                        "load_s": best_a.load_s,
                        "ttft_med_s": best_a.ttft_med,
                        "tps_med": best_a.tps_med,
                        "quality_mean": best_details.get("quality_mean"),
                        "quality_source": best_details.get("quality_source"),
                        "quality_cases_used": best_details.get("quality_cases_used"),
                        "details": best_details,
                    })

            global_candidates.sort(key=lambda x: x[0], reverse=True)
            if global_candidates:
                gsc, ga, gdetails = global_candidates[0]
                routing["global_defaults"][profile_name] = {
                    "model": ga.model_key,
                    "endpoint": ga.base_url,
                    "host_name": ga.host_name,
                    "host_ip": ga.host_ip,
                    "score": gsc,
                    "details": gdetails,
                }
            else:
                routing["global_defaults"][profile_name] = None

        # Write outputs
        md_path = out_dir / "recommendations.md"
        csv_path = out_dir / "recommendations.csv"
        routing_path = out_dir / "routing.json"

        write_recommendations_md(md_path, reco_rows_md)
        write_recommendations_csv(csv_path, reco_rows_md)
        routing_path.write_text(json.dumps(routing, indent=2), encoding="utf-8")

        print(f"Run: {run_id} (started {started_at})")
        print(f"Wrote: {md_path}")
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {routing_path}")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())



# python3 recommend_and_route.py \
#   --db lmstudio_inventory.sqlite \
#   --run-id latest \
#   --out-dir sidecar_bench/reco \
#   --min-ok-rate 0.60


# {
#   "routing": {
#     "interactive_chat": {
#       "http://100.96.121.98:1234/v1": {
#         "model": "openai/gpt-oss-20b",
#         "score": 0.8123,
#         "details": { "...": "..." }
#       }
#     }
#   },
#   "global_defaults": {
#     "interactive_chat": {
#       "endpoint": "http://100.96.121.98:1234/v1",
#       "model": "openai/gpt-oss-20b",
#       "score": 0.8123
#     }
#   }
# }
