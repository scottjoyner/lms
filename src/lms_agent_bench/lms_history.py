#!/usr/bin/env python3
"""SQLite-backed history index for lms-bench runs.

Agents can ingest completed run directories into a durable local database and
query the best known model/endpoint per task family without scanning folders.
The database is local-only and uses Python's stdlib sqlite3.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DB = Path(os.environ.get("LMS_BENCH_HISTORY_DB", "~/.local/share/lms-bench/history.sqlite3")).expanduser()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def db_path(path: Optional[str]) -> Path:
    return Path(path).expanduser() if path else DEFAULT_DB


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            run_dir TEXT NOT NULL,
            created_at_utc TEXT,
            ingested_at_utc TEXT NOT NULL,
            audit_status TEXT,
            audit_ok INTEGER,
            host_name TEXT,
            platform TEXT,
            endpoints_json TEXT
        );

        CREATE TABLE IF NOT EXISTS capabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            task_family TEXT NOT NULL,
            model_key TEXT NOT NULL,
            base_url TEXT NOT NULL,
            host_name TEXT,
            score REAL,
            grade TEXT,
            reliability_grade TEXT,
            throughput_grade TEXT,
            latency_grade TEXT,
            max_reliable_context_tokens INTEGER,
            evidence TEXT,
            recommended_use TEXT,
            avoid_use TEXT,
            UNIQUE(run_id, task_family, model_key, base_url),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_cap_task_score ON capabilities(task_family, score DESC);
        CREATE INDEX IF NOT EXISTS idx_cap_model ON capabilities(model_key);
        CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at_utc DESC);
        """
    )
    conn.commit()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def run_metadata(run_dir: Path) -> Dict[str, Any]:
    cfg = read_json(run_dir / "lms_run_config.json")
    audit = read_json(run_dir / "run_audit.json")
    profile = read_json(run_dir / "machine_profile.json")
    host = profile.get("host") or {}
    run_id = str(cfg.get("run_id") or run_dir.name)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "created_at_utc": cfg.get("created_at_utc") or cfg.get("started_at") or "",
        "ingested_at_utc": utc_now_iso(),
        "audit_status": audit.get("status", "unknown"),
        "audit_ok": 1 if audit.get("ok") else 0,
        "host_name": host.get("hostname") or "",
        "platform": host.get("platform") or "",
        "endpoints_json": json.dumps(cfg.get("endpoints", []), sort_keys=True),
    }


def ingest_run(conn: sqlite3.Connection, run_dir: Path) -> Dict[str, Any]:
    meta = run_metadata(run_dir)
    cap_rows = read_csv(run_dir / "capability_matrix.csv")
    conn.execute(
        """
        INSERT INTO runs(run_id, run_dir, created_at_utc, ingested_at_utc, audit_status, audit_ok, host_name, platform, endpoints_json)
        VALUES(:run_id, :run_dir, :created_at_utc, :ingested_at_utc, :audit_status, :audit_ok, :host_name, :platform, :endpoints_json)
        ON CONFLICT(run_id) DO UPDATE SET
            run_dir=excluded.run_dir,
            created_at_utc=excluded.created_at_utc,
            ingested_at_utc=excluded.ingested_at_utc,
            audit_status=excluded.audit_status,
            audit_ok=excluded.audit_ok,
            host_name=excluded.host_name,
            platform=excluded.platform,
            endpoints_json=excluded.endpoints_json
        """,
        meta,
    )
    conn.execute("DELETE FROM capabilities WHERE run_id = ?", (meta["run_id"],))
    inserted = 0
    for row in cap_rows:
        conn.execute(
            """
            INSERT INTO capabilities(
                run_id, task_family, model_key, base_url, host_name, score, grade,
                reliability_grade, throughput_grade, latency_grade,
                max_reliable_context_tokens, evidence, recommended_use, avoid_use
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta["run_id"],
                row.get("task_family", "general"),
                row.get("model_key", ""),
                row.get("base_url", ""),
                row.get("host_name", ""),
                fnum(row.get("score")),
                row.get("grade", ""),
                row.get("reliability_grade", ""),
                row.get("throughput_grade", ""),
                row.get("latency_grade", ""),
                int(fnum(row.get("max_reliable_context_tokens"))) if row.get("max_reliable_context_tokens") else None,
                row.get("evidence", ""),
                row.get("recommended_use", ""),
                row.get("avoid_use", ""),
            ),
        )
        inserted += 1
    conn.commit()
    return {"run_id": meta["run_id"], "capability_rows": inserted, "audit_status": meta["audit_status"]}


def ingest_many(conn: sqlite3.Connection, paths: Sequence[Path]) -> List[Dict[str, Any]]:
    results = []
    for path in paths:
        if path.is_dir() and (path / "capability_matrix.csv").exists():
            results.append(ingest_run(conn, path))
        elif path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir() and (child / "capability_matrix.csv").exists():
                    results.append(ingest_run(conn, child))
    return results


def query_rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def list_runs(conn: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    return query_rows(conn, "SELECT * FROM runs ORDER BY created_at_utc DESC, ingested_at_utc DESC LIMIT ?", (limit,))


def best_routes(conn: sqlite3.Connection, task: Optional[str], limit: int, min_score: float) -> List[Dict[str, Any]]:
    where = "WHERE c.score >= ?"
    params: List[Any] = [min_score]
    if task:
        where += " AND c.task_family = ?"
        params.append(task)
    params.append(limit)
    return query_rows(
        conn,
        f"""
        SELECT c.*, r.audit_status, r.audit_ok, r.created_at_utc, r.run_dir
        FROM capabilities c
        JOIN runs r ON r.run_id = c.run_id
        {where}
        ORDER BY c.task_family ASC, c.score DESC, r.created_at_utc DESC
        LIMIT ?
        """,
        params,
    )


def render_table(rows: List[Dict[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "No rows."
    widths = {field: len(field) for field in fields}
    for row in rows:
        for field in fields:
            widths[field] = max(widths[field], len(str(row.get(field, ""))))
    lines = []
    lines.append("  ".join(str(field).ljust(widths[field]) for field in fields))
    lines.append("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        lines.append("  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields))
    return "\n".join(lines)


def cmd_ingest(args: argparse.Namespace) -> int:
    with connect(db_path(args.db)) as conn:
        results = ingest_many(conn, [Path(p) for p in args.paths])
    print(json.dumps({"ok": bool(results), "ingested": results}, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if results else 1


def cmd_runs(args: argparse.Namespace) -> int:
    with connect(db_path(args.db)) as conn:
        rows = list_runs(conn, args.limit)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_table(rows, ["run_id", "audit_status", "host_name", "created_at_utc", "run_dir"]))
    return 0 if rows else 1


def cmd_best(args: argparse.Namespace) -> int:
    with connect(db_path(args.db)) as conn:
        rows = best_routes(conn, args.task, args.limit, args.min_score)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_table(rows, ["task_family", "model_key", "score", "grade", "audit_status", "created_at_utc", "run_dir"]))
    return 0 if rows else 1


def cmd_stats(args: argparse.Namespace) -> int:
    with connect(db_path(args.db)) as conn:
        rows = query_rows(
            conn,
            """
            SELECT task_family, COUNT(*) AS rows, MAX(score) AS best_score, AVG(score) AS avg_score
            FROM capabilities
            GROUP BY task_family
            ORDER BY task_family
            """,
        )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_table(rows, ["task_family", "rows", "best_score", "avg_score"]))
    return 0 if rows else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SQLite history index for lms-bench runs.")
    parser.add_argument("--db", default=None, help=f"History DB path; default {DEFAULT_DB}")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest one or more run directories or a parent runs directory")
    ingest.add_argument("paths", nargs="+", help="Run directories or parent directory containing runs")
    ingest.add_argument("--pretty", action="store_true")
    ingest.set_defaults(func=cmd_ingest)

    runs = sub.add_parser("runs", help="List ingested runs")
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    best = sub.add_parser("best", help="Show best known routes")
    best.add_argument("--task", default=None)
    best.add_argument("--limit", type=int, default=20)
    best.add_argument("--min-score", type=float, default=0.0)
    best.add_argument("--json", action="store_true")
    best.set_defaults(func=cmd_best)

    stats = sub.add_parser("stats", help="Show aggregate history stats by task family")
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
