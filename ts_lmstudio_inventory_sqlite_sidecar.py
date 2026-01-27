#!/usr/bin/env python3
"""
Discover LM Studio OpenAI-compatible models on a set of Tailscale IPs,
store inventory in SQLite, and emit Markdown sidecar files.

Install:
  pip install requests

Run:
  python3 ts_lmstudio_inventory_sqlite_sidecar.py \
    --db lmstudio_inventory.sqlite \
    --sidecar-dir sidecar_md \
    --port 1234 \
    --scheme http \
    --api-root /v1

Env (optional):
  LMSTUDIO_API_KEY=...   # if your OpenAI-compatible server requires it
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException


# ----------------------------
# Your Tailscale device list
# ----------------------------
RAW_DEVICES = [
    {"ts_name": "deathstar-xps-8920", "user_email": "kipnerter@gmail.com", "ts_ip": "100.78.106.121", "ts_version": "1.94.1", "os": "Linux 6.14.0-37-generic"},
    {"ts_name": "demo", "user_email": "kipnerter@gmail.com", "ts_ip": "100.67.106.114", "ts_version": "1.94.1", "os": "Windows 11 25H2"},
    {"ts_name": "destroyer", "user_email": "kipnerter@gmail.com", "ts_ip": "100.81.57.77", "ts_version": "1.92.3", "os": "Linux 6.14.0-33-generic"},
    {"ts_name": "iphone-12-pro-max", "user_email": "kipnerter@gmail.com", "ts_ip": "100.96.196.106", "ts_version": "1.92.3", "os": "iOS 17.5.1"},
    {"ts_name": "mini-pc-22", "user_email": "kipnerter@gmail.com", "ts_ip": "100.96.121.98", "ts_version": "1.92.3", "os": "Windows 11 25H2"},
    {"ts_name": "r2d2", "user_email": "kipnerter@gmail.com", "ts_ip": "100.105.87.118", "ts_version": "1.92.3", "os": "Windows 11 25H2"},
    {"ts_name": "raspberrypi", "user_email": "kipnerter@gmail.com", "ts_ip": "100.114.88.89", "ts_version": "1.92.5", "os": "Linux 6.12.62+rpt-rpi-v8"},
    {"ts_name": "scotts-macbook-air", "user_email": "kipnerter@gmail.com", "ts_ip": "100.85.64.117", "ts_version": "1.92.3", "os": "macOS 14.5.0"},
    {"ts_name": "scotts-macbook-pro-1", "user_email": "kipnerter@gmail.com", "ts_ip": "100.111.79.107", "ts_version": "1.92.3", "os": "macOS 13.0.0"},
    {"ts_name": "tie", "user_email": "kipnerter@gmail.com", "ts_ip": "100.96.18.65", "ts_version": "1.92.3", "os": "Linux 6.8.0-41-generic"},
]


# ----------------------------
# Helpers / Models
# ----------------------------
@dataclass
class ProbeResult:
    reachable: bool
    status_code: Optional[int]
    error: Optional[str]
    models: List[Dict[str, Any]]
    raw_response: Optional[Dict[str, Any]]
    checked_at_utc: str


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slugify_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def build_base_url(scheme: str, ip: str, port: int, api_root: str) -> str:
    api_root = api_root.strip()
    if not api_root.startswith("/"):
        api_root = "/" + api_root
    api_root = api_root.rstrip("/")
    return f"{scheme}://{ip}:{port}{api_root}"


def safe_json(resp: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return resp.json()
    except Exception:
        return None


def fetch_models_openai_compatible(
    base_url: str,
    timeout_s: float,
    api_key: Optional[str],
) -> ProbeResult:
    """
    Primary: GET {base_url}/models
    Fallbacks:
      - GET {base_url}/v1/models (if base_url doesn't already end with /v1)
    """
    checked_at = utc_now_iso()
    headers: Dict[str, str] = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    candidates = [f"{base_url}/models"]
    if not base_url.endswith("/v1"):
        candidates.append(f"{base_url}/v1/models")

    last_err = None
    last_status = None
    last_payload = None

    for url in candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            last_status = resp.status_code
            payload = safe_json(resp)
            last_payload = payload

            if resp.status_code >= 400:
                last_err = f"HTTP {resp.status_code}"
                continue

            if not isinstance(payload, dict):
                return ProbeResult(
                    reachable=True,
                    status_code=resp.status_code,
                    error="Reachable, but response is not a JSON object",
                    models=[],
                    raw_response=payload if isinstance(payload, dict) else None,
                    checked_at_utc=checked_at,
                )

            models: List[Dict[str, Any]] = []
            if isinstance(payload.get("data"), list):
                models = [m for m in payload["data"] if isinstance(m, dict)]
            elif isinstance(payload.get("models"), list):
                models = [m for m in payload["models"] if isinstance(m, dict)]
            elif isinstance(payload.get("result"), list):
                models = [m for m in payload["result"] if isinstance(m, dict)]

            if models:
                return ProbeResult(
                    reachable=True,
                    status_code=resp.status_code,
                    error=None,
                    models=models,
                    raw_response=payload,
                    checked_at_utc=checked_at,
                )

            return ProbeResult(
                reachable=True,
                status_code=resp.status_code,
                error="Reachable, but model list payload not recognized or empty",
                models=[],
                raw_response=payload,
                checked_at_utc=checked_at,
            )

        except RequestException as e:
            last_err = str(e)

    return ProbeResult(
        reachable=False,
        status_code=last_status,
        error=last_err,
        models=[],
        raw_response=last_payload,
        checked_at_utc=checked_at,
    )


# ----------------------------
# SQLite schema + upserts
# ----------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS hosts (
  host_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_name TEXT NOT NULL UNIQUE,
  ts_ip TEXT,
  user_email TEXT,
  ts_version TEXT,
  os TEXT,
  first_seen_utc TEXT,
  last_seen_utc TEXT
);

CREATE TABLE IF NOT EXISTS endpoints (
  endpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
  base_url TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  scheme TEXT,
  port INTEGER,
  api_root TEXT,
  first_seen_utc TEXT,
  last_checked_utc TEXT,
  reachable INTEGER NOT NULL DEFAULT 0,
  status_code INTEGER,
  error TEXT,
  raw_response_json TEXT
);

CREATE TABLE IF NOT EXISTS host_endpoints (
  host_id INTEGER NOT NULL,
  endpoint_id INTEGER NOT NULL,
  first_seen_utc TEXT,
  last_seen_utc TEXT,
  PRIMARY KEY (host_id, endpoint_id),
  FOREIGN KEY (host_id) REFERENCES hosts(host_id) ON DELETE CASCADE,
  FOREIGN KEY (endpoint_id) REFERENCES endpoints(endpoint_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS models (
  model_id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_key TEXT NOT NULL UNIQUE,   -- typically OpenAI-compatible "id"
  object TEXT,
  owned_by TEXT,
  created INTEGER,
  raw_json TEXT,
  first_seen_utc TEXT,
  last_seen_utc TEXT
);

CREATE TABLE IF NOT EXISTS endpoint_models (
  endpoint_id INTEGER NOT NULL,
  model_id INTEGER NOT NULL,
  first_seen_utc TEXT,
  last_seen_utc TEXT,
  PRIMARY KEY (endpoint_id, model_id),
  FOREIGN KEY (endpoint_id) REFERENCES endpoints(endpoint_id) ON DELETE CASCADE,
  FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

-- Optional history table (one row per run per endpoint)
CREATE TABLE IF NOT EXISTS endpoint_checks (
  check_id INTEGER PRIMARY KEY AUTOINCREMENT,
  endpoint_id INTEGER NOT NULL,
  checked_at_utc TEXT NOT NULL,
  reachable INTEGER NOT NULL,
  status_code INTEGER,
  error TEXT,
  model_count INTEGER NOT NULL,
  raw_response_json TEXT,
  FOREIGN KEY (endpoint_id) REFERENCES endpoints(endpoint_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_hosts_ip ON hosts(ts_ip);
CREATE INDEX IF NOT EXISTS idx_endpoints_reachable ON endpoints(reachable);
CREATE INDEX IF NOT EXISTS idx_endpoint_checks_time ON endpoint_checks(checked_at_utc);
"""


def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def upsert_host(conn: sqlite3.Connection, host: Dict[str, Any], seen_utc: str) -> int:
    conn.execute(
        """
        INSERT INTO hosts (ts_name, ts_ip, user_email, ts_version, os, first_seen_utc, last_seen_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ts_name) DO UPDATE SET
          ts_ip=excluded.ts_ip,
          user_email=excluded.user_email,
          ts_version=excluded.ts_version,
          os=excluded.os,
          last_seen_utc=excluded.last_seen_utc
        """,
        (
            host["ts_name"],
            host.get("ts_ip"),
            host.get("user_email"),
            host.get("ts_version"),
            host.get("os"),
            seen_utc,
            seen_utc,
        ),
    )
    row = conn.execute("SELECT host_id FROM hosts WHERE ts_name=?", (host["ts_name"],)).fetchone()
    return int(row["host_id"])


def upsert_endpoint(
    conn: sqlite3.Connection,
    base_url: str,
    scheme: str,
    port: int,
    api_root: str,
    probe: ProbeResult,
    seen_utc: str,
) -> int:
    raw_json = json.dumps(probe.raw_response) if probe.raw_response is not None else None

    conn.execute(
        """
        INSERT INTO endpoints (
          base_url, kind, scheme, port, api_root,
          first_seen_utc, last_checked_utc, reachable, status_code, error, raw_response_json
        )
        VALUES (?, 'lmstudio-openai-compatible', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(base_url) DO UPDATE SET
          last_checked_utc=excluded.last_checked_utc,
          reachable=excluded.reachable,
          status_code=excluded.status_code,
          error=excluded.error,
          raw_response_json=excluded.raw_response_json
        """,
        (
            base_url,
            scheme,
            port,
            api_root,
            seen_utc,
            probe.checked_at_utc,
            1 if probe.reachable else 0,
            probe.status_code,
            probe.error,
            raw_json,
        ),
    )

    row = conn.execute("SELECT endpoint_id FROM endpoints WHERE base_url=?", (base_url,)).fetchone()
    endpoint_id = int(row["endpoint_id"])

    # write history row each run
    conn.execute(
        """
        INSERT INTO endpoint_checks (endpoint_id, checked_at_utc, reachable, status_code, error, model_count, raw_response_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            endpoint_id,
            probe.checked_at_utc,
            1 if probe.reachable else 0,
            probe.status_code,
            probe.error,
            len(probe.models),
            raw_json,
        ),
    )

    return endpoint_id


def link_host_endpoint(conn: sqlite3.Connection, host_id: int, endpoint_id: int, seen_utc: str) -> None:
    conn.execute(
        """
        INSERT INTO host_endpoints (host_id, endpoint_id, first_seen_utc, last_seen_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(host_id, endpoint_id) DO UPDATE SET
          last_seen_utc=excluded.last_seen_utc
        """,
        (host_id, endpoint_id, seen_utc, seen_utc),
    )


def normalize_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in models:
        mid = m.get("id") or m.get("model") or m.get("name")
        if not mid:
            continue
        out.append(
            {
                "model_key": str(mid),
                "object": m.get("object"),
                "owned_by": m.get("owned_by"),
                "created": m.get("created"),
                "raw_json": json.dumps(m),
            }
        )
    return out


def upsert_model(conn: sqlite3.Connection, model: Dict[str, Any], seen_utc: str) -> int:
    conn.execute(
        """
        INSERT INTO models (model_key, object, owned_by, created, raw_json, first_seen_utc, last_seen_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_key) DO UPDATE SET
          object=excluded.object,
          owned_by=excluded.owned_by,
          created=excluded.created,
          raw_json=excluded.raw_json,
          last_seen_utc=excluded.last_seen_utc
        """,
        (
            model["model_key"],
            model.get("object"),
            model.get("owned_by"),
            model.get("created"),
            model.get("raw_json"),
            seen_utc,
            seen_utc,
        ),
    )
    row = conn.execute("SELECT model_id FROM models WHERE model_key=?", (model["model_key"],)).fetchone()
    return int(row["model_id"])


def link_endpoint_model(conn: sqlite3.Connection, endpoint_id: int, model_id: int, seen_utc: str) -> None:
    conn.execute(
        """
        INSERT INTO endpoint_models (endpoint_id, model_id, first_seen_utc, last_seen_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(endpoint_id, model_id) DO UPDATE SET
          last_seen_utc=excluded.last_seen_utc
        """,
        (endpoint_id, model_id, seen_utc, seen_utc),
    )


# ----------------------------
# Markdown sidecars
# ----------------------------
def write_host_sidecar(
    sidecar_dir: Path,
    host: Dict[str, Any],
    base_url: str,
    probe: ProbeResult,
) -> Path:
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    fname = f"HOST__{slugify_filename(host['ts_name'])}.md"
    path = sidecar_dir / fname

    lines = []
    lines.append(f"# Host: `{host['ts_name']}`")
    lines.append("")
    lines.append("## Identity")
    lines.append("")
    lines.append(f"- **Tailscale IP:** `{host.get('ts_ip')}`")
    lines.append(f"- **User:** `{host.get('user_email')}`")
    lines.append(f"- **Tailscale version:** `{host.get('ts_version')}`")
    lines.append(f"- **OS:** `{host.get('os')}`")
    lines.append("")
    lines.append("## LM Studio / OpenAI-compatible Endpoint")
    lines.append("")
    lines.append(f"- **Base URL:** `{base_url}`")
    lines.append(f"- **Checked (UTC):** `{probe.checked_at_utc}`")
    lines.append(f"- **Reachable:** `{probe.reachable}`")
    if probe.status_code is not None:
        lines.append(f"- **HTTP status:** `{probe.status_code}`")
    if probe.error:
        lines.append(f"- **Error:** `{probe.error}`")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    if probe.models:
        norm = normalize_models(probe.models)
        for m in sorted(norm, key=lambda x: x["model_key"]):
            lines.append(f"- `{m['model_key']}`")
    else:
        lines.append("_No models discovered (or endpoint unreachable)._")

    lines.append("")
    lines.append("## Raw Response (truncated)")
    lines.append("")
    raw = json.dumps(probe.raw_response, indent=2) if probe.raw_response is not None else ""
    if len(raw) > 8000:
        raw = raw[:8000] + "\n... (truncated)\n"
    lines.append("```json")
    lines.append(raw)
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_index_md(sidecar_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / "INDEX.md"

    lines = []
    lines.append("# LM Studio Inventory Index")
    lines.append("")
    lines.append(f"- Generated (UTC): `{utc_now_iso()}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Host | IP | Reachable | Model count | Sidecar |")
    lines.append("|---|---:|:---:|---:|---|")

    for r in rows:
        sidecar_file = f"HOST__{slugify_filename(r['ts_name'])}.md"
        reachable = "✅" if r["reachable"] else "❌"
        lines.append(f"| `{r['ts_name']}` | `{r['ts_ip']}` | {reachable} | {r['model_count']} | {sidecar_file} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- If a host is unreachable, verify LM Studio is running and bound to a non-local interface (or Tailscale).")
    lines.append("- Default probe is `http://<tailscale-ip>:1234/v1/models`.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="lmstudio_inventory.sqlite")
    ap.add_argument("--sidecar-dir", default="sidecar_md")
    ap.add_argument("--scheme", default="http", choices=["http", "https"])
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--api-root", default="/v1")
    ap.add_argument("--timeout", type=float, default=3.5)
    ap.add_argument("--api-key", default=None, help="Optional bearer token; or set LMSTUDIO_API_KEY")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("LMSTUDIO_API_KEY") or None

    db_path = args.db
    sidecar_dir = Path(args.sidecar_dir)

    conn = db_connect(db_path)
    try:
        db_init(conn)

        summary_rows: List[Dict[str, Any]] = []
        seen_utc = utc_now_iso()

        for host in RAW_DEVICES:
            base_url = build_base_url(args.scheme, host["ts_ip"], args.port, args.api_root)
            probe = fetch_models_openai_compatible(base_url=base_url, timeout_s=args.timeout, api_key=api_key)

            # DB writes (single transaction per host for integrity)
            conn.execute("BEGIN")
            try:
                host_id = upsert_host(conn, host, seen_utc=seen_utc)
                endpoint_id = upsert_endpoint(
                    conn,
                    base_url=base_url,
                    scheme=args.scheme,
                    port=args.port,
                    api_root=args.api_root,
                    probe=probe,
                    seen_utc=seen_utc,
                )
                link_host_endpoint(conn, host_id, endpoint_id, seen_utc=seen_utc)

                norm_models = normalize_models(probe.models)
                for m in norm_models:
                    mid = upsert_model(conn, m, seen_utc=seen_utc)
                    link_endpoint_model(conn, endpoint_id, mid, seen_utc=seen_utc)

                conn.commit()
            except Exception:
                conn.rollback()
                raise

            # Sidecar md
            write_host_sidecar(sidecar_dir, host, base_url, probe)

            summary_rows.append(
                {
                    "ts_name": host["ts_name"],
                    "ts_ip": host["ts_ip"],
                    "reachable": probe.reachable,
                    "model_count": len(normalize_models(probe.models)),
                }
            )

            if probe.reachable:
                print(f"[OK]   {host['ts_name']:20} {host['ts_ip']:15} models={len(probe.models)} url={base_url}")
            else:
                print(f"[FAIL] {host['ts_name']:20} {host['ts_ip']:15} err={probe.error} url={base_url}")

        write_index_md(sidecar_dir, summary_rows)
        print(f"\nSQLite: {db_path}")
        print(f"Sidecars: {sidecar_dir.resolve()}")
        print("Done.")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())


# sqlite3 lmstudio_inventory.sqlite "
# SELECT h.ts_name, h.ts_ip, e.base_url, e.reachable, e.status_code, e.error
# FROM hosts h
# JOIN host_endpoints he ON he.host_id=h.host_id
# JOIN endpoints e ON e.endpoint_id=he.endpoint_id
# ORDER BY e.reachable DESC, h.ts_name;
# "
