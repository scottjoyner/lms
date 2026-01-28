#!/usr/bin/env python3
"""
Two-phase concurrent benchmark runner for LM Studio OpenAI-compatible endpoints/models.

PHASE 1: Load probe for every (endpoint, model) pair.
PHASE 2: Full benchmark cases only for pairs that are LOAD OK.

Concurrency:
- Each endpoint runs at most ONE model at a time (per-endpoint semaphore = 1).
- Endpoints run concurrently (global ThreadPoolExecutor).

Outputs:
- SQLite: appends/creates benchmark tables in the same DB as inventory.
- JSONL stream: out_dir/run_<id>/events.jsonl
- CSV export: out_dir/run_<id>/bench_results.csv
- JSON export: out_dir/run_<id>/bench_results.json + routing_inputs.json

Run:
  python3 benchmark_lmstudio_two_phase_concurrent.py \
    --db lmstudio_inventory.sqlite \
    --workers 10 \
    --repeats 2 \
    --timeout 900 \
    --stream \
    --out-dir sidecar_bench_two_phase

Notes:
- "Load time" is approximated by a minimal completion request (max_tokens=1).
- TTFT measured via SSE streaming (if --stream).
- Tokens/sec uses usage.completion_tokens when present; otherwise a stable estimate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import queue
import re
import sqlite3
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException


# ----------------------------
# Utilities
# ----------------------------
def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slugify_filename(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def now_s() -> float:
    return time.perf_counter()


def safe_json(resp: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return resp.json()
    except Exception:
        return None


def approx_tokens_from_text(text: str) -> int:
    # stable rough estimate: ~4 chars/token
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def median(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(statistics.median(xs))


def mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    if not xs:
        return None
    return float(statistics.mean(xs))


# ----------------------------
# Benchmark Cases
# ----------------------------
@dataclass
class BenchCase:
    case_key: str
    task_type: str
    system: str
    prompt: str
    max_output_tokens: int
    temperature: float
    expected: Optional[Dict[str, Any]]
    notes: str


DEFAULT_CASES: List[BenchCase] = [
    BenchCase(
        case_key="math_exact_01",
        task_type="math",
        system="You are a precise assistant. Answer exactly and only with the final answer.",
        prompt="Compute 17 * 23. Output ONLY the number.",
        max_output_tokens=20,
        temperature=0.0,
        expected={"exact": "391"},
        notes="Exact-match arithmetic sanity check.",
    ),
    BenchCase(
        case_key="json_extract_01",
        task_type="json",
        system="You are a strict JSON generator. Output must be valid JSON with double quotes.",
        prompt=(
            "Extract a structured record from this text.\n\n"
            "Text: \"Scott drove to Winston-Salem on Jan 16, 2026 and met 3 friends at 7:30pm.\"\n\n"
            "Return JSON with keys: person, city, date, people_count, time.\n"
            "Output ONLY JSON."
        ),
        max_output_tokens=200,
        temperature=0.0,
        expected={"json_keys": ["person", "city", "date", "people_count", "time"]},
        notes="JSON validity + required keys.",
    ),
    BenchCase(
        case_key="reasoning_01",
        task_type="reasoning",
        system="You are a careful reasoning assistant. Keep it short.",
        prompt=(
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Answer with just the amount like 0.05."
        ),
        max_output_tokens=50,
        temperature=0.0,
        expected={"exact": "0.05"},
        notes="Classic reasoning trap; quick quality signal.",
    ),
    BenchCase(
        case_key="summarize_01",
        task_type="summarize",
        system="You are a summarization assistant. Be concise and include key facts.",
        prompt=(
            "Summarize the following in 3 bullets:\n"
            "LM Studio exposes an OpenAI-compatible API. We have multiple machines over Tailscale. "
            "We want to discover models per endpoint and benchmark them for speed and quality. "
            "Store results in SQLite and emit sidecar markdown reports."
        ),
        max_output_tokens=180,
        temperature=0.2,
        expected={"contains_any": ["OpenAI-compatible", "Tailscale", "benchmark", "SQLite"]},
        notes="Heuristic scoring: contains key terms and formatted bullets.",
    ),
    BenchCase(
        case_key="code_shape_01",
        task_type="code",
        system="You are a senior Python engineer. Output code only.",
        prompt=(
            "Write a Python function:\n\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    \"\"\"Return True if s is a palindrome ignoring case and non-alphanumeric.\"\"\"\n\n"
            "Return ONLY Python code (no markdown)."
        ),
        max_output_tokens=250,
        temperature=0.0,
        expected={"regex_all": [r"def\s+is_palindrome\s*\(", r"return\s+"]},
        notes="Shape-based scoring: has function + return.",
    ),
]


def score_response(case: BenchCase, text: str) -> Tuple[float, Dict[str, Any]]:
    details: Dict[str, Any] = {"case_key": case.case_key, "task_type": case.task_type}
    exp = case.expected or {}

    if case.task_type in ("math", "reasoning"):
        want = (exp.get("exact") or "").strip()
        got = (text or "").strip()
        ok = (got == want)
        details.update({"expected_exact": want, "got": got, "exact_match": ok})
        return (1.0 if ok else 0.0), details

    if case.task_type == "json":
        try:
            obj = json.loads(text)
            details["json_parse"] = True
        except Exception as e:
            details["json_parse"] = False
            details["json_error"] = str(e)
            return 0.0, details
        want_keys = exp.get("json_keys") or []
        missing = [k for k in want_keys if k not in obj]
        details["missing_keys"] = missing
        return (1.0 if not missing else max(0.2, 1.0 - (len(missing) / max(1, len(want_keys))))), details

    if case.task_type == "summarize":
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        bullet_like = sum(1 for ln in lines if ln.startswith(("-", "*")))
        contains_any = exp.get("contains_any") or []
        hit = sum(1 for w in contains_any if w.lower() in (text or "").lower())
        details.update({"bullet_lines": bullet_like, "keyword_hits": hit, "keyword_total": len(contains_any)})
        score = min(1.0, bullet_like / 3.0) * 0.6 + (hit / max(1, len(contains_any))) * 0.4
        return max(0.0, min(1.0, score)), details

    if case.task_type == "code":
        regex_all = exp.get("regex_all") or []
        missing = [pat for pat in regex_all if not re.search(pat, text or "")]
        ok = not missing
        details.update({"regex_missing": missing, "regex_ok": ok})
        return (1.0 if ok else 0.0), details

    return 0.0, {"error": "unknown_task_type"}


# ----------------------------
# OpenAI-compatible calls
# ----------------------------
@dataclass
class CompletionMetrics:
    ok: bool
    http_status: Optional[int]
    error: Optional[str]
    output_text: str
    wall_s: float
    ttft_s: Optional[float]
    tokens_per_sec: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    finish_reason: Optional[str]
    raw_json: Optional[Dict[str, Any]]


def call_chat_completions_once(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    api_key: Optional[str],
) -> CompletionMetrics:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t0 = now_s()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        wall = now_s() - t0

        if resp.status_code >= 400:
            return CompletionMetrics(
                ok=False,
                http_status=resp.status_code,
                error=f"HTTP {resp.status_code}: {(resp.text or '')[:2000]}",
                output_text="",
                wall_s=wall,
                ttft_s=None,
                tokens_per_sec=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                finish_reason=None,
                raw_json=None,
            )

        j = safe_json(resp) or {}
        text = ""
        finish_reason = None
        try:
            ch0 = (j.get("choices") or [None])[0] or {}
            finish_reason = ch0.get("finish_reason")
            msg = ch0.get("message") or {}
            text = msg.get("content") or ""
        except Exception:
            pass

        usage = j.get("usage") if isinstance(j.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        if completion_tokens is None and text:
            completion_tokens = approx_tokens_from_text(text)
        tps = (float(completion_tokens) / wall) if (completion_tokens and wall > 0) else None

        return CompletionMetrics(
            ok=True,
            http_status=resp.status_code,
            error=None,
            output_text=text,
            wall_s=wall,
            ttft_s=None,
            tokens_per_sec=tps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
            raw_json=j if isinstance(j, dict) else None,
        )

    except RequestException as e:
        return CompletionMetrics(
            ok=False,
            http_status=None,
            error=str(e),
            output_text="",
            wall_s=now_s() - t0,
            ttft_s=None,
            tokens_per_sec=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            finish_reason=None,
            raw_json=None,
        )


def call_chat_completions_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    api_key: Optional[str],
) -> CompletionMetrics:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }

    t0 = now_s()
    ttft = None
    output_parts: List[str] = []
    last_chunk = None
    finish_reason = None
    prompt_tokens = completion_tokens = total_tokens = None

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout_s) as resp:
            if resp.status_code >= 400:
                return CompletionMetrics(
                    ok=False,
                    http_status=resp.status_code,
                    error=f"HTTP {resp.status_code}: {(resp.text or '')[:2000]}",
                    output_text="",
                    wall_s=now_s() - t0,
                    ttft_s=None,
                    tokens_per_sec=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    finish_reason=None,
                    raw_json=None,
                )

            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue

                last_chunk = chunk
                choice0 = (chunk.get("choices") or [None])[0] or {}
                delta = choice0.get("delta") or {}

                if ttft is None and delta.get("content") is not None:
                    ttft = now_s() - t0

                if delta.get("content"):
                    output_parts.append(delta["content"])

                if choice0.get("finish_reason"):
                    finish_reason = choice0.get("finish_reason")

                if isinstance(chunk.get("usage"), dict):
                    u = chunk["usage"]
                    prompt_tokens = u.get("prompt_tokens")
                    completion_tokens = u.get("completion_tokens")
                    total_tokens = u.get("total_tokens")

            out = "".join(output_parts)
            wall = now_s() - t0

            tps = None
            if completion_tokens is not None and wall > 0:
                tps = float(completion_tokens) / wall
            elif out:
                est = approx_tokens_from_text(out)
                completion_tokens = est
                if wall > 0:
                    tps = est / wall

            return CompletionMetrics(
                ok=True,
                http_status=resp.status_code,
                error=None,
                output_text=out,
                wall_s=wall,
                ttft_s=ttft,
                tokens_per_sec=tps,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                raw_json=last_chunk if isinstance(last_chunk, dict) else None,
            )

    except RequestException as e:
        return CompletionMetrics(
            ok=False,
            http_status=None,
            error=str(e),
            output_text="",
            wall_s=now_s() - t0,
            ttft_s=None,
            tokens_per_sec=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            finish_reason=None,
            raw_json=None,
        )


def probe_model_load_time(base_url: str, model: str, timeout_s: float, api_key: Optional[str]) -> CompletionMetrics:
    return call_chat_completions_once(
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": "You are a minimal assistant."},
            {"role": "user", "content": "Say OK."},
        ],
        max_tokens=1,
        temperature=0.0,
        timeout_s=timeout_s,
        api_key=api_key,
    )


# ----------------------------
# SQLite schema (bench tables)
# ----------------------------
BENCH_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS bench_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at_utc TEXT NOT NULL,
  finished_at_utc TEXT,
  notes TEXT,
  config_json TEXT
);

CREATE TABLE IF NOT EXISTS bench_cases (
  case_id INTEGER PRIMARY KEY AUTOINCREMENT,
  case_key TEXT NOT NULL UNIQUE,
  task_type TEXT NOT NULL,
  system TEXT NOT NULL,
  prompt TEXT NOT NULL,
  max_output_tokens INTEGER NOT NULL,
  temperature REAL NOT NULL,
  expected_json TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS bench_results (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL,
  endpoint_id INTEGER NOT NULL,
  model_id INTEGER NOT NULL,
  case_id INTEGER NOT NULL,

  repeat_index INTEGER NOT NULL,
  phase TEXT NOT NULL,   -- load | warmup | run

  ok INTEGER NOT NULL,
  http_status INTEGER,
  error TEXT,

  wall_s REAL,
  ttft_s REAL,
  tokens_per_sec REAL,

  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  total_tokens INTEGER,

  finish_reason TEXT,

  auto_quality REAL,
  auto_quality_details_json TEXT,

  output_text TEXT,
  raw_json TEXT,

  created_at_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bench_results_run ON bench_results(run_id);
CREATE INDEX IF NOT EXISTS idx_bench_results_ep_model ON bench_results(endpoint_id, model_id);
"""


def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(BENCH_SCHEMA_SQL)
    conn.commit()


def upsert_cases(conn: sqlite3.Connection, cases: List[BenchCase]) -> None:
    for c in cases:
        conn.execute(
            """
            INSERT INTO bench_cases (case_key, task_type, system, prompt, max_output_tokens, temperature, expected_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_key) DO UPDATE SET
              task_type=excluded.task_type,
              system=excluded.system,
              prompt=excluded.prompt,
              max_output_tokens=excluded.max_output_tokens,
              temperature=excluded.temperature,
              expected_json=excluded.expected_json,
              notes=excluded.notes
            """,
            (
                c.case_key, c.task_type, c.system, c.prompt,
                c.max_output_tokens, c.temperature,
                json.dumps(c.expected) if c.expected is not None else None,
                c.notes,
            ),
        )
    conn.commit()


def create_run(conn: sqlite3.Connection, notes: str, config: Dict[str, Any]) -> int:
    conn.execute(
        "INSERT INTO bench_runs (started_at_utc, notes, config_json) VALUES (?, ?, ?)",
        (utc_now_iso(), notes, json.dumps(config)),
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])


def finish_run(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute("UPDATE bench_runs SET finished_at_utc=? WHERE run_id=?", (utc_now_iso(), run_id))
    conn.commit()


def get_case_id(conn: sqlite3.Connection, case_key: str) -> int:
    row = conn.execute("SELECT case_id FROM bench_cases WHERE case_key=?", (case_key,)).fetchone()
    return int(row["case_id"])


def load_endpoint_model_groups(conn: sqlite3.Connection, only_reachable: bool, limit_models_per_endpoint: int) -> List[Dict[str, Any]]:
    where = "WHERE e.kind='lmstudio-openai-compatible'"
    if only_reachable:
        where += " AND e.reachable=1"

    q = f"""
    SELECT
      h.ts_name AS host_name,
      h.ts_ip AS host_ip,
      e.endpoint_id,
      e.base_url,
      m.model_id,
      m.model_key
    FROM endpoints e
    JOIN endpoint_models em ON em.endpoint_id=e.endpoint_id
    JOIN models m ON m.model_id=em.model_id
    LEFT JOIN host_endpoints he ON he.endpoint_id=e.endpoint_id
    LEFT JOIN hosts h ON h.host_id=he.host_id
    {where}
    GROUP BY e.endpoint_id, m.model_id
    ORDER BY e.endpoint_id, m.model_key
    """
    rows = [dict(r) for r in conn.execute(q).fetchall()]

    if limit_models_per_endpoint and limit_models_per_endpoint > 0:
        out: List[Dict[str, Any]] = []
        counts: Dict[int, int] = {}
        for r in rows:
            eid = r["endpoint_id"]
            counts.setdefault(eid, 0)
            if counts[eid] < limit_models_per_endpoint:
                out.append(r)
                counts[eid] += 1
        return out

    return rows


# ----------------------------
# Per-endpoint limiter
# ----------------------------
class EndpointLimiter:
    def __init__(self, per_endpoint_limit: int = 1):
        self.per_endpoint_limit = max(1, int(per_endpoint_limit))
        self._sems: Dict[int, threading.Semaphore] = {}
        self._guard = threading.Lock()

    def sem(self, endpoint_id: int) -> threading.Semaphore:
        with self._guard:
            if endpoint_id not in self._sems:
                self._sems[endpoint_id] = threading.Semaphore(self.per_endpoint_limit)
            return self._sems[endpoint_id]


# ----------------------------
# DB writer + JSONL logger
# ----------------------------
@dataclass
class DBWriteItem:
    kind: str  # "result" | "event"
    payload: Dict[str, Any]


def db_writer_loop(
    db_path: str,
    q: "queue.Queue[Optional[DBWriteItem]]",
    stop_event: threading.Event,
    jsonl_path: Path,
) -> None:
    conn = db_connect(db_path)
    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        f = jsonl_path.open("a", encoding="utf-8")
        try:
            while not stop_event.is_set():
                item = q.get()
                if item is None:
                    break

                if item.kind == "result":
                    p = item.payload
                    # --- SAFE INSERT: columns and values stay in lock-step ---
                    cols = [
                        "run_id", "endpoint_id", "model_id", "case_id",
                        "repeat_index", "phase",
                        "ok", "http_status", "error",
                        "wall_s", "ttft_s", "tokens_per_sec",
                        "prompt_tokens", "completion_tokens", "total_tokens",
                        "finish_reason",
                        "auto_quality", "auto_quality_details_json",
                        "output_text", "raw_json",
                        "created_at_utc",
                    ]

                    vals = [
                        p["run_id"], p["endpoint_id"], p["model_id"], p["case_id"],
                        p["repeat_index"], p["phase"],
                        1 if p["ok"] else 0, p.get("http_status"), p.get("error"),
                        p.get("wall_s"), p.get("ttft_s"), p.get("tokens_per_sec"),
                        p.get("prompt_tokens"), p.get("completion_tokens"), p.get("total_tokens"),
                        p.get("finish_reason"),
                        p.get("auto_quality"),
                        json.dumps(p.get("auto_details")) if p.get("auto_details") is not None else None,
                        (p.get("output_text") or "")[:20000],
                        json.dumps(p.get("raw_json"))[:20000] if p.get("raw_json") is not None else None,
                        p["created_at_utc"],
                    ]

                    assert len(cols) == len(vals), f"bench_results insert mismatch: {len(cols)} cols vs {len(vals)} vals"

                    sql = f"INSERT INTO bench_results ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"
                    conn.execute(sql, vals)
                    conn.commit()

                       

                    # also log as JSONL event
                    f.write(json.dumps({"type": "bench_result", **p}) + "\n")
                    f.flush()

                elif item.kind == "event":
                    f.write(json.dumps({"type": "event", **item.payload}) + "\n")
                    f.flush()

                q.task_done()
        finally:
            f.close()
    finally:
        conn.close()


# ----------------------------
# Workers
# ----------------------------
def phase1_load_worker(
    group: Dict[str, Any],
    run_id: int,
    load_case_id: int,
    api_key: Optional[str],
    timeout_s: float,
    writer_q: "queue.Queue[Optional[DBWriteItem]]",
    limiter: EndpointLimiter,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (load_ok, group) and writes a 'load' bench_result row.
    """
    endpoint_id = group["endpoint_id"]
    model_id = group["model_id"]
    base_url = group["base_url"]
    model_key = group["model_key"]

    sem = limiter.sem(endpoint_id)
    with sem:
        met = probe_model_load_time(base_url, model_key, timeout_s=timeout_s, api_key=api_key)

    writer_q.put(DBWriteItem(kind="result", payload={
        "run_id": run_id,
        "endpoint_id": endpoint_id,
        "model_id": model_id,
        "case_id": load_case_id,
        "repeat_index": 0,
        "phase": "load",
        "ok": met.ok,
        "http_status": met.http_status,
        "error": met.error,
        "wall_s": met.wall_s,
        "ttft_s": None,
        "tokens_per_sec": met.tokens_per_sec,
        "prompt_tokens": met.prompt_tokens,
        "completion_tokens": met.completion_tokens,
        "total_tokens": met.total_tokens,
        "finish_reason": met.finish_reason,
        "auto_quality": None,
        "auto_details": {"note": "load probe"},
        "output_text": "",
        "raw_json": met.raw_json,
        "created_at_utc": utc_now_iso(),
    }))

    return met.ok, group


def phase2_full_worker(
    group: Dict[str, Any],
    run_id: int,
    case_ids: Dict[str, int],
    cases: List[BenchCase],
    api_key: Optional[str],
    timeout_s: float,
    repeats: int,
    stream: bool,
    writer_q: "queue.Queue[Optional[DBWriteItem]]",
    limiter: EndpointLimiter,
) -> None:
    endpoint_id = group["endpoint_id"]
    model_id = group["model_id"]
    base_url = group["base_url"]
    model_key = group["model_key"]

    sem = limiter.sem(endpoint_id)

    # Warmup (still one-at-a-time per endpoint)
    with sem:
        warm = call_chat_completions_once(
            base_url=base_url,
            model=model_key,
            messages=[{"role": "system", "content": "You are a warmup."}, {"role": "user", "content": "Reply OK."}],
            max_tokens=16,
            temperature=0.0,
            timeout_s=timeout_s,
            api_key=api_key,
        )

    writer_q.put(DBWriteItem(kind="result", payload={
        "run_id": run_id,
        "endpoint_id": endpoint_id,
        "model_id": model_id,
        "case_id": case_ids[cases[0].case_key],
        "repeat_index": 0,
        "phase": "warmup",
        "ok": warm.ok,
        "http_status": warm.http_status,
        "error": warm.error,
        "wall_s": warm.wall_s,
        "ttft_s": None,
        "tokens_per_sec": warm.tokens_per_sec,
        "prompt_tokens": warm.prompt_tokens,
        "completion_tokens": warm.completion_tokens,
        "total_tokens": warm.total_tokens,
        "finish_reason": warm.finish_reason,
        "auto_quality": None,
        "auto_details": {"note": "warmup"},
        "output_text": warm.output_text,
        "raw_json": warm.raw_json,
        "created_at_utc": utc_now_iso(),
    }))

    # Full cases
    for case in cases:
        messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]
        for rep in range(repeats):
            with sem:
                if stream:
                    met = call_chat_completions_stream(
                        base_url=base_url,
                        model=model_key,
                        messages=messages,
                        max_tokens=case.max_output_tokens,
                        temperature=case.temperature,
                        timeout_s=timeout_s,
                        api_key=api_key,
                    )
                    # if usage missing, do a non-stream call for usage merge (still protected by sem)
                    if met.ok and met.completion_tokens is None:
                        met2 = call_chat_completions_once(
                            base_url=base_url,
                            model=model_key,
                            messages=messages,
                            max_tokens=case.max_output_tokens,
                            temperature=case.temperature,
                            timeout_s=timeout_s,
                            api_key=api_key,
                        )
                        if met2.ok and met2.completion_tokens is not None:
                            met.completion_tokens = met2.completion_tokens
                            met.prompt_tokens = met2.prompt_tokens
                            met.total_tokens = met2.total_tokens
                            if met.wall_s and met.wall_s > 0:
                                met.tokens_per_sec = float(met.completion_tokens) / met.wall_s
                else:
                    met = call_chat_completions_once(
                        base_url=base_url,
                        model=model_key,
                        messages=messages,
                        max_tokens=case.max_output_tokens,
                        temperature=case.temperature,
                        timeout_s=timeout_s,
                        api_key=api_key,
                    )

            auto_q = None
            auto_details = None
            if met.ok:
                auto_q, auto_details = score_response(case, met.output_text)

            writer_q.put(DBWriteItem(kind="result", payload={
                "run_id": run_id,
                "endpoint_id": endpoint_id,
                "model_id": model_id,
                "case_id": case_ids[case.case_key],
                "repeat_index": rep,
                "phase": "run",
                "ok": met.ok,
                "http_status": met.http_status,
                "error": met.error,
                "wall_s": met.wall_s,
                "ttft_s": met.ttft_s,
                "tokens_per_sec": met.tokens_per_sec,
                "prompt_tokens": met.prompt_tokens,
                "completion_tokens": met.completion_tokens,
                "total_tokens": met.total_tokens,
                "finish_reason": met.finish_reason,
                "auto_quality": auto_q,
                "auto_details": auto_details,
                "output_text": met.output_text,
                "raw_json": met.raw_json,
                "created_at_utc": utc_now_iso(),
            }))


# ----------------------------
# Exporters (CSV + JSON)
# ----------------------------
def export_results(conn: sqlite3.Connection, run_id: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Flattened rows
    q = """
    SELECT
      br.run_id,
      br.created_at_utc,
      br.phase,
      br.repeat_index,
      br.ok,
      br.http_status,
      br.error,
      br.wall_s,
      br.ttft_s,
      br.tokens_per_sec,
      br.prompt_tokens,
      br.completion_tokens,
      br.total_tokens,
      br.finish_reason,
      br.auto_quality,
      br.auto_quality_details_json,

      e.endpoint_id,
      e.base_url,

      m.model_id,
      m.model_key,

      bc.case_id,
      bc.case_key,
      bc.task_type
    FROM bench_results br
    JOIN endpoints e ON e.endpoint_id = br.endpoint_id
    JOIN models m ON m.model_id = br.model_id
    JOIN bench_cases bc ON bc.case_id = br.case_id
    WHERE br.run_id=?
    ORDER BY e.endpoint_id, m.model_key, br.phase, bc.case_key, br.repeat_index
    """
    rows = [dict(r) for r in conn.execute(q, (run_id,)).fetchall()]

    # CSV
    csv_path = out_dir / "bench_results.csv"
    if rows:
        cols = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        csv_path.write_text("", encoding="utf-8")

    # JSON (hierarchical by endpoint -> model -> phases/cases)
    by_ep: Dict[str, Any] = {}
    for r in rows:
        ep = r["base_url"]
        model = r["model_key"]
        by_ep.setdefault(ep, {"endpoint_id": r["endpoint_id"], "base_url": ep, "models": {}})
        by_ep[ep]["models"].setdefault(model, {"model_id": r["model_id"], "model_key": model, "results": []})
        by_ep[ep]["models"][model]["results"].append(r)

    json_path = out_dir / "bench_results.json"
    json_path.write_text(json.dumps({
        "generated_at_utc": utc_now_iso(),
        "run_id": run_id,
        "endpoints": by_ep,
    }, indent=2), encoding="utf-8")

    # Routing inputs (aggregated metrics per endpoint+model)
    agg = aggregate_for_routing(rows)
    routing_inputs_path = out_dir / "routing_inputs.json"
    routing_inputs_path.write_text(json.dumps({
        "generated_at_utc": utc_now_iso(),
        "run_id": run_id,
        "endpoint_model_metrics": agg,
        "notes": "Use this as input to recommendation/routing scripts.",
    }, indent=2), encoding="utf-8")


def aggregate_for_routing(flat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return list of aggregates per (endpoint, model):
      - load_s (median of load wall_s)
      - ok_rate (on run rows)
      - ttft_med, tps_med (on ok run rows)
      - auto_quality_mean (on ok run rows)
    """
    # group rows
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in flat_rows:
        key = (r["base_url"], r["model_key"])
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (base_url, model_key), rs in groups.items():
        load_ws = [r["wall_s"] for r in rs if r["phase"] == "load" and r.get("wall_s") is not None]
        load_s = median([float(x) for x in load_ws]) if load_ws else None

        run_rows = [r for r in rs if r["phase"] == "run"]
        ok_rate = (sum(1 for r in run_rows if int(r["ok"]) == 1) / len(run_rows)) if run_rows else 0.0

        ok_runs = [r for r in run_rows if int(r["ok"]) == 1]
        ttft_med = median([float(r["ttft_s"]) for r in ok_runs if r.get("ttft_s") is not None])
        tps_med = median([float(r["tokens_per_sec"]) for r in ok_runs if r.get("tokens_per_sec") is not None])
        auto_q_mean = mean([float(r["auto_quality"]) for r in ok_runs if r.get("auto_quality") is not None])

        out.append({
            "base_url": base_url,
            "model_key": model_key,
            "load_s_med": load_s,
            "ok_rate": ok_rate,
            "ttft_s_med": ttft_med,
            "tokens_per_sec_med": tps_med,
            "auto_quality_mean": auto_q_mean,
        })

    # stable order
    out.sort(key=lambda x: (x["base_url"], x["model_key"]))
    return out


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite inventory DB; benchmark tables will be added here")
    ap.add_argument("--only-reachable-endpoints", action="store_true", default=True)
    ap.add_argument("--limit-models-per-endpoint", type=int, default=0, help="0 = no limit")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--stream", action="store_true", default=True)

    # global concurrency across endpoints; per-endpoint is fixed to 1 as requested
    ap.add_argument("--workers", type=int, default=10)

    ap.add_argument("--out-dir", default="sidecar_bench_two_phase")
    ap.add_argument("--notes", default="")
    ap.add_argument("--api-key", default=None, help="Bearer token if endpoints require it (or set LMSTUDIO_API_KEY)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("LMSTUDIO_API_KEY") or None

    conn = db_connect(args.db)
    try:
        ensure_schema(conn)
        upsert_cases(conn, DEFAULT_CASES)
        case_ids = {c.case_key: get_case_id(conn, c.case_key) for c in DEFAULT_CASES}
        load_case_id = case_ids[DEFAULT_CASES[0].case_key]

        groups = load_endpoint_model_groups(
            conn,
            only_reachable=args.only_reachable_endpoints,
            limit_models_per_endpoint=args.limit_models_per_endpoint,
        )
        if not groups:
            print("No endpoint/model groups found. Run inventory first.", flush=True)
            return 2

        # Run record
        config = {
            "two_phase": True,
            "timeout": args.timeout,
            "repeats": args.repeats,
            "stream": args.stream,
            "workers": args.workers,
            "per_endpoint": 1,
            "limit_models_per_endpoint": args.limit_models_per_endpoint,
        }
        run_id = create_run(conn, notes=args.notes, config=config)
        started_at = conn.execute("SELECT started_at_utc FROM bench_runs WHERE run_id=?", (run_id,)).fetchone()["started_at_utc"]

        run_dir = Path(args.out_dir) / f"RUN__{run_id}__{slugify_filename(started_at)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSONL event log
        jsonl_path = run_dir / "events.jsonl"

        # DB writer thread
        writer_q: "queue.Queue[Optional[DBWriteItem]]" = queue.Queue(maxsize=20000)
        stop_event = threading.Event()
        writer_thread = threading.Thread(
            target=db_writer_loop,
            args=(args.db, writer_q, stop_event, jsonl_path),
            daemon=True,
        )
        writer_thread.start()

        limiter = EndpointLimiter(per_endpoint_limit=1)

        # -------- Phase 1: Load probe all pairs --------
        writer_q.put(DBWriteItem(kind="event", payload={
            "run_id": run_id, "phase": "phase1_load", "at_utc": utc_now_iso(),
            "groups": len(groups),
        }))

        load_ok_groups: List[Dict[str, Any]] = []
        load_failures = 0

        # global workers: high is fine; limiter prevents per-endpoint parallelism
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [
                ex.submit(
                    phase1_load_worker,
                    g, run_id, load_case_id, api_key, args.timeout, writer_q, limiter
                )
                for g in groups
            ]
            for fut in as_completed(futs):
                try:
                    ok, g = fut.result()
                    if ok:
                        load_ok_groups.append(g)
                    else:
                        load_failures += 1
                except Exception as e:
                    load_failures += 1
                    print(f"[PHASE1 WORKER FAIL] {e}", flush=True)

        writer_q.put(DBWriteItem(kind="event", payload={
            "run_id": run_id, "phase": "phase1_done", "at_utc": utc_now_iso(),
            "load_ok": len(load_ok_groups),
            "load_fail": load_failures,
        }))

        print(f"\nPhase 1 done: load_ok={len(load_ok_groups)} load_fail={load_failures}", flush=True)

        # -------- Phase 2: Full benchmark only for load-ok --------
        writer_q.put(DBWriteItem(kind="event", payload={
            "run_id": run_id, "phase": "phase2_full", "at_utc": utc_now_iso(),
            "groups": len(load_ok_groups),
        }))

        phase2_failures = 0
        if load_ok_groups:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = [
                    ex.submit(
                        phase2_full_worker,
                        g, run_id, case_ids, DEFAULT_CASES,
                        api_key, args.timeout, args.repeats, args.stream,
                        writer_q, limiter
                    )
                    for g in load_ok_groups
                ]
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        phase2_failures += 1
                        print(f"[PHASE2 WORKER FAIL] {e}", flush=True)

        writer_q.put(DBWriteItem(kind="event", payload={
            "run_id": run_id, "phase": "phase2_done", "at_utc": utc_now_iso(),
            "failures": phase2_failures,
        }))

        # flush writer
        writer_q.join()
        stop_event.set()
        writer_q.put(None)
        writer_thread.join(timeout=15)

        finish_run(conn, run_id)

        # exports (CSV + JSON) from DB
        export_results(conn, run_id, run_dir)

        print(f"\nRun complete: {run_id}")
        print(f"Artifacts: {run_dir.resolve()}")
        print(f"Phase1 load failures: {load_failures}")
        print(f"Phase2 worker failures: {phase2_failures}")
        return 0 if (load_failures == 0 and phase2_failures == 0) else 1

    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

# python3 benchmark_lmstudio_two_phase_concurrent.py
#   --db lmstudio_inventory.sqlite
#   --workers 6
#   --repeats 1
#   --timeout 600
#   --stream
#   --out-dir sidecar_bench_two_phase
