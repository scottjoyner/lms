#!/usr/bin/env python3
"""
Concurrent benchmark runner for LM Studio OpenAI-compatible endpoints/models.

- Uses a ThreadPoolExecutor to test multiple endpoint+model groups concurrently.
- Uses a single DB writer thread to keep SQLite safe.
- Optional per-endpoint concurrency limit (to avoid thrashing a single box).

Requires inventory+benchmark schema present (same as prior scripts).
"""

from __future__ import annotations

import argparse
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


# ---------- Utilities ----------
def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slugify_filename(s: str) -> str:
    s = s.strip().lower()
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
    if not text:
        return 0
    return max(1, int(len(text) / 4))


# ---------- Benchmark cases ----------
@dataclass
class BenchCase:
    case_key: str
    task_type: str
    prompt: str
    system: str
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


# ---------- HTTP call helpers ----------
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
            wall_start = now_s()
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
                if not raw_line:
                    continue
                if not raw_line.startswith("data:"):
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


# ---------- SQLite schema additions ----------
BENCH_SCHEMA_SQL = """
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
  phase TEXT NOT NULL,  -- load | warmup | run

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

INVENTORY_REQUIRED_SQL = """
-- these must exist from your inventory script
-- endpoints(endpoint_id, base_url, kind, reachable, ...)
-- models(model_id, model_key, ...)
-- endpoint_models(endpoint_id, model_id, ...)
-- host_endpoints(endpoint_id, host_id)
-- hosts(host_id, ts_name, ts_ip)
"""


# ---------- DB helpers ----------
def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
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
                c.case_key,
                c.task_type,
                c.system,
                c.prompt,
                c.max_output_tokens,
                c.temperature,
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
    return int(conn.execute("SELECT case_id FROM bench_cases WHERE case_key=?", (case_key,)).fetchone()["case_id"])


def load_endpoint_model_groups(conn: sqlite3.Connection, only_reachable: bool) -> List[Dict[str, Any]]:
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
    return [dict(r) for r in conn.execute(q).fetchall()]


# ---------- Sidecar writer (simple) ----------
def write_model_sidecar(run_dir: Path, host: str, base_url: str, model: str, rows: List[Dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / f"MODEL__{slugify_filename(host)}__{slugify_filename(model)}.md"
    lines = []
    lines.append(f"# `{model}`")
    lines.append("")
    lines.append(f"- Host: `{host}`")
    lines.append(f"- Endpoint: `{base_url}`")
    lines.append("")
    lines.append("| Phase | Case | Rep | OK | Wall s | TTFT s | TPS | AutoQ | Error |")
    lines.append("|---|---|---:|:---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| `{r['phase']}` | `{r.get('case_key','')}` | {r.get('repeat_index',0)} | "
            f"{'✅' if r['ok'] else '❌'} | {fmt(r.get('wall_s'))} | {fmt(r.get('ttft_s'))} | "
            f"{fmt(r.get('tokens_per_sec'))} | {fmt(r.get('auto_quality'))} | {(r.get('error') or '')[:80]} |"
        )
    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


# ---------- DB writer thread ----------
@dataclass
class DBWriteItem:
    kind: str  # "result" | "sidecar"
    payload: Dict[str, Any]


def db_writer_loop(
    db_path: str,
    q: "queue.Queue[Optional[DBWriteItem]]",
    stop_event: threading.Event,
    sidecar_root: Path,
) -> None:
    conn = db_connect(db_path)
    try:
        while not stop_event.is_set():
            item = q.get()
            if item is None:
                break
            if item.kind == "result":
                p = item.payload
                conn.execute(
                    """
                    INSERT INTO bench_results (
                      run_id, endpoint_id, model_id, case_id,
                      repeat_index, phase,
                      ok, http_status, error,
                      wall_s, ttft_s, tokens_per_sec,
                      prompt_tokens, completion_tokens, total_tokens,
                      finish_reason,
                      auto_quality, auto_quality_details_json,
                      output_text, raw_json,
                      created_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
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
                    ),
                )
                conn.commit()

            elif item.kind == "sidecar":
                p = item.payload
                write_model_sidecar(
                    run_dir=sidecar_root,
                    host=p["host_name"],
                    base_url=p["base_url"],
                    model=p["model_key"],
                    rows=p["rows"],
                )
            else:
                # unknown
                pass
            q.task_done()
    finally:
        conn.close()


# ---------- Concurrency controls ----------
class EndpointLimiter:
    """
    Per-endpoint semaphore to avoid hammering one machine with multiple models at once.
    """
    def __init__(self, per_endpoint_limit: int):
        self.per_endpoint_limit = max(1, int(per_endpoint_limit))
        self._locks: Dict[int, threading.Semaphore] = {}
        self._guard = threading.Lock()

    def sem(self, endpoint_id: int) -> threading.Semaphore:
        with self._guard:
            if endpoint_id not in self._locks:
                self._locks[endpoint_id] = threading.Semaphore(self.per_endpoint_limit)
            return self._locks[endpoint_id]


# ---------- Worker (one endpoint+model group) ----------
def benchmark_group(
    group: Dict[str, Any],
    run_id: int,
    cases: List[BenchCase],
    case_ids: Dict[str, int],
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
    host_name = group.get("host_name") or f"endpoint-{endpoint_id}"

    sem = limiter.sem(endpoint_id)
    with sem:
        rows_for_md: List[Dict[str, Any]] = []

        # Load probe
        load_m = probe_model_load_time(base_url, model_key, timeout_s=timeout_s, api_key=api_key)
        writer_q.put(DBWriteItem(kind="result", payload={
            "run_id": run_id, "endpoint_id": endpoint_id, "model_id": model_id, "case_id": case_ids[cases[0].case_key],
            "repeat_index": 0, "phase": "load",
            "ok": load_m.ok, "http_status": load_m.http_status, "error": load_m.error,
            "wall_s": load_m.wall_s, "ttft_s": None, "tokens_per_sec": load_m.tokens_per_sec,
            "prompt_tokens": load_m.prompt_tokens, "completion_tokens": load_m.completion_tokens, "total_tokens": load_m.total_tokens,
            "finish_reason": load_m.finish_reason,
            "auto_quality": None, "auto_details": {"note": "load probe"},
            "output_text": "", "raw_json": load_m.raw_json,
            "created_at_utc": utc_now_iso(),
        }))

        rows_for_md.append({
            "phase": "load", "case_key": cases[0].case_key, "repeat_index": 0,
            "ok": load_m.ok, "wall_s": load_m.wall_s, "ttft_s": None, "tokens_per_sec": load_m.tokens_per_sec,
            "auto_quality": None, "error": load_m.error,
        })

        # Warmup (optional-ish; keep it always-on since it helps stabilize timings)
        warm_m = call_chat_completions_once(
            base_url, model_key,
            [{"role": "system", "content": "You are a warmup."}, {"role": "user", "content": "Reply OK."}],
            max_tokens=16, temperature=0.0, timeout_s=timeout_s, api_key=api_key
        )
        writer_q.put(DBWriteItem(kind="result", payload={
            "run_id": run_id, "endpoint_id": endpoint_id, "model_id": model_id, "case_id": case_ids[cases[0].case_key],
            "repeat_index": 0, "phase": "warmup",
            "ok": warm_m.ok, "http_status": warm_m.http_status, "error": warm_m.error,
            "wall_s": warm_m.wall_s, "ttft_s": None, "tokens_per_sec": warm_m.tokens_per_sec,
            "prompt_tokens": warm_m.prompt_tokens, "completion_tokens": warm_m.completion_tokens, "total_tokens": warm_m.total_tokens,
            "finish_reason": warm_m.finish_reason,
            "auto_quality": None, "auto_details": {"note": "warmup"},
            "output_text": warm_m.output_text, "raw_json": warm_m.raw_json,
            "created_at_utc": utc_now_iso(),
        }))

        rows_for_md.append({
            "phase": "warmup", "case_key": cases[0].case_key, "repeat_index": 0,
            "ok": warm_m.ok, "wall_s": warm_m.wall_s, "ttft_s": None, "tokens_per_sec": warm_m.tokens_per_sec,
            "auto_quality": None, "error": warm_m.error,
        })

        # Run benchmark cases
        for case in cases:
            messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]
            for rep in range(repeats):
                if stream:
                    met = call_chat_completions_stream(
                        base_url, model_key, messages,
                        max_tokens=case.max_output_tokens,
                        temperature=case.temperature,
                        timeout_s=timeout_s,
                        api_key=api_key
                    )
                    # if usage missing, do one non-stream for usage only (rare but happens)
                    if met.ok and met.completion_tokens is None:
                        met2 = call_chat_completions_once(
                            base_url, model_key, messages,
                            max_tokens=case.max_output_tokens,
                            temperature=case.temperature,
                            timeout_s=timeout_s,
                            api_key=api_key
                        )
                        if met2.ok and met2.completion_tokens is not None:
                            met.completion_tokens = met2.completion_tokens
                            met.prompt_tokens = met2.prompt_tokens
                            met.total_tokens = met2.total_tokens
                            met.tokens_per_sec = (float(met.completion_tokens) / met.wall_s) if met.wall_s > 0 else met.tokens_per_sec
                else:
                    met = call_chat_completions_once(
                        base_url, model_key, messages,
                        max_tokens=case.max_output_tokens,
                        temperature=case.temperature,
                        timeout_s=timeout_s,
                        api_key=api_key
                    )

                auto_q = None
                auto_details = None
                if met.ok:
                    auto_q, auto_details = score_response(case, met.output_text)

                writer_q.put(DBWriteItem(kind="result", payload={
                    "run_id": run_id, "endpoint_id": endpoint_id, "model_id": model_id, "case_id": case_ids[case.case_key],
                    "repeat_index": rep, "phase": "run",
                    "ok": met.ok, "http_status": met.http_status, "error": met.error,
                    "wall_s": met.wall_s, "ttft_s": met.ttft_s, "tokens_per_sec": met.tokens_per_sec,
                    "prompt_tokens": met.prompt_tokens, "completion_tokens": met.completion_tokens, "total_tokens": met.total_tokens,
                    "finish_reason": met.finish_reason,
                    "auto_quality": auto_q, "auto_details": auto_details,
                    "output_text": met.output_text, "raw_json": met.raw_json,
                    "created_at_utc": utc_now_iso(),
                }))

                rows_for_md.append({
                    "phase": "run", "case_key": case.case_key, "repeat_index": rep,
                    "ok": met.ok, "wall_s": met.wall_s, "ttft_s": met.ttft_s, "tokens_per_sec": met.tokens_per_sec,
                    "auto_quality": auto_q, "error": met.error,
                    "output_text": met.output_text,
                })

        # Sidecar write request
        writer_q.put(DBWriteItem(kind="sidecar", payload={
            "host_name": host_name,
            "base_url": base_url,
            "model_key": model_key,
            "rows": rows_for_md,
        }))


# ---------- Main ----------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLite DB containing inventory tables; will append benchmark tables")
    ap.add_argument("--only-reachable-endpoints", action="store_true", default=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--stream", action="store_true", default=True)

    # Concurrency controls:
    ap.add_argument("--workers", type=int, default=4, help="Global concurrency across endpoint+model groups")
    ap.add_argument("--per-endpoint", type=int, default=1, help="Max concurrent groups per endpoint (protects a single machine)")

    ap.add_argument("--sidecar-dir", default="sidecar_bench_concurrent")
    ap.add_argument("--notes", default="")

    ap.add_argument("--api-key", default=None, help="Bearer token if endpoints require it (or set LMSTUDIO_API_KEY)")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("LMSTUDIO_API_KEY") or None

    conn = db_connect(args.db)
    try:
        ensure_schema(conn)
        upsert_cases(conn, DEFAULT_CASES)
        case_ids = {c.case_key: get_case_id(conn, c.case_key) for c in DEFAULT_CASES}

        config = {
            "timeout": args.timeout,
            "repeats": args.repeats,
            "stream": args.stream,
            "workers": args.workers,
            "per_endpoint": args.per_endpoint,
        }
        run_id = create_run(conn, notes=args.notes, config=config)
        started_at = conn.execute("SELECT started_at_utc FROM bench_runs WHERE run_id=?", (run_id,)).fetchone()["started_at_utc"]

        groups = load_endpoint_model_groups(conn, only_reachable=args.only_reachable_endpoints)
        if not groups:
            print("No endpoint/model groups found. Run inventory first.", flush=True)
            finish_run(conn, run_id)
            return 2

        run_dir = Path(args.sidecar_dir) / f"RUN__{run_id}__{slugify_filename(started_at)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Start DB writer
        writer_q: "queue.Queue[Optional[DBWriteItem]]" = queue.Queue(maxsize=10000)
        stop_event = threading.Event()
        writer_thread = threading.Thread(
            target=db_writer_loop,
            args=(args.db, writer_q, stop_event, run_dir),
            daemon=True,
        )
        writer_thread.start()

        limiter = EndpointLimiter(per_endpoint_limit=args.per_endpoint)

        # Run workers
        failures = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = []
            for g in groups:
                futs.append(ex.submit(
                    benchmark_group,
                    g, run_id, DEFAULT_CASES, case_ids,
                    api_key, args.timeout, args.repeats, args.stream,
                    writer_q, limiter
                ))

            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    failures += 1
                    print(f"[WORKER FAIL] {e}", flush=True)

        # Wait for DB flush
        writer_q.join()
        stop_event.set()
        writer_q.put(None)
        writer_thread.join(timeout=10)

        finish_run(conn, run_id)

        print(f"\nRun complete: {run_id}")
        print(f"Sidecars: {run_dir.resolve()}")
        print(f"Worker failures: {failures}")
        return 0 if failures == 0 else 1

    finally:
        conn.close()


def create_run(conn: sqlite3.Connection, notes: str, config: Dict[str, Any]) -> int:
    conn.execute(
        "INSERT INTO bench_runs (started_at_utc, notes, config_json) VALUES (?, ?, ?)",
        (utc_now_iso(), notes, json.dumps(config)),
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])


if __name__ == "__main__":
    raise SystemExit(main())

# python3 benchmark_lmstudio_inventory_concurrent.py
#   --db lmstudio_inventory.sqlite
#   --workers 4
#   --per-endpoint 1
#   --repeats 1
#   --timeout 600
#   --stream
#   --sidecar-dir sidecar_bench_concurrent
