#!/usr/bin/env python3
"""
Benchmark LM Studio OpenAI-compatible endpoints/models discovered in an inventory CSV.

Reads rows created by the inventory export:
  - host_name, host_ip, endpoint_id, base_url, reachable, model_id, model_key

Writes benchmark artifacts:
  - run_results.csv
  - run_summary.csv
  - config.json

Emits Markdown sidecars per run.

Install:
  pip install requests

Run:
  python3 benchmark_lmstudio_inventory.py \
    --inventory-csv lmstudio_inventory.csv \
    --output-dir bench_csv \
    --sidecar-dir sidecar_bench \
    --max-models-per-endpoint 0 \
    --timeout 900 \
    --repeats 2 \
    --stream \
    --context-probe 4096

Filter endpoints/models:
  python3 benchmark_lmstudio_inventory.py \
    --inventory-csv lmstudio_inventory.csv \
    --include-endpoints http://10.0.0.5:1234/v1,http://10.0.0.6:1234/v1 \
    --exclude-models openai/gpt-oss-20b \
    --endpoint-models-file endpoint_models.json

Export inventory CSV from SQLite:
  python3 benchmark_lmstudio_inventory.py \
    --inventory-db lmstudio_inventory.sqlite \
    --export-inventory-csv lmstudio_inventory.csv \
    --output-dir bench_csv

Example endpoint_models.json:
  {
    "http://10.0.0.5:1234/v1": ["model-a", "model-b"],
    "12": ["openai/gpt-4.1-mini"]
  }

Optional judge (LLM-as-judge):
  export JUDGE_BASE_URL="http://100.105.87.118:1234/v1"
  export JUDGE_MODEL="openai/gpt-oss-20b"
  export JUDGE_API_KEY=""  # if needed

Notes:
- "Load time" is approximated by a minimal completion that forces model load.
  True unload/load control isn't part of OpenAI API; this measures practical first-use latency.
- Tokens/sec uses OpenAI-style usage when present; otherwise uses a character-based estimate.
- Inventory CSV rows can be exported from the SQLite inventory using --inventory-db and --export-inventory-csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import time
import uuid
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
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def safe_json(resp: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return resp.json()
    except Exception:
        return None


def approx_tokens_from_text(text: str) -> int:
    """
    Rough but stable: ~4 chars/token for English-ish text.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def now_s() -> float:
    return time.perf_counter()


# ----------------------------
# Benchmark Cases
# ----------------------------
@dataclass
class BenchCase:
    case_key: str
    task_type: str   # "math", "json", "reasoning", "summarize", "code"
    prompt: str
    system: str
    max_output_tokens: int
    temperature: float
    expected: Optional[Dict[str, Any]]  # for auto scoring
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
        notes="Exact-match arithmetic sanity check."
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
        notes="JSON validity + required keys."
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
        notes="Classic reasoning trap; quick quality signal."
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
        notes="Heuristic scoring: contains key terms and formatted bullets."
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
        notes="Shape-based scoring: has function + return."
    ),
]


def score_response(case: BenchCase, text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (score_0_to_1, details).
    """
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
        # heuristic: bullets + contains keywords
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        bullet_like = sum(1 for ln in lines if ln.startswith(("-", "*")))

        contains_any = exp.get("contains_any") or []
        hit = sum(1 for w in contains_any if w.lower() in (text or "").lower())

        details.update({"bullet_lines": bullet_like, "keyword_hits": hit, "keyword_total": len(contains_any)})
        # prefer 3 bullets and keyword hits
        score = 0.0
        score += min(1.0, bullet_like / 3.0) * 0.6
        score += (hit / max(1, len(contains_any))) * 0.4
        return max(0.0, min(1.0, score)), details

    if case.task_type == "code":
        regex_all = exp.get("regex_all") or []
        ok = True
        missing = []
        for pat in regex_all:
            if not re.search(pat, text or ""):
                ok = False
                missing.append(pat)
        details.update({"regex_missing": missing, "regex_ok": ok})
        return (1.0 if ok else 0.0), details

    return 0.0, {"error": "unknown_task_type"}


# ----------------------------
# OpenAI-compatible streaming parser
# ----------------------------
@dataclass
class CompletionMetrics:
    ok: bool
    http_status: Optional[int]
    error: Optional[str]
    output_text: str

    # timing
    wall_s: float
    ttft_s: Optional[float]   # time to first token (streaming only)
    load_s: Optional[float]   # for the separate load probe

    # usage
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    tokens_per_sec: Optional[float]

    finish_reason: Optional[str]
    raw_last_chunk_json: Optional[Dict[str, Any]]


def call_chat_completions_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    api_key: Optional[str],
) -> CompletionMetrics:
    """
    Streaming call to capture TTFT + wall time; attempts to capture usage if the server provides it.
    """
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
            http_status = resp.status_code
            if resp.status_code >= 400:
                body = resp.text[:2000] if resp.text else ""
                return CompletionMetrics(
                    ok=False,
                    http_status=http_status,
                    error=f"HTTP {resp.status_code}: {body}",
                    output_text="",
                    wall_s=now_s() - t0,
                    ttft_s=None,
                    load_s=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    tokens_per_sec=None,
                    finish_reason=None,
                    raw_last_chunk_json=None,
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

                # OpenAI streaming shape: choices[0].delta.content
                try:
                    choice0 = (chunk.get("choices") or [None])[0] or {}
                    delta = choice0.get("delta") or {}
                    if ttft is None and (delta.get("content") is not None):
                        ttft = now_s() - t0
                    if delta.get("content"):
                        output_parts.append(delta["content"])
                    if choice0.get("finish_reason"):
                        finish_reason = choice0.get("finish_reason")
                except Exception:
                    pass

                # Some servers include usage in final chunk or periodically
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
                # fallback estimate
                est = approx_tokens_from_text(out)
                if wall > 0:
                    completion_tokens = est
                    total_tokens = (prompt_tokens or 0) + est if prompt_tokens is not None else None
                    tps = est / wall

            return CompletionMetrics(
                ok=True,
                http_status=http_status,
                error=None,
                output_text=out,
                wall_s=wall,
                ttft_s=ttft,
                load_s=None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                tokens_per_sec=tps,
                finish_reason=finish_reason,
                raw_last_chunk_json=last_chunk if isinstance(last_chunk, dict) else None,
            )

    except RequestException as e:
        return CompletionMetrics(
            ok=False,
            http_status=None,
            error=str(e),
            output_text="",
            wall_s=now_s() - t0,
            ttft_s=None,
            load_s=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_sec=None,
            finish_reason=None,
            raw_last_chunk_json=None,
        )


def call_chat_completions_once(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    api_key: Optional[str],
) -> CompletionMetrics:
    """
    Non-streaming call to capture reliable usage on servers that provide it.
    """
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
        http_status = resp.status_code
        wall = now_s() - t0
        if resp.status_code >= 400:
            body = resp.text[:2000] if resp.text else ""
            return CompletionMetrics(
                ok=False,
                http_status=http_status,
                error=f"HTTP {resp.status_code}: {body}",
                output_text="",
                wall_s=wall,
                ttft_s=None,
                load_s=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                tokens_per_sec=None,
                finish_reason=None,
                raw_last_chunk_json=None,
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
            http_status=http_status,
            error=None,
            output_text=text,
            wall_s=wall,
            ttft_s=None,
            load_s=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tokens_per_sec=tps,
            finish_reason=finish_reason,
            raw_last_chunk_json=j,
        )

    except RequestException as e:
        return CompletionMetrics(
            ok=False,
            http_status=None,
            error=str(e),
            output_text="",
            wall_s=now_s() - t0,
            ttft_s=None,
            load_s=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_sec=None,
            finish_reason=None,
            raw_last_chunk_json=None,
        )


def probe_model_load_time(
    base_url: str,
    model: str,
    timeout_s: float,
    api_key: Optional[str],
) -> Tuple[bool, float, Optional[str], Optional[int]]:
    """
    Approximate "load into memory time" by forcing the model to respond with 1 token.
    This is the most practical measure available through OpenAI-compatible APIs.

    Returns: (ok, load_s, error, http_status)
    """
    messages = [
        {"role": "system", "content": "You are a minimal assistant."},
        {"role": "user", "content": "Say 'ok'."},
    ]
    m = call_chat_completions_once(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=1,
        temperature=0.0,
        timeout_s=timeout_s,
        api_key=api_key,
    )
    return m.ok, m.wall_s, m.error, m.http_status


def context_limit_probe(
    base_url: str,
    model: str,
    target_chars: int,
    timeout_s: float,
    api_key: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """
    Coarse context/window probe: sends a long user message of ~target_chars.
    If the server/model rejects due to context length, we capture that error.

    Returns: (ok, error)
    """
    long_text = ("x" * max(1, target_chars))
    messages = [
        {"role": "system", "content": "You are a validator."},
        {"role": "user", "content": f"Reply with just '1'. Here is filler:\n{long_text}"},
    ]
    m = call_chat_completions_once(
        base_url=base_url,
        model=model,
        messages=messages,
        max_tokens=3,
        temperature=0.0,
        timeout_s=timeout_s,
        api_key=api_key,
    )
    return m.ok, m.error


# ----------------------------
# Optional Judge Scoring
# ----------------------------
def judge_score(
    judge_base_url: str,
    judge_model: str,
    judge_api_key: Optional[str],
    case: BenchCase,
    response_text: str,
    timeout_s: float,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Ask a judge model to rate 1-10. Returns (score_0_to_1, error).
    """
    sys_msg = (
        "You are an evaluator. Rate the assistant response quality from 1 to 10 for the given task. "
        "Output ONLY a JSON object like {\"score\": 7, \"notes\": \"...\"}."
    )
    user_msg = (
        f"Task type: {case.task_type}\n"
        f"Task prompt:\n{case.prompt}\n\n"
        f"Assistant response:\n{response_text}\n\n"
        "Return JSON with integer score 1-10 and brief notes."
    )
    m = call_chat_completions_once(
        base_url=judge_base_url,
        model=judge_model,
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
        max_tokens=200,
        temperature=0.0,
        timeout_s=timeout_s,
        api_key=judge_api_key,
    )
    if not m.ok:
        return None, m.error
    try:
        obj = json.loads(m.output_text.strip())
        sc = obj.get("score")
        if isinstance(sc, (int, float)):
            return max(0.0, min(1.0, float(sc) / 10.0)), None
        return None, "Judge returned non-numeric score"
    except Exception as e:
        return None, f"Judge JSON parse failed: {e}"


# ----------------------------
# CSV Output
# ----------------------------
RESULTS_COLUMNS = [
    "run_id",
    "created_at_utc",
    "host_name",
    "host_ip",
    "endpoint_id",
    "base_url",
    "model_id",
    "model_key",
    "case_key",
    "repeat_index",
    "phase",
    "ok",
    "http_status",
    "error",
    "wall_s",
    "ttft_s",
    "tokens_per_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "finish_reason",
    "auto_quality",
    "auto_quality_details_json",
    "judge_quality",
    "judge_error",
    "output_text",
    "raw_json",
]

SUMMARY_COLUMNS = [
    "run_id",
    "host_name",
    "host_ip",
    "endpoint_id",
    "base_url",
    "model_id",
    "model_key",
    "load_s",
    "ttft_med",
    "tps_med",
    "auto_q",
    "judge_q",
    "ok_rate",
]


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def write_run_config(run_dir: Path, config: Dict[str, Any]) -> None:
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


# ----------------------------
# Inventory selection
# ----------------------------
def load_inventory_from_csv(path: str) -> List[Dict[str, Any]]:
    required_fields = {
        "host_name",
        "host_ip",
        "endpoint_id",
        "base_url",
        "reachable",
        "model_id",
        "model_key",
    }
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required_fields.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"inventory CSV missing columns: {sorted(missing)}")
        for row in reader:
            parsed = dict(row)
            for key in ("endpoint_id", "model_id"):
                try:
                    parsed[key] = int(parsed[key])
                except Exception:
                    pass
            try:
                parsed["reachable"] = int(parsed.get("reachable", 0))
            except Exception:
                parsed["reachable"] = 0
            rows.append(parsed)
    return rows


def load_inventory_from_db(path: str, export_csv_path: Optional[str]) -> List[Dict[str, Any]]:
    import sqlite3

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        where = "WHERE e.kind='lmstudio-openai-compatible'"
        q = f"""
        SELECT
          h.ts_name AS host_name,
          h.ts_ip AS host_ip,
          e.endpoint_id,
          e.base_url,
          e.reachable,
          m.model_id,
          m.model_key
        FROM endpoints e
        JOIN host_endpoints he ON he.endpoint_id=e.endpoint_id
        JOIN hosts h ON h.host_id=he.host_id
        JOIN endpoint_models em ON em.endpoint_id=e.endpoint_id
        JOIN models m ON m.model_id=em.model_id
        {where}
        ORDER BY e.endpoint_id, m.model_key
        """
        rows = [dict(r) for r in conn.execute(q).fetchall()]
    finally:
        conn.close()

    if export_csv_path:
        write_csv(Path(export_csv_path), rows, list(rows[0].keys()) if rows else list(sorted({
            "host_name",
            "host_ip",
            "endpoint_id",
            "base_url",
            "reachable",
            "model_id",
            "model_key",
        })))
    return rows


def filter_inventory_pairs(
    rows: List[Dict[str, Any]],
    only_reachable_endpoints: bool,
    max_models_per_endpoint: int,
    include_endpoints: Optional[set[str]],
    exclude_endpoints: Optional[set[str]],
    include_models: Optional[set[str]],
    exclude_models: Optional[set[str]],
    endpoint_model_map: Optional[Dict[str, set[str]]],
) -> List[Dict[str, Any]]:
    def endpoint_matches(token_set: Optional[set[str]], row: Dict[str, Any]) -> bool:
        if not token_set:
            return True
        endpoint_id = str(row["endpoint_id"])
        base_url = row["base_url"]
        host_name = row["host_name"]
        return any(tok in (endpoint_id, base_url, host_name) for tok in token_set)

    def endpoint_key_matches(token: str, row: Dict[str, Any]) -> bool:
        return token in (str(row["endpoint_id"]), row["base_url"], row["host_name"])

    filtered_rows = []
    for r in rows:
        if only_reachable_endpoints and not r.get("reachable"):
            continue
        if include_endpoints and not endpoint_matches(include_endpoints, r):
            continue
        if exclude_endpoints and endpoint_matches(exclude_endpoints, r):
            continue
        if include_models and r["model_key"] not in include_models:
            continue
        if exclude_models and r["model_key"] in exclude_models:
            continue
        if endpoint_model_map:
            matched_models = None
            for key, models in endpoint_model_map.items():
                if endpoint_key_matches(key, r):
                    matched_models = models
                    break
            if matched_models is not None and r["model_key"] not in matched_models:
                continue
        filtered_rows.append(r)

    if max_models_per_endpoint and max_models_per_endpoint > 0:
        # keep first N per endpoint
        filtered = []
        counts: Dict[int, int] = {}
        for r in filtered_rows:
            eid = r["endpoint_id"]
            counts.setdefault(eid, 0)
            if counts[eid] < max_models_per_endpoint:
                filtered.append(r)
                counts[eid] += 1
        return filtered

    return filtered_rows


def parse_csv_set(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return set(items) if items else None


def load_endpoint_model_map(path: Optional[str]) -> Optional[Dict[str, set[str]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"endpoint models file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("endpoint models file must be a JSON object mapping endpoints to model lists")
    mapped: Dict[str, set[str]] = {}
    for key, models in data.items():
        if not isinstance(key, str):
            raise ValueError("endpoint models file keys must be strings (endpoint base_url, host name, or id)")
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError(f"endpoint models for '{key}' must be a list of strings")
        mapped[key] = set(models)
    return mapped


# ----------------------------
# Sidecar Markdown
# ----------------------------
def write_run_index_md(
    run_dir: Path,
    run_id: int,
    started_at: str,
    config: Dict[str, Any],
    summary_rows: List[Dict[str, Any]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "INDEX.md"
    lines = []
    lines.append(f"# Benchmark Run `{run_id}`")
    lines.append("")
    lines.append(f"- Started (UTC): `{started_at}`")
    lines.append(f"- Generated (UTC): `{utc_now_iso()}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Summary (per endpoint/model)")
    lines.append("")
    lines.append("| Host | Endpoint | Model | Load s | Median TTFT s | Median TPS | Auto Q | Judge Q | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")

    for r in summary_rows:
        status = "✅" if r["ok_rate"] >= 0.8 else "⚠️" if r["ok_rate"] > 0 else "❌"
        lines.append(
            f"| `{r['host_name']}` | `{r['base_url']}` | `{r['model_key']}` | "
            f"{r.get('load_s','')} | {r.get('ttft_med','')} | {r.get('tps_med','')} | "
            f"{r.get('auto_q','')} | {r.get('judge_q','')} | {status} |"
        )

    lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")


def write_model_report_md(
    run_dir: Path,
    host_name: str,
    host_ip: str,
    base_url: str,
    model_key: str,
    items: List[Dict[str, Any]],
) -> None:
    fname = f"MODEL__{slugify_filename(host_name)}__{slugify_filename(model_key)}.md"
    p = run_dir / fname

    lines = []
    lines.append(f"# Model Report: `{model_key}`")
    lines.append("")
    lines.append("## Endpoint")
    lines.append("")
    lines.append(f"- Host: `{host_name}` (`{host_ip}`)")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Case | Phase | OK | Wall s | TTFT s | TPS | Auto Q | Judge Q | Error |")
    lines.append("|---|---|:---:|---:|---:|---:|---:|---:|---|")

    for it in items:
        lines.append(
            f"| `{it['case_key']}` | `{it['phase']}` | {'✅' if it['ok'] else '❌'} | "
            f"{it.get('wall_s','')} | {it.get('ttft_s','')} | {it.get('tokens_per_sec','')} | "
            f"{it.get('auto_quality','')} | {it.get('judge_quality','')} | "
            f"{(it.get('error') or '')[:80]} |"
        )

    lines.append("")
    lines.append("## Sample Outputs (truncated)")
    lines.append("")
    for it in items:
        if it.get("phase") != "run":
            continue
        out = (it.get("output_text") or "").strip()
        if not out:
            continue
        lines.append(f"### {it['case_key']} (repeat {it['repeat_index']})")
        lines.append("")
        lines.append("```text")
        lines.append(out[:2000] + ("\n... (truncated)\n" if len(out) > 2000 else ""))
        lines.append("```")
        lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Aggregation
# ----------------------------
def median_or_blank(vals: List[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    return f"{statistics.median(vals):.3f}"


def mean_or_blank(vals: List[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    return f"{statistics.mean(vals):.3f}"


def ok_rate(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if it["ok"]) / len(items)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    inventory_group = ap.add_mutually_exclusive_group(required=True)
    inventory_group.add_argument("--inventory-csv", help="Inventory CSV with endpoints/models")
    inventory_group.add_argument("--inventory-db", help="Inventory SQLite (optional, used for export)")
    ap.add_argument("--export-inventory-csv", default=None, help="When using --inventory-db, export rows to this CSV")
    ap.add_argument("--output-dir", default="bench_csv", help="Directory for CSV outputs")
    ap.add_argument("--sidecar-dir", default="sidecar_bench")

    ap.add_argument("--only-reachable-endpoints", action="store_true", default=False)
    ap.add_argument("--max-models-per-endpoint", type=int, default=0, help="0 = no limit")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)

    ap.add_argument("--stream", action="store_true", default=True, help="Use streaming for TTFT")
    ap.add_argument("--warmup", action="store_true", default=True, help="Warmup each model with a tiny call before benchmarking cases")
    ap.add_argument("--cooldown-secs", type=float, default=0.0, help="Sleep between models (helpful on small boxes)")

    ap.add_argument("--context-probe", type=int, default=0, help="If >0, do a coarse context probe with ~this many characters")
    ap.add_argument("--api-key", default=None, help="Bearer token for endpoints, if needed")
    ap.add_argument("--notes", default="")
    ap.add_argument(
        "--include-endpoints",
        default=None,
        help="Comma-separated list of endpoint base_url, host name, or endpoint_id to include",
    )
    ap.add_argument(
        "--exclude-endpoints",
        default=None,
        help="Comma-separated list of endpoint base_url, host name, or endpoint_id to exclude",
    )
    ap.add_argument("--include-models", default=None, help="Comma-separated list of model keys to include")
    ap.add_argument("--exclude-models", default=None, help="Comma-separated list of model keys to exclude")
    ap.add_argument(
        "--endpoint-models-file",
        default=None,
        help="JSON file mapping endpoint base_url/host name/endpoint_id to list of model keys to include",
    )

    args = ap.parse_args()

    inventory_csv = args.inventory_csv
    inventory_db = args.inventory_db

    api_key = args.api_key or os.environ.get("LMSTUDIO_API_KEY") or None

    judge_base = os.environ.get("JUDGE_BASE_URL") or ""
    judge_model = os.environ.get("JUDGE_MODEL") or ""
    judge_api_key = os.environ.get("JUDGE_API_KEY") or None
    use_judge = bool(judge_base.strip() and judge_model.strip())

    config = {
        "repeats": args.repeats,
        "timeout": args.timeout,
        "stream": args.stream,
        "warmup": args.warmup,
        "cooldown_secs": args.cooldown_secs,
        "context_probe_chars": args.context_probe,
        "only_reachable_endpoints": args.only_reachable_endpoints,
        "max_models_per_endpoint": args.max_models_per_endpoint,
        "inventory_csv": inventory_csv,
        "inventory_db": inventory_db,
        "export_inventory_csv": args.export_inventory_csv,
        "output_dir": args.output_dir,
        "include_endpoints": args.include_endpoints,
        "exclude_endpoints": args.exclude_endpoints,
        "include_models": args.include_models,
        "exclude_models": args.exclude_models,
        "endpoint_models_file": args.endpoint_models_file,
        "use_judge": use_judge,
        "judge_base_url": judge_base if use_judge else None,
        "judge_model": judge_model if use_judge else None,
    }

    include_endpoints = parse_csv_set(args.include_endpoints)
    exclude_endpoints = parse_csv_set(args.exclude_endpoints)
    include_models = parse_csv_set(args.include_models)
    exclude_models = parse_csv_set(args.exclude_models)
    endpoint_model_map = load_endpoint_model_map(args.endpoint_models_file)

    started_at = utc_now_iso()
    run_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

    if inventory_csv:
        inventory_rows = load_inventory_from_csv(inventory_csv)
    else:
        inventory_rows = load_inventory_from_db(inventory_db, args.export_inventory_csv)

    pairs = filter_inventory_pairs(
        inventory_rows,
        only_reachable_endpoints=args.only_reachable_endpoints,
        max_models_per_endpoint=args.max_models_per_endpoint,
        include_endpoints=include_endpoints,
        exclude_endpoints=exclude_endpoints,
        include_models=include_models,
        exclude_models=exclude_models,
        endpoint_model_map=endpoint_model_map,
    )

    if not pairs:
        print("No endpoint/model pairs found. Did you run the inventory export first?", flush=True)
        return 2

    run_dir = Path(args.output_dir) / f"RUN__{run_id}__{slugify_filename(started_at)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(run_dir, config)

    sidecar_dir = Path(args.sidecar_dir) / f"RUN__{run_id}__{slugify_filename(started_at)}"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    # Group by endpoint/model so we can do warmup/load + then cases
    groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in pairs:
        key = (r["endpoint_id"], r["model_id"])
        groups.setdefault(key, r)

    summary_rows: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []

    for (endpoint_id, model_id), r in groups.items():
        host_name = r["host_name"]
        host_ip = r["host_ip"]
        base_url = r["base_url"]
        model_key = r["model_key"]

        print(f"\n=== {host_name} | {base_url} | {model_key} ===", flush=True)

        # 1) Load probe (approx)
        ok_load, load_s, load_err, load_http = probe_model_load_time(
            base_url=base_url, model=model_key, timeout_s=args.timeout, api_key=api_key
        )
        load_metrics = CompletionMetrics(
            ok=ok_load,
            http_status=load_http,
            error=load_err,
            output_text="",
            wall_s=load_s,
            ttft_s=None,
            load_s=load_s,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            tokens_per_sec=None,
            finish_reason=None,
            raw_last_chunk_json=None,
        )
        created_at = utc_now_iso()
        results_rows.append({
            "run_id": run_id,
            "created_at_utc": created_at,
            "host_name": host_name,
            "host_ip": host_ip,
            "endpoint_id": endpoint_id,
            "base_url": base_url,
            "model_id": model_id,
            "model_key": model_key,
            "case_key": DEFAULT_CASES[0].case_key,
            "repeat_index": 0,
            "phase": "load",
            "ok": int(load_metrics.ok),
            "http_status": load_metrics.http_status or "",
            "error": load_metrics.error or "",
            "wall_s": load_metrics.wall_s,
            "ttft_s": load_metrics.ttft_s or "",
            "tokens_per_sec": load_metrics.tokens_per_sec or "",
            "prompt_tokens": load_metrics.prompt_tokens or "",
            "completion_tokens": load_metrics.completion_tokens or "",
            "total_tokens": load_metrics.total_tokens or "",
            "finish_reason": load_metrics.finish_reason or "",
            "auto_quality": "",
            "auto_quality_details_json": json.dumps({"note": "load probe"}),
            "judge_quality": "",
            "judge_error": "",
            "output_text": "",
            "raw_json": "",
        })

        if not ok_load:
            # if it can't even load, we still proceed to record errors for the run phases, but likely will fail
            print(f"  [LOAD FAIL] {load_err}", flush=True)
        else:
            print(f"  [LOAD OK] {load_s:.3f}s", flush=True)

        # 2) Optional warmup
        if args.warmup:
            warm = call_chat_completions_once(
                base_url=base_url,
                model=model_key,
                messages=[{"role": "system", "content": "You are a warmup."}, {"role": "user", "content": "Respond with OK."}],
                max_tokens=16,
                temperature=0.0,
                timeout_s=args.timeout,
                api_key=api_key,
            )
            results_rows.append({
                "run_id": run_id,
                "created_at_utc": utc_now_iso(),
                "host_name": host_name,
                "host_ip": host_ip,
                "endpoint_id": endpoint_id,
                "base_url": base_url,
                "model_id": model_id,
                "model_key": model_key,
                "case_key": DEFAULT_CASES[0].case_key,
                "repeat_index": 0,
                "phase": "warmup",
                "ok": int(warm.ok),
                "http_status": warm.http_status or "",
                "error": warm.error or "",
                "wall_s": warm.wall_s,
                "ttft_s": warm.ttft_s or "",
                "tokens_per_sec": warm.tokens_per_sec or "",
                "prompt_tokens": warm.prompt_tokens or "",
                "completion_tokens": warm.completion_tokens or "",
                "total_tokens": warm.total_tokens or "",
                "finish_reason": warm.finish_reason or "",
                "auto_quality": "",
                "auto_quality_details_json": json.dumps({"note": "warmup call"}),
                "judge_quality": "",
                "judge_error": "",
                "output_text": warm.output_text or "",
                "raw_json": json.dumps(warm.raw_last_chunk_json) if warm.raw_last_chunk_json else "",
            })
            print(f"  [WARMUP] ok={warm.ok} wall={warm.wall_s:.3f}s", flush=True)

        # 3) Optional context probe
        if args.context_probe and args.context_probe > 0:
            ok_ctx, ctx_err = context_limit_probe(
                base_url=base_url,
                model=model_key,
                target_chars=args.context_probe,
                timeout_s=args.timeout,
                api_key=api_key,
            )
            results_rows.append({
                "run_id": run_id,
                "created_at_utc": utc_now_iso(),
                "host_name": host_name,
                "host_ip": host_ip,
                "endpoint_id": endpoint_id,
                "base_url": base_url,
                "model_id": model_id,
                "model_key": model_key,
                "case_key": DEFAULT_CASES[0].case_key,
                "repeat_index": 0,
                "phase": "context_probe",
                "ok": int(ok_ctx),
                "http_status": "",
                "error": ctx_err or "",
                "wall_s": "",
                "ttft_s": "",
                "tokens_per_sec": "",
                "prompt_tokens": "",
                "completion_tokens": "",
                "total_tokens": "",
                "finish_reason": "",
                "auto_quality": "",
                "auto_quality_details_json": json.dumps({"probe_chars": args.context_probe}),
                "judge_quality": "",
                "judge_error": "",
                "output_text": "",
                "raw_json": "",
            })
            print(f"  [CTX] ok={ok_ctx} err={(ctx_err or '')[:120]}", flush=True)

        # 4) Run cases x repeats
        model_items_for_md: List[Dict[str, Any]] = []
        per_model_run_rows: List[Dict[str, Any]] = []

        for case in DEFAULT_CASES:
            messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]

            for rep in range(args.repeats):
                if args.stream:
                    met = call_chat_completions_stream(
                        base_url=base_url,
                        model=model_key,
                        messages=messages,
                        max_tokens=case.max_output_tokens,
                        temperature=case.temperature,
                        timeout_s=args.timeout,
                        api_key=api_key,
                    )
                    # If streaming didn't provide usage, add a cheap non-stream pass for usage (optional).
                    # To avoid doubling cost by default, only do it if completion_tokens is missing.
                    if met.ok and (met.completion_tokens is None):
                        met2 = call_chat_completions_once(
                            base_url=base_url,
                            model=model_key,
                            messages=messages,
                            max_tokens=case.max_output_tokens,
                            temperature=case.temperature,
                            timeout_s=args.timeout,
                            api_key=api_key,
                        )
                        # Merge usage from met2 into met when possible
                        if met2.ok and met2.completion_tokens is not None:
                            met.completion_tokens = met2.completion_tokens
                            met.prompt_tokens = met2.prompt_tokens
                            met.total_tokens = met2.total_tokens
                            if met.wall_s > 0:
                                met.tokens_per_sec = float(met.completion_tokens) / met.wall_s
                else:
                    met = call_chat_completions_once(
                        base_url=base_url,
                        model=model_key,
                        messages=messages,
                        max_tokens=case.max_output_tokens,
                        temperature=case.temperature,
                        timeout_s=args.timeout,
                        api_key=api_key,
                    )

                auto_q = None
                auto_details = None
                judge_q = None
                judge_err = None

                if met.ok:
                    auto_q, auto_details = score_response(case, met.output_text)
                    if use_judge:
                        judge_q, judge_err = judge_score(
                            judge_base_url=judge_base,
                            judge_model=judge_model,
                            judge_api_key=judge_api_key,
                            case=case,
                            response_text=met.output_text,
                            timeout_s=args.timeout,
                        )

                results_rows.append({
                    "run_id": run_id,
                    "created_at_utc": utc_now_iso(),
                    "host_name": host_name,
                    "host_ip": host_ip,
                    "endpoint_id": endpoint_id,
                    "base_url": base_url,
                    "model_id": model_id,
                    "model_key": model_key,
                    "case_key": case.case_key,
                    "repeat_index": rep,
                    "phase": "run",
                    "ok": int(met.ok),
                    "http_status": met.http_status or "",
                    "error": met.error or "",
                    "wall_s": met.wall_s,
                    "ttft_s": met.ttft_s or "",
                    "tokens_per_sec": met.tokens_per_sec or "",
                    "prompt_tokens": met.prompt_tokens or "",
                    "completion_tokens": met.completion_tokens or "",
                    "total_tokens": met.total_tokens or "",
                    "finish_reason": met.finish_reason or "",
                    "auto_quality": f"{auto_q:.2f}" if auto_q is not None else "",
                    "auto_quality_details_json": json.dumps(auto_details) if auto_details is not None else "",
                    "judge_quality": f"{judge_q:.2f}" if judge_q is not None else "",
                    "judge_error": judge_err or "",
                    "output_text": (met.output_text or "")[:20000],
                    "raw_json": json.dumps(met.raw_last_chunk_json)[:20000] if met.raw_last_chunk_json is not None else "",
                })

                model_items_for_md.append({
                    "case_key": case.case_key,
                    "phase": "run",
                    "repeat_index": rep,
                    "ok": met.ok,
                    "wall_s": f"{met.wall_s:.3f}" if met.wall_s is not None else "",
                    "ttft_s": f"{met.ttft_s:.3f}" if met.ttft_s is not None else "",
                    "tokens_per_sec": f"{met.tokens_per_sec:.2f}" if met.tokens_per_sec is not None else "",
                    "auto_quality": f"{auto_q:.2f}" if auto_q is not None else "",
                    "judge_quality": f"{judge_q:.2f}" if judge_q is not None else "",
                    "error": met.error,
                    "output_text": met.output_text,
                })

                per_model_run_rows.append({
                    "ok": met.ok,
                    "ttft_s": met.ttft_s,
                    "tps": met.tokens_per_sec,
                    "auto_q": auto_q,
                    "judge_q": judge_q,
                })

                status = "OK" if met.ok else "FAIL"
                print(
                    f"  [{status}] {case.case_key} rep={rep} wall={met.wall_s:.3f}s "
                    f"ttft={(met.ttft_s if met.ttft_s is not None else '')} "
                    f"tps={(met.tokens_per_sec if met.tokens_per_sec is not None else '')} "
                    f"autoQ={(auto_q if auto_q is not None else '')} "
                    f"judgeQ={(judge_q if judge_q is not None else '')}",
                    flush=True
                )

        # Sidecar per model
        # Also include load/warmup/context rows in MD table
        model_items_for_md.insert(0, {
            "case_key": DEFAULT_CASES[0].case_key,
            "phase": "load",
            "repeat_index": 0,
            "ok": ok_load,
            "wall_s": f"{load_s:.3f}",
            "ttft_s": "",
            "tokens_per_sec": "",
            "auto_quality": "",
            "judge_quality": "",
            "error": load_err,
            "output_text": "",
        })

        # Aggregate summary
        ttfts = [x["ttft_s"] for x in per_model_run_rows if x["ok"] and x["ttft_s"] is not None]
        tpss = [x["tps"] for x in per_model_run_rows if x["ok"] and x["tps"] is not None]
        autoqs = [x["auto_q"] for x in per_model_run_rows if x["ok"] and x["auto_q"] is not None]
        judgeqs = [x["judge_q"] for x in per_model_run_rows if x["ok"] and x["judge_q"] is not None]

        srow = {
            "run_id": run_id,
            "host_name": host_name,
            "host_ip": host_ip,
            "endpoint_id": endpoint_id,
            "base_url": base_url,
            "model_id": model_id,
            "model_key": model_key,
            "load_s": f"{load_s:.3f}" if ok_load else "",
            "ttft_med": median_or_blank(ttfts),
            "tps_med": median_or_blank(tpss),
            "auto_q": mean_or_blank(autoqs),
            "judge_q": mean_or_blank(judgeqs),
            "ok_rate": ok_rate(per_model_run_rows),
        }
        summary_rows.append(srow)

        write_model_report_md(
            run_dir=sidecar_dir,
            host_name=host_name,
            host_ip=host_ip,
            base_url=base_url,
            model_key=model_key,
            items=model_items_for_md,
        )

        if args.cooldown_secs and args.cooldown_secs > 0:
            time.sleep(args.cooldown_secs)

    write_run_index_md(sidecar_dir, run_id, started_at, config, summary_rows)
    write_csv(run_dir / "run_results.csv", results_rows, RESULTS_COLUMNS)
    write_csv(run_dir / "run_summary.csv", summary_rows, SUMMARY_COLUMNS)

    print(f"\nRun complete: {run_id}")
    print(f"CSV output: {run_dir.resolve()}")
    print(f"Sidecars: {sidecar_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# python3 benchmark_lmstudio_inventory.py \
#   --inventory-csv lmstudio_inventory.csv \
#   --sidecar-dir sidecar_bench \
#   --repeats 2 \
#   --timeout 900 \
#   --stream \
#   --context-probe 4096

# python3 benchmark_lmstudio_inventory.py \
#   --inventory-csv lmstudio_inventory.csv \
#   --sidecar-dir sidecar_bench \
#   --max-models-per-endpoint 3 \
#   --repeats 1 \
#   --cooldown-secs 2 \
#   --timeout 600

# export JUDGE_BASE_URL="http://100.105.87.118:1234/v1"
# export JUDGE_MODEL="openai/gpt-oss-20b"
# python3 benchmark_lmstudio_inventory.py --inventory-csv lmstudio_inventory.csv
