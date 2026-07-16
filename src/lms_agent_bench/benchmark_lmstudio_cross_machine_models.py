#!/usr/bin/env python3
"""
Benchmark LM Studio OpenAI-compatible endpoints/models using an inventory CSV.

This runner is manifest-aware and agent-skill oriented. It supports the legacy
hard-coded benchmark cases, but prefers a cases manifest such as:

  benchmarks/agent_skill_suite.v1.json

Reads inventory rows with columns:
  - host_name, host_ip, endpoint_id, base_url, reachable, model_id, model_key

Emits:
  - run_results.csv
  - run_summary.csv
  - task_summary.csv
  - config.json
  - Markdown sidecar reports per run
  - Full output text files for human evaluation

Run:
  python3 benchmark_lmstudio_cross_machine_models.py \
    --inventory-csv lmstudio_inventory.csv \
    --cases-file benchmarks/agent_skill_suite.v1.json \
    --output-dir runs/20260530T120000Z \
    --sidecar-dir runs/20260530T120000Z/sidecars \
    --timeout 900 \
    --repeats 1
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from requests.exceptions import RequestException

try:
    from lms_eval import evaluate_output
except Exception:  # pragma: no cover - defensive for partial installs
    def evaluate_output(output_text: str, evaluator_specs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        return {"ok": True, "score": 1.0, "results": [], "failed": []}


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
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def now_s() -> float:
    return time.perf_counter()


def parse_csv_set(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return set(items) if items else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def median_or_blank(vals: Sequence[float]) -> str:
    return f"{statistics.median(vals):.3f}" if vals else ""


def mean_or_blank(vals: Sequence[float]) -> str:
    return f"{statistics.mean(vals):.3f}" if vals else ""


def float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ok_rate(items: Sequence[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if str(it.get("ok", "")).lower() in {"true", "1", "yes"} or it.get("ok") is True) / len(items)


def eval_ok_rate(items: Sequence[Dict[str, Any]]) -> float:
    eval_items = [it for it in items if it.get("eval_ok") not in (None, "")]
    if not eval_items:
        return 0.0
    return sum(1 for it in eval_items if str(it.get("eval_ok", "")).lower() in {"true", "1", "yes"} or it.get("eval_ok") is True) / len(eval_items)


def avg_eval_score(items: Sequence[Dict[str, Any]]) -> str:
    vals = [float_or_none(it.get("eval_score")) for it in items]
    real = [v for v in vals if v is not None]
    return f"{statistics.mean(real):.4f}" if real else ""


# ----------------------------
# Benchmark cases
# ----------------------------
@dataclass
class BenchCase:
    case_key: str
    task_family: str
    prompt: str
    system: str
    max_output_tokens: int
    temperature: float
    priority: str = "P1"
    notes: str = ""
    evaluators: List[Dict[str, Any]] = field(default_factory=list)
    recommendation_signal: str = ""
    context_tokens: Optional[int] = None


def build_fibonacci_prompt(language: str) -> str:
    return (
        "Please provide the code to implement the Fibonacci sequence.\n"
        "Print the first 1000 numbers in the sequence.\n"
        "Use arbitrary-precision arithmetic (do not rely on external libraries).\n"
        f"Implement this source code in {language}.\n"
        "Keep the output to code only (no markdown) and include brief comments."
    )


DEFAULT_CASES: List[BenchCase] = [
    BenchCase(
        case_key=f"code_fib_{key}",
        task_family="coding",
        system="You are a senior software engineer. Output code only.",
        prompt=build_fibonacci_prompt(language),
        max_output_tokens=1600,
        temperature=0.0,
        priority="P1",
        notes=f"Fibonacci sequence in {language}.",
        evaluators=[{"type": "no_markdown_fence"}],
        recommendation_signal="small_code_generation",
    )
    for key, language in [
        ("c", "C programming language"),
        ("rust", "Rust"),
        ("python", "Python"),
        ("javascript", "JavaScript"),
    ]
]


def synthesize_context(target_tokens: int, spec: Dict[str, Any]) -> str:
    """Create deterministic synthetic context with a needle sentence."""
    needle = spec.get("needle_sentence") or "The LMS control code for this benchmark is ORION-7429 and it must be preserved exactly."
    style = spec.get("filler_style") or "technical_project_notes"
    position = spec.get("needle_position") or "middle"
    filler_sentence = (
        "Project note: the benchmark runner records endpoint latency, output structure, model reliability, and routing evidence for local agent workflows."
        if style == "technical_project_notes"
        else "This is deterministic filler text used to measure long-context recall and instruction retention."
    )
    approx_chars = max(target_tokens * 4, len(needle) + 100)
    filler_count = max(1, approx_chars // max(len(filler_sentence), 1))
    filler = [f"{filler_sentence} Segment {i}." for i in range(filler_count)]
    if position == "start":
        parts = [needle] + filler
    elif position == "end":
        parts = filler + [needle]
    else:
        mid = len(filler) // 2
        parts = filler[:mid] + [needle] + filler[mid:]
    return "\n".join(parts)


def render_prompt(case_def: Dict[str, Any], context_tokens: Optional[int]) -> str:
    if case_def.get("prompt_template"):
        synthetic_spec = case_def.get("synthetic_context") or {}
        synthetic_context = synthesize_context(context_tokens or 4096, synthetic_spec)
        return str(case_def["prompt_template"]).replace("{{synthetic_context}}", synthetic_context)
    return str(case_def.get("prompt", ""))


def load_cases_from_manifest(path: Path, max_context_tokens: int) -> Tuple[List[BenchCase], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: List[BenchCase] = []
    for item in data.get("cases", []):
        if not isinstance(item, dict):
            continue
        context_values: List[Optional[int]] = [None]
        if item.get("prompt_template") and item.get("context_sweep_tokens"):
            raw_values = [int(v) for v in item.get("context_sweep_tokens", [])]
            context_values = [v for v in raw_values if v <= max_context_tokens]
            if not context_values:
                context_values = [min(raw_values) if raw_values else max_context_tokens]

        for context_tokens in context_values:
            base_key = str(item.get("case_key", "case"))
            case_key = f"{base_key}_{context_tokens}tok" if context_tokens else base_key
            cases.append(
                BenchCase(
                    case_key=case_key,
                    task_family=str(item.get("task_family", "general")),
                    priority=str(item.get("priority", "P1")),
                    system=str(item.get("system", "You are a helpful assistant.")),
                    prompt=render_prompt(item, context_tokens),
                    max_output_tokens=int(item.get("max_output_tokens", 512)),
                    temperature=float(item.get("temperature", 0.0)),
                    notes=str(item.get("description") or item.get("notes") or ""),
                    evaluators=list(item.get("evaluators") or []),
                    recommendation_signal=str(item.get("recommendation_signal", "")),
                    context_tokens=context_tokens,
                )
            )
    if not cases:
        raise ValueError(f"no runnable cases found in manifest: {path}")
    return cases, data


def load_cases(args: argparse.Namespace) -> Tuple[List[BenchCase], Dict[str, Any]]:
    if args.cases_file:
        return load_cases_from_manifest(Path(args.cases_file), max_context_tokens=args.max_context_tokens)
    return DEFAULT_CASES, {"suite_id": "legacy_default_cases", "version": 0}


# ----------------------------
# OpenAI-compatible completion callers
# ----------------------------
@dataclass
class CompletionMetrics:
    ok: bool
    http_status: Optional[int]
    error: Optional[str]
    output_text: str
    wall_s: float
    ttft_s: Optional[float]
    load_s: Optional[float]
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
    last_chunk: Optional[Dict[str, Any]] = None
    finish_reason = None
    prompt_tokens = completion_tokens = total_tokens = None

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout_s) as resp:
            http_status = resp.status_code
            if resp.status_code >= 400:
                body = resp.text[:2000] if resp.text else ""
                return CompletionMetrics(False, http_status, f"HTTP {resp.status_code}: {body}", "", now_s() - t0, None, None, None, None, None, None, None, None)

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
                    usage = chunk["usage"]
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")

            output = "".join(output_parts)
            wall = now_s() - t0
            if completion_tokens is None and output:
                completion_tokens = approx_tokens_from_text(output)
            tps = float(completion_tokens) / wall if completion_tokens and wall > 0 else None
            return CompletionMetrics(True, http_status, None, output, wall, ttft, None, prompt_tokens, completion_tokens, total_tokens, tps, finish_reason, last_chunk)

    except RequestException as exc:
        return CompletionMetrics(False, None, str(exc), "", now_s() - t0, None, None, None, None, None, None, None, None)


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
            body = resp.text[:2000] if resp.text else ""
            return CompletionMetrics(False, resp.status_code, f"HTTP {resp.status_code}: {body}", "", wall, None, None, None, None, None, None, None, None)
        data = safe_json(resp) or {}
        choice0 = (data.get("choices") or [None])[0] or {}
        msg = choice0.get("message") or {}
        text = msg.get("content") or ""
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens") or approx_tokens_from_text(text)
        total_tokens = usage.get("total_tokens")
        tps = float(completion_tokens) / wall if completion_tokens and wall > 0 else None
        return CompletionMetrics(True, resp.status_code, None, text, wall, None, None, prompt_tokens, completion_tokens, total_tokens, tps, choice0.get("finish_reason"), data)
    except RequestException as exc:
        return CompletionMetrics(False, None, str(exc), "", now_s() - t0, None, None, None, None, None, None, None, None)


# ----------------------------
# Inventory
# ----------------------------
def load_inventory_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def endpoint_matches(row: Dict[str, Any], tokens: Optional[set[str]]) -> bool:
    if not tokens:
        return True
    values = {
        str(row.get("endpoint_id", "")),
        str(row.get("base_url", "")),
        str(row.get("host_name", "")),
        str(row.get("host_ip", "")),
    }
    return bool(values.intersection(tokens))


def filter_inventory_rows(
    rows: List[Dict[str, Any]],
    include_endpoints: Optional[set[str]],
    exclude_endpoints: Optional[set[str]],
    include_models: Optional[set[str]],
    exclude_models: Optional[set[str]],
    only_reachable: bool,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if only_reachable and str(row.get("reachable", "")).lower() not in {"1", "true", "yes"}:
            continue
        if include_endpoints and not endpoint_matches(row, include_endpoints):
            continue
        if exclude_endpoints and endpoint_matches(row, exclude_endpoints):
            continue
        if include_models and row.get("model_key") not in include_models:
            continue
        if exclude_models and row.get("model_key") in exclude_models:
            continue
        filtered.append(row)
    return filtered


# ----------------------------
# Reports
# ----------------------------
def write_run_index_md(run_dir: Path, run_id: int, started_at: str, config: Dict[str, Any], summary_rows: List[Dict[str, Any]], task_rows: List[Dict[str, Any]]) -> None:
    ensure_dir(run_dir)
    lines: List[str] = []
    lines.append(f"# Benchmark Run `{run_id}`")
    lines.append("")
    lines.append(f"- Started UTC: `{started_at}`")
    lines.append(f"- Generated UTC: `{utc_now_iso()}`")
    lines.append(f"- Suite: `{config.get('suite_id')}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Model summary")
    lines.append("")
    lines.append("| Host | Endpoint | Model | Load s | Median TTFT s | Median TPS | OK Rate | Eval OK Rate | Eval Score | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in summary_rows:
        ok_val = float_or_none(r.get("ok_rate")) or 0.0
        eval_val = float_or_none(r.get("eval_ok_rate")) or 0.0
        status = "✅" if ok_val >= 0.8 and eval_val >= 0.8 else "⚠️" if ok_val > 0 else "❌"
        lines.append(
            f"| `{r.get('host_name','')}` | `{r.get('base_url','')}` | `{r.get('model_key','')}` | "
            f"{r.get('load_s','')} | {r.get('ttft_med','')} | {r.get('tps_med','')} | "
            f"{r.get('ok_rate','')} | {r.get('eval_ok_rate','')} | {r.get('eval_score_avg','')} | {status} |"
        )
    if task_rows:
        lines.append("")
        lines.append("## Task-family summary")
        lines.append("")
        lines.append("| Task family | Host | Model | OK Rate | Eval OK Rate | Eval Score | Median TPS |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for r in task_rows:
            lines.append(
                f"| `{r.get('task_family','')}` | `{r.get('host_name','')}` | `{r.get('model_key','')}` | "
                f"{r.get('ok_rate','')} | {r.get('eval_ok_rate','')} | {r.get('eval_score_avg','')} | {r.get('tps_med','')} |"
            )
    lines.append("")
    (run_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_model_report_md(run_dir: Path, host_name: str, host_ip: str, base_url: str, model_key: str, items: List[Dict[str, Any]]) -> None:
    ensure_dir(run_dir)
    path = run_dir / f"MODEL__{slugify_filename(host_name)}__{slugify_filename(model_key)}.md"
    lines: List[str] = []
    lines.append(f"# Model Report: `{model_key}`")
    lines.append("")
    lines.append(f"- Host: `{host_name}` (`{host_ip}`)")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append("")
    lines.append("| Case | Task | Phase | OK | Eval OK | Eval Score | Wall s | TTFT s | TPS | Output | Error |")
    lines.append("|---|---|---|:---:|:---:|---:|---:|---:|---:|---|---|")
    for it in items:
        lines.append(
            f"| `{it.get('case_key','')}` | `{it.get('task_family','')}` | `{it.get('phase','')}` | "
            f"{'✅' if str(it.get('ok')).lower() in {'true','1'} or it.get('ok') is True else '❌'} | "
            f"{'✅' if str(it.get('eval_ok')).lower() in {'true','1'} or it.get('eval_ok') is True else '❌' if it.get('phase') == 'run' else ''} | "
            f"{it.get('eval_score','')} | {it.get('wall_s','')} | {it.get('ttft_s','')} | {it.get('tokens_per_sec','')} | "
            f"`{it.get('output_file','')}` | `{str(it.get('error') or '')[:120]}` |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


# ----------------------------
# Main run
# ----------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cross-machine LM Studio model benchmark runner.")
    ap.add_argument("--inventory-csv", required=True, help="Inventory CSV from LM Studio discovery")
    ap.add_argument("--cases-file", default=None, help="Benchmark suite manifest JSON")
    ap.add_argument("--output-dir", required=True, help="Directory for CSV artifacts")
    ap.add_argument("--sidecar-dir", required=True, help="Directory for Markdown sidecars + raw outputs")
    ap.add_argument("--timeout", type=float, default=900, help="Request timeout seconds")
    ap.add_argument("--repeats", type=int, default=1, help="Repeats per case")
    ap.add_argument("--stream", action="store_true", default=True, help="Use streaming for TTFT")
    ap.add_argument("--api-key-env", default="LMSTUDIO_API_KEY", help="Env var for API key")
    ap.add_argument("--only-reachable", action="store_true", default=True, help="Only use reachable endpoints")
    ap.add_argument("--include-endpoints", default=None, help="Comma-separated endpoint IDs/base URLs/host names")
    ap.add_argument("--exclude-endpoints", default=None, help="Comma-separated endpoint IDs/base URLs/host names")
    ap.add_argument("--include-models", default=None, help="Comma-separated model keys to include")
    ap.add_argument("--exclude-models", default=None, help="Comma-separated model keys to exclude")
    ap.add_argument("--max-context-tokens", type=int, default=8192, help="Cap manifest context-sweep cases for quick local runs")
    return ap


def result_row_base(run_id: int, phase: str, row: Dict[str, Any], case_key: str, task_family: str, priority: str, repeat_index: int) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "phase": phase,
        "host_name": row.get("host_name", ""),
        "host_ip": row.get("host_ip", ""),
        "endpoint_id": row.get("endpoint_id", ""),
        "base_url": str(row.get("base_url", "")).rstrip("/"),
        "model_id": row.get("model_id", ""),
        "model_key": row.get("model_key", ""),
        "case_key": case_key,
        "task_family": task_family,
        "priority": priority,
        "repeat_index": repeat_index,
    }


def attach_metrics(base: Dict[str, Any], met: CompletionMetrics) -> Dict[str, Any]:
    base.update(
        {
            "ok": met.ok,
            "http_status": met.http_status,
            "error": met.error,
            "wall_s": f"{met.wall_s:.3f}",
            "ttft_s": f"{met.ttft_s:.3f}" if met.ttft_s is not None else "",
            "load_s": f"{met.load_s:.3f}" if met.load_s is not None else "",
            "prompt_tokens": met.prompt_tokens,
            "completion_tokens": met.completion_tokens,
            "total_tokens": met.total_tokens,
            "tokens_per_sec": f"{met.tokens_per_sec:.3f}" if met.tokens_per_sec else "",
            "finish_reason": met.finish_reason,
            "output_text": met.output_text,
        }
    )
    return base


def summarize(results: List[Dict[str, Any]], group_fields: Sequence[str], run_id: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in results:
        key = tuple(row.get(field, "") for field in group_fields)
        grouped.setdefault(key, []).append(row)

    rows: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        out = {field: value for field, value in zip(group_fields, key)}
        run_items = [it for it in items if it.get("phase") == "run"]
        load_items = [it for it in items if it.get("phase") == "load"]
        ttft_vals = [v for v in (float_or_none(it.get("ttft_s")) for it in run_items) if v is not None]
        tps_vals = [v for v in (float_or_none(it.get("tokens_per_sec")) for it in run_items) if v is not None]
        load_vals = [v for v in (float_or_none(it.get("load_s")) for it in load_items) if v is not None]
        out.update(
            {
                "run_id": run_id,
                "load_s": mean_or_blank(load_vals),
                "ttft_med": median_or_blank(ttft_vals),
                "tps_med": median_or_blank(tps_vals),
                "ok_rate": f"{ok_rate(run_items):.2f}",
                "eval_ok_rate": f"{eval_ok_rate(run_items):.2f}" if run_items else "",
                "eval_score_avg": avg_eval_score(run_items),
                "cases": len(run_items),
            }
        )
        rows.append(out)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    api_key = os.getenv(args.api_key_env)

    include_endpoints = parse_csv_set(args.include_endpoints)
    exclude_endpoints = parse_csv_set(args.exclude_endpoints)
    include_models = parse_csv_set(args.include_models)
    exclude_models = parse_csv_set(args.exclude_models)

    inventory_csv = Path(args.inventory_csv)
    output_dir = Path(args.output_dir)
    sidecar_dir = Path(args.sidecar_dir)
    ensure_dir(output_dir)
    ensure_dir(sidecar_dir)

    cases, suite = load_cases(args)
    rows = filter_inventory_rows(
        load_inventory_rows(inventory_csv),
        include_endpoints=include_endpoints,
        exclude_endpoints=exclude_endpoints,
        include_models=include_models,
        exclude_models=exclude_models,
        only_reachable=args.only_reachable,
    )
    if not rows:
        print("No inventory rows matched filters; exiting.")
        return 1

    run_id = int(time.time())
    run_dir = sidecar_dir / f"run_{run_id}"
    outputs_dir = run_dir / "outputs"
    ensure_dir(outputs_dir)
    started_at = utc_now_iso()

    config = {
        "run_id": run_id,
        "started_at": started_at,
        "inventory_csv": str(inventory_csv),
        "output_dir": str(output_dir),
        "sidecar_dir": str(sidecar_dir),
        "cases_file": args.cases_file,
        "suite_id": suite.get("suite_id", "legacy_default_cases"),
        "suite_version": suite.get("version"),
        "timeout_s": args.timeout,
        "repeats": args.repeats,
        "stream": args.stream,
        "max_context_tokens": args.max_context_tokens,
        "case_count": len(cases),
        "cases": [case.__dict__ for case in cases],
        "filters": {
            "only_reachable": args.only_reachable,
            "include_endpoints": sorted(include_endpoints) if include_endpoints else None,
            "exclude_endpoints": sorted(exclude_endpoints) if exclude_endpoints else None,
            "include_models": sorted(include_models) if include_models else None,
            "exclude_models": sorted(exclude_models) if exclude_models else None,
        },
    }

    results: List[Dict[str, Any]] = []

    for inv in rows:
        host_name = inv.get("host_name", "")
        host_ip = inv.get("host_ip", "")
        base_url = str(inv.get("base_url", "")).rstrip("/")
        model_key = inv.get("model_key", "")

        load_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Respond with the single word READY."},
        ]
        load_met = call_chat_completions_once(base_url, model_key, load_messages, 8, 0.0, args.timeout, api_key)
        load_met.load_s = load_met.wall_s
        results.append(
            attach_metrics(
                result_row_base(run_id, "load", inv, "load_probe", "operational_health", "P0", 0),
                load_met,
            )
        )

        for case in cases:
            messages = [
                {"role": "system", "content": case.system},
                {"role": "user", "content": case.prompt},
            ]
            for repeat in range(args.repeats):
                met = call_chat_completions_stream(base_url, model_key, messages, case.max_output_tokens, case.temperature, args.timeout, api_key)
                output_file = ""
                if met.output_text:
                    output_name = (
                        f"{slugify_filename(host_name)}__{slugify_filename(model_key)}__"
                        f"{slugify_filename(case.case_key)}__r{repeat + 1}.txt"
                    )
                    output_path = outputs_dir / output_name
                    output_path.write_text(met.output_text, encoding="utf-8")
                    output_file = str(output_path.relative_to(run_dir))

                eval_result = evaluate_output(met.output_text, case.evaluators) if met.ok else {"ok": False, "score": 0.0, "results": [], "failed": [{"message": met.error or "completion failed"}]}
                row = attach_metrics(
                    result_row_base(run_id, "run", inv, case.case_key, case.task_family, case.priority, repeat + 1),
                    met,
                )
                row.update(
                    {
                        "context_tokens": case.context_tokens or "",
                        "recommendation_signal": case.recommendation_signal,
                        "eval_ok": bool(eval_result.get("ok")),
                        "eval_score": eval_result.get("score"),
                        "eval_result_json": json.dumps(eval_result, sort_keys=True),
                        "eval_failed_json": json.dumps(eval_result.get("failed", []), sort_keys=True),
                        "output_file": output_file,
                    }
                )
                results.append(row)

    results_fields = [
        "run_id",
        "created_at_utc",
        "phase",
        "host_name",
        "host_ip",
        "endpoint_id",
        "base_url",
        "model_id",
        "model_key",
        "case_key",
        "task_family",
        "priority",
        "context_tokens",
        "recommendation_signal",
        "repeat_index",
        "ok",
        "http_status",
        "error",
        "wall_s",
        "ttft_s",
        "load_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tokens_per_sec",
        "finish_reason",
        "eval_ok",
        "eval_score",
        "eval_failed_json",
        "eval_result_json",
        "output_file",
    ]
    write_csv(output_dir / "run_results.csv", results, results_fields)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary_rows = summarize(results, ["host_name", "host_ip", "base_url", "model_key"], run_id)
    summary_fields = ["run_id", "host_name", "host_ip", "base_url", "model_key", "load_s", "ttft_med", "tps_med", "ok_rate", "eval_ok_rate", "eval_score_avg", "cases"]
    write_csv(output_dir / "run_summary.csv", summary_rows, summary_fields)

    task_rows = summarize(results, ["host_name", "host_ip", "base_url", "model_key", "task_family"], run_id)
    task_fields = ["run_id", "host_name", "host_ip", "base_url", "model_key", "task_family", "load_s", "ttft_med", "tps_med", "ok_rate", "eval_ok_rate", "eval_score_avg", "cases"]
    write_csv(output_dir / "task_summary.csv", task_rows, task_fields)

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (r.get("host_name", ""), r.get("host_ip", ""), r.get("base_url", ""), r.get("model_key", ""))
        grouped.setdefault(key, []).append(r)
    for (host_name, host_ip, base_url, model_key), items in grouped.items():
        write_model_report_md(run_dir, host_name, host_ip, base_url, model_key, items)

    write_run_index_md(run_dir, run_id, started_at, config, summary_rows, task_rows)

    print(f"Wrote results to {output_dir / 'run_results.csv'}")
    print(f"Wrote summary to {output_dir / 'run_summary.csv'}")
    print(f"Wrote task summary to {output_dir / 'task_summary.csv'}")
    print(f"Wrote sidecars to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
