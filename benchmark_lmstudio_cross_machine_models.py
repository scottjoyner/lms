#!/usr/bin/env python3
"""
Benchmark the same model across multiple machines/endpoints using an inventory CSV.

Reads inventory rows with columns:
  - host_name, host_ip, endpoint_id, base_url, reachable, model_id, model_key

Emits:
  - run_results.csv
  - run_summary.csv
  - config.json
  - Markdown sidecar reports per run
  - Full output text files for human evaluation

Install:
  pip install requests

Run:
  python3 benchmark_lmstudio_cross_machine_models.py \
    --inventory-csv lmstudio_inventory.csv \
    --output-dir bench_cross_csv \
    --sidecar-dir sidecar_cross_md \
    --timeout 900 \
    --repeats 1 \
    --stream
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


# ----------------------------
# Benchmark Cases
# ----------------------------
@dataclass
class BenchCase:
    case_key: str
    task_type: str
    prompt: str
    system: str
    max_output_tokens: int
    temperature: float
    notes: str


def build_fibonacci_prompt(language: str) -> str:
    return (
        "Please provide the code to implement the Fibonacci sequence.\n"
        "Print the first 1000 numbers in the sequence.\n"
        "Use arbitrary-precision arithmetic (do not rely on external libraries).\n"
        f"Implement this source code in {language}.\n"
        "Keep the output to code only (no markdown) and include brief comments."
    )


LANG_CASES = [
    ("c", "C programming language"),
    ("rust", "Rust"),
    ("python", "Python"),
    ("javascript", "JavaScript"),
]

DEFAULT_CASES: List[BenchCase] = [
    BenchCase(
        case_key=f"code_fib_{key}",
        task_type="code",
        system="You are a senior software engineer. Output code only.",
        prompt=build_fibonacci_prompt(language),
        max_output_tokens=1600,
        temperature=0.0,
        notes=f"Fibonacci sequence (first 1000 numbers) in {language}.",
    )
    for key, language in LANG_CASES
]


# ----------------------------
# OpenAI-compatible streaming parser
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
    lines.append("| Host | Endpoint | Model | Load s | Median TTFT s | Median TPS | OK Rate | Status |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    for r in summary_rows:
        status = "✅" if r["ok_rate"] >= 0.8 else "⚠️" if r["ok_rate"] > 0 else "❌"
        lines.append(
            f"| `{r['host_name']}` | `{r['base_url']}` | `{r['model_key']}` | "
            f"{r.get('load_s','')} | {r.get('ttft_med','')} | {r.get('tps_med','')} | "
            f"{r.get('ok_rate','')} | {status} |"
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
    lines.append("| Case | Phase | OK | Wall s | TTFT s | TPS | Output File | Error |")
    lines.append("|---|---|:---:|---:|---:|---:|---|---|")

    for it in items:
        lines.append(
            f"| `{it['case_key']}` | `{it['phase']}` | {'✅' if it['ok'] else '❌'} | "
            f"{it.get('wall_s','')} | {it.get('ttft_s','')} | {it.get('tokens_per_sec','')} | "
            f"{it.get('output_file','')} | {(it.get('error') or '')[:80]} |"
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


def load_inventory_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"inventory csv not found: {csv_path}")
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def filter_inventory_rows(
    rows: List[Dict[str, Any]],
    include_endpoints: Optional[set[str]],
    exclude_endpoints: Optional[set[str]],
    include_models: Optional[set[str]],
    exclude_models: Optional[set[str]],
    only_reachable: bool,
) -> List[Dict[str, Any]]:
    def endpoint_matches(token_set: Optional[set[str]], row: Dict[str, Any]) -> bool:
        if not token_set:
            return True
        endpoint_id = str(row.get("endpoint_id", ""))
        base_url = row.get("base_url", "")
        host_name = row.get("host_name", "")
        return any(tok in (endpoint_id, base_url, host_name) for tok in token_set)

    filtered = []
    for r in rows:
        reachable = str(r.get("reachable", "")).lower() in ("1", "true", "yes", "y")
        if only_reachable and not reachable:
            continue
        if include_endpoints and not endpoint_matches(include_endpoints, r):
            continue
        if exclude_endpoints and endpoint_matches(exclude_endpoints, r):
            continue
        if include_models and r.get("model_key") not in include_models:
            continue
        if exclude_models and r.get("model_key") in exclude_models:
            continue
        filtered.append(r)
    return filtered


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-machine benchmark for the same models.")
    ap.add_argument("--inventory-csv", required=True, help="Inventory CSV from LM Studio discovery")
    ap.add_argument("--output-dir", required=True, help="Directory for CSV artifacts")
    ap.add_argument("--sidecar-dir", required=True, help="Directory for Markdown sidecars + outputs")
    ap.add_argument("--timeout", type=float, default=900, help="Request timeout seconds")
    ap.add_argument("--repeats", type=int, default=1, help="Repeats per case")
    ap.add_argument("--stream", action="store_true", default=True, help="Use streaming for TTFT")
    ap.add_argument("--api-key-env", default="LMSTUDIO_API_KEY", help="Env var for API key")
    ap.add_argument("--only-reachable", action="store_true", default=True, help="Only use reachable endpoints")
    ap.add_argument("--include-endpoints", default=None, help="Comma-separated endpoint IDs/base URLs/host names")
    ap.add_argument("--exclude-endpoints", default=None, help="Comma-separated endpoint IDs/base URLs/host names")
    ap.add_argument("--include-models", default=None, help="Comma-separated model keys to include")
    ap.add_argument("--exclude-models", default=None, help="Comma-separated model keys to exclude")
    args = ap.parse_args()

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

    rows = load_inventory_rows(inventory_csv)
    rows = filter_inventory_rows(
        rows,
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
        "timeout_s": args.timeout,
        "repeats": args.repeats,
        "stream": args.stream,
        "cases": [case.__dict__ for case in DEFAULT_CASES],
        "filters": {
            "only_reachable": args.only_reachable,
            "include_endpoints": sorted(include_endpoints) if include_endpoints else None,
            "exclude_endpoints": sorted(exclude_endpoints) if exclude_endpoints else None,
            "include_models": sorted(include_models) if include_models else None,
            "exclude_models": sorted(exclude_models) if exclude_models else None,
        },
    }

    results: List[Dict[str, Any]] = []

    for row in rows:
        host_name = row.get("host_name", "")
        host_ip = row.get("host_ip", "")
        base_url = row.get("base_url", "").rstrip("/")
        model_key = row.get("model_key", "")

        load_prompt = "Respond with the single word READY."
        load_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": load_prompt},
        ]

        load_met = call_chat_completions_once(
            base_url=base_url,
            model=model_key,
            messages=load_messages,
            max_tokens=8,
            temperature=0.0,
            timeout_s=args.timeout,
            api_key=api_key,
        )
        load_met.load_s = load_met.wall_s

        results.append(
            {
                "run_id": run_id,
                "phase": "load",
                "host_name": host_name,
                "host_ip": host_ip,
                "base_url": base_url,
                "model_key": model_key,
                "case_key": "load_probe",
                "repeat_index": 0,
                "ok": load_met.ok,
                "http_status": load_met.http_status,
                "error": load_met.error,
                "wall_s": f"{load_met.wall_s:.3f}",
                "ttft_s": "",
                "load_s": f"{load_met.load_s:.3f}",
                "prompt_tokens": load_met.prompt_tokens,
                "completion_tokens": load_met.completion_tokens,
                "total_tokens": load_met.total_tokens,
                "tokens_per_sec": f"{load_met.tokens_per_sec:.3f}" if load_met.tokens_per_sec else "",
                "finish_reason": load_met.finish_reason,
                "output_text": load_met.output_text,
                "output_file": "",
            }
        )

        for case in DEFAULT_CASES:
            messages = [
                {"role": "system", "content": case.system},
                {"role": "user", "content": case.prompt},
            ]
            for repeat in range(args.repeats):
                met = call_chat_completions_stream(
                    base_url=base_url,
                    model=model_key,
                    messages=messages,
                    max_tokens=case.max_output_tokens,
                    temperature=case.temperature,
                    timeout_s=args.timeout,
                    api_key=api_key,
                )
                output_file = ""
                if met.output_text:
                    output_name = (
                        f"{slugify_filename(host_name)}__{slugify_filename(model_key)}__"
                        f"{case.case_key}__r{repeat + 1}.txt"
                    )
                    output_path = outputs_dir / output_name
                    output_path.write_text(met.output_text, encoding="utf-8")
                    output_file = str(output_path.relative_to(run_dir))

                results.append(
                    {
                        "run_id": run_id,
                        "phase": "run",
                        "host_name": host_name,
                        "host_ip": host_ip,
                        "base_url": base_url,
                        "model_key": model_key,
                        "case_key": case.case_key,
                        "repeat_index": repeat + 1,
                        "ok": met.ok,
                        "http_status": met.http_status,
                        "error": met.error,
                        "wall_s": f"{met.wall_s:.3f}",
                        "ttft_s": f"{met.ttft_s:.3f}" if met.ttft_s is not None else "",
                        "load_s": "",
                        "prompt_tokens": met.prompt_tokens,
                        "completion_tokens": met.completion_tokens,
                        "total_tokens": met.total_tokens,
                        "tokens_per_sec": f"{met.tokens_per_sec:.3f}" if met.tokens_per_sec else "",
                        "finish_reason": met.finish_reason,
                        "output_text": met.output_text,
                        "output_file": output_file,
                    }
                )

    output_csv = output_dir / "run_results.csv"
    summary_csv = output_dir / "run_summary.csv"
    config_json = output_dir / "config.json"

    results_fields = [
        "run_id",
        "phase",
        "host_name",
        "host_ip",
        "base_url",
        "model_key",
        "case_key",
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
        "output_file",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in results_fields}
            writer.writerow(row)

    config_json.write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary_rows: List[Dict[str, Any]] = []
    summary_fields = [
        "run_id",
        "host_name",
        "host_ip",
        "base_url",
        "model_key",
        "load_s",
        "ttft_med",
        "tps_med",
        "ok_rate",
    ]

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (r["host_name"], r["host_ip"], r["base_url"], r["model_key"])
        grouped.setdefault(key, []).append(r)

    for (host_name, host_ip, base_url, model_key), items in grouped.items():
        run_items = [it for it in items if it["phase"] == "run"]
        load_items = [it for it in items if it["phase"] == "load"]
        load_vals = [
            float(it["load_s"]) for it in load_items if it.get("load_s")
        ]
        ttft_vals = [
            float(it["ttft_s"]) for it in run_items if it.get("ttft_s")
        ]
        tps_vals = [
            float(it["tokens_per_sec"]) for it in run_items if it.get("tokens_per_sec")
        ]
        ok_rate_val = ok_rate(run_items)
        summary_rows.append(
            {
                "run_id": run_id,
                "host_name": host_name,
                "host_ip": host_ip,
                "base_url": base_url,
                "model_key": model_key,
                "load_s": mean_or_blank(load_vals),
                "ttft_med": median_or_blank(ttft_vals),
                "tps_med": median_or_blank(tps_vals),
                "ok_rate": f"{ok_rate_val:.2f}",
            }
        )

        write_model_report_md(
            run_dir=run_dir,
            host_name=host_name,
            host_ip=host_ip,
            base_url=base_url,
            model_key=model_key,
            items=items,
        )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    write_run_index_md(
        run_dir=run_dir,
        run_id=run_id,
        started_at=started_at,
        config=config,
        summary_rows=summary_rows,
    )

    print(f"Wrote results to {output_csv}")
    print(f"Wrote summary to {summary_csv}")
    print(f"Wrote sidecars to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
