#!/usr/bin/env python3
"""Manifest-aware LMS agent CLI.

This is the active installed CLI entrypoint. It keeps the agent interface simple:

  lms doctor
  lms probe
  lms quick
  lms runs
  lms show latest
  lms route latest --task coding
  lms compare runs/a runs/b

The CLI intentionally requires no config file for the common path.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "http://127.0.0.1:1234/v1"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_SUITE = Path("benchmarks") / "agent_skill_suite.v1.json"
BENCHMARK_SCRIPT = Path("benchmark_lmstudio_cross_machine_models.py")
PROFILE_SCRIPT = Path("lms_machine_profile.py")
EVAL_SCRIPT = Path("lms_eval.py")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    out = []
    for ch in value.lower().strip():
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "-")
    return "".join(out).strip("-") or "run"


def resolve_asset(path: Path) -> Path:
    for candidate in [REPO_ROOT / path, Path.cwd() / path, Path.cwd().parent / path]:
        if candidate.exists():
            return candidate
    return REPO_ROOT / path


def default_endpoint() -> str:
    return os.environ.get("LMS_BASE_URL") or os.environ.get("LMSTUDIO_BASE_URL") or DEFAULT_ENDPOINT


def normalize_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url:
        return url
    return url if url.endswith("/v1") else f"{url}/v1"


def parse_csv_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str], strict: bool = False) -> int:
    print("$ " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(list(cmd), check=False)
    if strict and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def http_get_json(url: str, timeout: int = 8) -> Tuple[Optional[Any], Optional[str], Optional[int], float]:
    started = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(5_000_000).decode("utf-8", errors="replace")
            return json.loads(raw), None, getattr(resp, "status", None), time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, repr(exc), None, time.perf_counter() - started


def probe_endpoint(base_url: str, timeout: int = 8) -> Dict[str, Any]:
    base_url = normalize_base_url(base_url)
    data, error, status, elapsed = http_get_json(f"{base_url}/models", timeout=timeout)
    models: List[str] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        models = [str(m["id"]) for m in data["data"] if isinstance(m, dict) and m.get("id")]
    return {
        "base_url": base_url,
        "reachable": error is None,
        "status": status,
        "elapsed_s": round(elapsed, 4),
        "model_count": len(models),
        "models": models,
        "error": error,
    }


def probe_endpoints(endpoints: Sequence[str], timeout: int = 8) -> List[Dict[str, Any]]:
    return [probe_endpoint(endpoint, timeout=timeout) for endpoint in endpoints]


def local_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def write_inventory_csv(probes: Sequence[Dict[str, Any]], out: Path, models: Optional[str] = None, max_models: int = 0) -> List[Dict[str, Any]]:
    include = set(parse_csv_arg(models))
    rows: List[Dict[str, Any]] = []
    endpoint_id = 1
    model_id = 1
    for probe in probes:
        probe_models = list(probe.get("models") or [])
        if include:
            probe_models = [model for model in probe_models if model in include]
        if max_models > 0:
            probe_models = probe_models[:max_models]
        for model_key in probe_models:
            rows.append(
                {
                    "host_name": socket.gethostname(),
                    "host_ip": local_host_ip(),
                    "endpoint_id": endpoint_id,
                    "base_url": probe["base_url"],
                    "reachable": 1 if probe.get("reachable") else 0,
                    "model_id": model_id,
                    "model_key": model_key,
                }
            )
            model_id += 1
        endpoint_id += 1
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["host_name", "host_ip", "endpoint_id", "base_url", "reachable", "model_id", "model_key"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def grade(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.55:
        return "C"
    if score > 0:
        return "D"
    return "F"


def latency_grade(ttft: float) -> str:
    if ttft <= 0:
        return "unknown"
    if ttft <= 2:
        return "A"
    if ttft <= 5:
        return "B"
    if ttft <= 12:
        return "C"
    if ttft <= 25:
        return "D"
    return "F"


def throughput_grade(tps: float) -> str:
    if tps >= 40:
        return "A"
    if tps >= 20:
        return "B"
    if tps >= 8:
        return "C"
    if tps > 0:
        return "D"
    return "unknown"


def score_row(row: Dict[str, str]) -> float:
    ok = float_or_zero(row.get("ok_rate"))
    eval_ok = float_or_zero(row.get("eval_ok_rate"))
    eval_score = float_or_zero(row.get("eval_score_avg"))
    tps = float_or_zero(row.get("tps_med"))
    ttft = float_or_zero(row.get("ttft_med"))
    quality = max(eval_score, eval_ok)
    return round(ok * 0.35 + quality * 0.45 + min(tps / 40.0, 1.0) * 0.15 + (0.05 if ttft and ttft <= 8 else 0.0), 4)


def discover_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    runs = []
    for path in runs_dir.iterdir():
        if path.is_dir() and any((path / marker).exists() for marker in ["lms_run_config.json", "run_summary.csv", "task_summary.csv", "machine_profile.json"]):
            runs.append(path)
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_run_dir(value: str, runs_dir: str) -> Path:
    if value != "latest":
        return Path(value)
    runs = discover_runs(Path(runs_dir))
    if not runs:
        raise SystemExit(f"no runs found under {runs_dir}")
    return runs[0]


def print_table(rows: List[List[str]], headers: List[str]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def recommendation_text(score: float) -> str:
    if score >= 0.85:
        return "preferred for this task family"
    if score >= 0.70:
        return "usable with review"
    if score >= 0.50:
        return "draft-only; require stronger review"
    return "avoid for this task family"


def avoid_text(score: float, row: Dict[str, str]) -> str:
    if float_or_zero(row.get("ok_rate")) == 0:
        return "endpoint/model did not complete benchmark cases"
    if score < 0.50:
        return "avoid autonomous work and route to fallback"
    if float_or_zero(row.get("tps_med")) < 4:
        return "avoid interactive or large-output workflows"
    return "avoid high-risk work unless task-specific checks pass"


def load_task_or_general_rows(run_dir: Path, task: Optional[str] = None) -> List[Dict[str, str]]:
    task_rows = read_csv(run_dir / "task_summary.csv")
    if task_rows:
        if task and task != "general":
            filtered = [r for r in task_rows if r.get("task_family") == task]
            if filtered:
                return filtered
        if task == "general":
            general_summary = read_csv(run_dir / "run_summary.csv")
            for row in general_summary:
                row.setdefault("task_family", "general")
            return general_summary or task_rows
        return task_rows
    rows = read_csv(run_dir / "run_summary.csv")
    for row in rows:
        row.setdefault("task_family", "general")
    return rows


def reliable_context_by_route(run_dir: Path) -> Dict[Tuple[str, str, str], int]:
    results = read_csv(run_dir / "run_results.csv")
    best: Dict[Tuple[str, str, str], int] = {}
    for row in results:
        context = int(float_or_zero(row.get("context_tokens")))
        if context <= 0:
            continue
        eval_ok = str(row.get("eval_ok", "")).lower() in {"true", "1", "yes"}
        ok = str(row.get("ok", "")).lower() in {"true", "1", "yes"}
        if ok and eval_ok:
            key = (row.get("model_key", ""), row.get("task_family", ""), row.get("base_url", ""))
            best[key] = max(best.get(key, 0), context)
    return best


def route_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (row.get("task_family", "general"), row.get("model_key", ""), row.get("base_url", ""), row.get("host_name", ""))


def synthesize_recommendations(run_dir: Path) -> None:
    rows = load_task_or_general_rows(run_dir)
    reliable_context = reliable_context_by_route(run_dir)
    capability_rows: List[Dict[str, Any]] = []
    for row in rows:
        score = score_row(row)
        max_ctx = reliable_context.get((row.get("model_key", ""), row.get("task_family", "general"), row.get("base_url", "")), 0)
        capability_rows.append(
            {
                "run_id": row.get("run_id", ""),
                "host_name": row.get("host_name", ""),
                "host_ip": row.get("host_ip", ""),
                "base_url": row.get("base_url", ""),
                "model_key": row.get("model_key", ""),
                "context_tokens": row.get("context_tokens", ""),
                "max_reliable_context_tokens": max_ctx or "",
                "task_family": row.get("task_family", "general"),
                "score": f"{score:.4f}",
                "grade": grade(score),
                "latency_grade": latency_grade(float_or_zero(row.get("ttft_med"))),
                "throughput_grade": throughput_grade(float_or_zero(row.get("tps_med"))),
                "reliability_grade": grade(max(float_or_zero(row.get("ok_rate")), float_or_zero(row.get("eval_ok_rate")))),
                "recommended_use": recommendation_text(score),
                "avoid_use": avoid_text(score, row),
                "evidence": f"task={row.get('task_family','general')}; ok_rate={row.get('ok_rate','')}; eval_ok_rate={row.get('eval_ok_rate','')}; eval_score={row.get('eval_score_avg','')}; ttft={row.get('ttft_med','')}; tps={row.get('tps_med','')}; max_ctx={max_ctx or ''}",
                "notes": "Generated from task_summary.csv when available, otherwise run_summary.csv.",
            }
        )
    fields = ["run_id", "host_name", "host_ip", "base_url", "model_key", "context_tokens", "max_reliable_context_tokens", "task_family", "score", "grade", "latency_grade", "throughput_grade", "reliability_grade", "recommended_use", "avoid_use", "evidence", "notes"]
    write_csv(run_dir / "capability_matrix.csv", capability_rows, fields)

    ranked = sorted(capability_rows, key=lambda r: float_or_zero(r.get("score")), reverse=True)
    lines = ["# LMS Agent Recommendations", "", f"- Generated UTC: `{utc_now_iso()}`", f"- Run directory: `{run_dir}`", ""]
    profile = read_json(run_dir / "machine_profile.json")
    if profile and profile.get("recommendations"):
        lines += ["## Machine synopsis", ""] + [f"- {r}" for r in profile.get("recommendations", [])] + [""]
    lines += ["## Task-specific routing", ""]
    if ranked:
        lines += ["| Task | Host | Model | Score | Grade | Max reliable context | Evidence |", "|---|---|---|---:|---|---:|---|"]
        for row in ranked[:40]:
            lines.append(f"| `{row.get('task_family')}` | `{row.get('host_name')}` | `{row.get('model_key')}` | {row.get('score')} | {row.get('grade')} | {row.get('max_reliable_context_tokens','')} | {row.get('evidence')} |")
    else:
        lines.append("No benchmark rows found. Run `lms quick` first.")
    lines += ["", "## Operating rules", "", "- Prefer task-family routes over general routes.", "- Use fallback routes when the preferred route is below threshold or unavailable.", "- Fall back to a stronger model when deterministic evaluator scores are low.", "- Treat routing as evidence-based guidance, not a guarantee.", ""]
    (run_dir / "agent_recommendations.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {run_dir / 'capability_matrix.csv'}")
    print(f"wrote {run_dir / 'agent_recommendations.md'}")


def sorted_routes(run_dir: Path, task: str) -> List[Dict[str, str]]:
    cap = read_csv(run_dir / "capability_matrix.csv")
    if not cap:
        synthesize_recommendations(run_dir)
        cap = read_csv(run_dir / "capability_matrix.csv")
    rows = [r for r in cap if r.get("task_family") == task]
    if not rows and task != "general":
        rows = [r for r in cap if r.get("task_family") == "general"]
    rows = rows or cap
    rows.sort(key=lambda r: float_or_zero(r.get("score")), reverse=True)
    return rows


def choose_route(run_dir: Path, task: str) -> Optional[Dict[str, str]]:
    rows = sorted_routes(run_dir, task)
    return rows[0] if rows else None


def choose_route_with_fallback(run_dir: Path, task: str) -> Optional[Dict[str, Any]]:
    rows = sorted_routes(run_dir, task)
    if not rows:
        return None
    preferred = rows[0]
    fallback = None
    for row in rows[1:]:
        if row.get("model_key") != preferred.get("model_key") or row.get("base_url") != preferred.get("base_url"):
            fallback = row
            break
    return {"preferred": preferred, "fallback": fallback, "task_family": task}


def render_route_yaml(route_bundle: Dict[str, Any]) -> str:
    preferred = route_bundle.get("preferred") or route_bundle
    fallback = route_bundle.get("fallback")
    task = route_bundle.get("task_family") or preferred.get("task_family", "general")
    lines = [
        "routing:",
        f"  {task}:",
        f"    preferred_model: {json.dumps(preferred.get('model_key', ''))}",
        f"    preferred_base_url: {json.dumps(preferred.get('base_url', ''))}",
        f"    preferred_host_name: {json.dumps(preferred.get('host_name', ''))}",
        f"    preferred_score: {json.dumps(preferred.get('score', ''))}",
        f"    preferred_grade: {json.dumps(preferred.get('grade', ''))}",
    ]
    if fallback:
        lines += [
            f"    fallback_model: {json.dumps(fallback.get('model_key', ''))}",
            f"    fallback_base_url: {json.dumps(fallback.get('base_url', ''))}",
            f"    fallback_host_name: {json.dumps(fallback.get('host_name', ''))}",
            f"    fallback_score: {json.dumps(fallback.get('score', ''))}",
            f"    fallback_grade: {json.dumps(fallback.get('grade', ''))}",
        ]
    else:
        lines += ["    fallback_model: null", "    fallback_base_url: null"]
    lines += [
        f"    max_reliable_context_tokens: {json.dumps(preferred.get('max_reliable_context_tokens', ''))}",
        f"    evidence: {json.dumps(preferred.get('evidence', ''))}",
        f"    recommended_use: {json.dumps(preferred.get('recommended_use', ''))}",
        f"    avoid_use: {json.dumps(preferred.get('avoid_use', ''))}",
        "    source: lms capability_matrix.csv",
        "",
    ]
    return "\n".join(lines)


def compare_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (row.get("task_family", "general"), row.get("model_key", ""), row.get("base_url", ""))


def compare_runs(run_a: Path, run_b: Path, out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    if not (run_a / "capability_matrix.csv").exists():
        synthesize_recommendations(run_a)
    if not (run_b / "capability_matrix.csv").exists():
        synthesize_recommendations(run_b)
    rows_a = {compare_key(r): r for r in read_csv(run_a / "capability_matrix.csv")}
    rows_b = {compare_key(r): r for r in read_csv(run_b / "capability_matrix.csv")}
    keys = sorted(set(rows_a) | set(rows_b))
    deltas: List[Dict[str, Any]] = []
    for key in keys:
        a = rows_a.get(key, {})
        b = rows_b.get(key, {})
        score_a = float_or_zero(a.get("score"))
        score_b = float_or_zero(b.get("score"))
        deltas.append(
            {
                "task_family": key[0],
                "model_key": key[1],
                "base_url": key[2],
                "score_a": f"{score_a:.4f}" if a else "",
                "score_b": f"{score_b:.4f}" if b else "",
                "score_delta": f"{score_b - score_a:.4f}" if a and b else "",
                "grade_a": a.get("grade", ""),
                "grade_b": b.get("grade", ""),
                "status": "added" if not a else "removed" if not b else "improved" if score_b > score_a else "regressed" if score_b < score_a else "unchanged",
                "evidence_a": a.get("evidence", ""),
                "evidence_b": b.get("evidence", ""),
            }
        )
    out_dir = out_dir or (run_b / "comparisons" / safe_slug(run_a.name + "_vs_" + run_b.name))
    ensure_dir(out_dir)
    csv_path = out_dir / "compare_delta.csv"
    md_path = out_dir / "compare_summary.md"
    fields = ["task_family", "model_key", "base_url", "score_a", "score_b", "score_delta", "grade_a", "grade_b", "status", "evidence_a", "evidence_b"]
    write_csv(csv_path, deltas, fields)

    lines = ["# LMS Run Comparison", "", f"- Run A: `{run_a}`", f"- Run B: `{run_b}`", f"- Generated UTC: `{utc_now_iso()}`", ""]
    lines += ["## Summary", ""]
    for status in ["added", "removed", "improved", "regressed", "unchanged"]:
        count = sum(1 for row in deltas if row["status"] == status)
        lines.append(f"- {status}: {count}")
    lines += ["", "## Largest changes", "", "| Status | Task | Model | Score A | Score B | Delta |", "|---|---|---|---:|---:|---:|"]
    sortable = sorted(deltas, key=lambda r: abs(float_or_zero(r.get("score_delta"))), reverse=True)
    for row in sortable[:30]:
        lines.append(f"| {row['status']} | `{row['task_family']}` | `{row['model_key']}` | {row['score_a']} | {row['score_b']} | {row['score_delta']} |")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def cmd_doctor(args: argparse.Namespace) -> int:
    print("LMS doctor")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    for path in [PROFILE_SCRIPT, EVAL_SCRIPT, BENCHMARK_SCRIPT, DEFAULT_SUITE]:
        resolved = resolve_asset(path)
        print(f"{'OK ' if resolved.exists() else 'MISS'} {resolved}")
    probes = probe_endpoints(args.endpoint or [default_endpoint()], timeout=args.timeout)
    for probe in probes:
        print(f"{'OK' if probe['reachable'] else 'FAIL'} {probe['base_url']} models={probe['model_count']} error={probe['error'] or ''}")
    return 0 if all(p["reachable"] for p in probes) else 1


def cmd_probe(args: argparse.Namespace) -> int:
    probes = probe_endpoints(args.endpoint or [default_endpoint()], timeout=args.timeout)
    if args.json:
        print(json.dumps(probes, indent=2))
    else:
        for probe in probes:
            print(f"{probe['base_url']} — {'reachable' if probe['reachable'] else 'unreachable'} — {probe['model_count']} model(s)")
            if probe.get("error"):
                print(f"  error: {probe['error']}")
            for model in probe.get("models", []):
                print(f"  - {model}")
    return 0 if all(p["reachable"] for p in probes) else 1


def cmd_inventory(args: argparse.Namespace) -> int:
    probes = probe_endpoints(args.endpoint or [default_endpoint()], timeout=args.timeout)
    rows = write_inventory_csv(probes, Path(args.out), models=args.models, max_models=args.max_models)
    print(f"wrote {args.out} with {len(rows)} model row(s)")
    return 0 if rows else 1


def cmd_profile(args: argparse.Namespace) -> int:
    script = resolve_asset(PROFILE_SCRIPT)
    cmd = [sys.executable, str(script), "--output-dir", args.output_dir, "--timeout", str(args.timeout)]
    if args.inventory_csv:
        cmd += ["--inventory-csv", args.inventory_csv]
    for endpoint in args.endpoint or [default_endpoint()]:
        cmd += ["--probe-base-url", endpoint]
    return run_cmd(cmd)


def cmd_quick(args: argparse.Namespace) -> int:
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / safe_slug(run_id)
    ensure_dir(run_dir)
    endpoints = args.endpoint or [default_endpoint()]
    suite = Path(args.suite_file) if args.suite_file else resolve_asset(DEFAULT_SUITE)
    inventory = run_dir / "lmstudio_inventory.csv"

    print(f"Creating LMS run: {run_dir}")
    probes = probe_endpoints(endpoints, timeout=args.timeout)
    (run_dir / "endpoint_probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    rows = write_inventory_csv(probes, inventory, models=args.models, max_models=args.max_models)
    print(f"Inventory rows: {len(rows)}")

    config = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "endpoints": [normalize_base_url(e) for e in endpoints],
        "inventory_csv": str(inventory),
        "suite_file": str(suite),
        "models": parse_csv_arg(args.models),
        "exclude_models": parse_csv_arg(args.exclude_models),
        "max_models": args.max_models,
        "repeats": args.repeats,
        "timeout": args.timeout,
        "max_context_tokens": args.max_context_tokens,
    }
    (run_dir / "lms_run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if suite.exists():
        shutil.copy2(suite, run_dir / suite.name)

    profile_code = cmd_profile(argparse.Namespace(output_dir=str(run_dir), timeout=args.timeout, inventory_csv=str(inventory), endpoint=endpoints))
    if profile_code != 0 and args.strict:
        return profile_code
    if args.profile_only or not rows:
        print(f"Run directory ready: {run_dir}")
        return 0 if rows else 1

    bench = resolve_asset(BENCHMARK_SCRIPT)
    bench_cmd = [sys.executable, str(bench), "--inventory-csv", str(inventory), "--cases-file", str(suite), "--output-dir", str(run_dir), "--sidecar-dir", str(run_dir / "sidecars"), "--timeout", str(args.timeout), "--repeats", str(args.repeats), "--max-context-tokens", str(args.max_context_tokens)]
    if args.models:
        bench_cmd += ["--include-models", args.models]
    if args.exclude_models:
        bench_cmd += ["--exclude-models", args.exclude_models]
    bench_code = run_cmd(bench_cmd)
    if bench_code != 0 and args.strict:
        return bench_code

    synthesize_recommendations(run_dir)
    routes: Dict[str, Dict[str, Any]] = {}
    for task in ["general", "coding", "debugging", "agent_planning", "structured_output", "long_context", "repo_work", "operational_health", "safety"]:
        route = choose_route_with_fallback(run_dir, task)
        if route:
            routes[task] = route
    (run_dir / "routing_rules.json").write_text(json.dumps({"routing": routes}, indent=2), encoding="utf-8")
    (run_dir / "routing_rules.yaml").write_text("".join(render_route_yaml(route) for route in routes.values()), encoding="utf-8")
    print(f"wrote {run_dir / 'routing_rules.yaml'}")
    print(f"Done. Read {run_dir / 'agent_recommendations.md'}")
    return bench_code


def cmd_runs(args: argparse.Namespace) -> int:
    runs = discover_runs(Path(args.runs_dir))[: args.limit]
    if not runs:
        print(f"No runs found under {args.runs_dir}")
        return 1
    table = []
    for run in runs:
        status = "recommended" if (run / "agent_recommendations.md").exists() else "benchmarked" if (run / "run_summary.csv").exists() else "profiled"
        table.append([run.name, status, dt.datetime.fromtimestamp(run.stat().st_mtime).isoformat(timespec="seconds"), str(run)])
    print_table(table, ["run_id", "status", "modified", "path"])
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    print(f"Run: {run_dir}")
    cfg = read_json(run_dir / "lms_run_config.json") or {}
    if cfg:
        print(f"Created: {cfg.get('created_at_utc')}")
        print(f"Endpoints: {', '.join(cfg.get('endpoints', []))}")
    rows = load_task_or_general_rows(run_dir, args.task)
    ranked = sorted(rows, key=score_row, reverse=True)[: args.limit]
    if ranked:
        table = [[r.get("task_family", "general"), r.get("host_name", ""), r.get("model_key", ""), r.get("ok_rate", ""), r.get("eval_ok_rate", ""), r.get("eval_score_avg", ""), r.get("tps_med", ""), f"{score_row(r):.3f}"] for r in ranked]
        print_table(table, ["task", "host", "model", "ok", "eval_ok", "eval", "tps", "score"])
    route = choose_route_with_fallback(run_dir, args.task)
    if route:
        print("\nSelected route")
        print(render_route_yaml(route), end="")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    synthesize_recommendations(run_dir)
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    route = choose_route_with_fallback(run_dir, args.task)
    if not route:
        print("No route available. Run `lms quick` first.")
        return 1
    if args.json:
        print(json.dumps(route, indent=2))
    else:
        print(render_route_yaml(route), end="")
    if args.write:
        (run_dir / "routing_rules.json").write_text(json.dumps({"routing": {args.task: route}}, indent=2), encoding="utf-8")
        (run_dir / "routing_rules.yaml").write_text(render_route_yaml(route), encoding="utf-8")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    run_a = resolve_run_dir(args.run_a, args.runs_dir)
    run_b = resolve_run_dir(args.run_b, args.runs_dir)
    out_dir = Path(args.output_dir) if args.output_dir else None
    csv_path, md_path = compare_runs(run_a, run_b, out_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    if args.show:
        print(md_path.read_text(encoding="utf-8"))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    script = resolve_asset(EVAL_SCRIPT)
    cmd = [sys.executable, str(script)]
    if args.output_file:
        cmd += ["--output-file", args.output_file]
    if args.evaluators_json:
        cmd += ["--evaluators-json", args.evaluators_json]
    if args.evaluators_file:
        cmd += ["--evaluators-file", args.evaluators_file]
    if args.pretty:
        cmd += ["--pretty"]
    return run_cmd(cmd)


def add_endpoint_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", action="append", default=[], help="LM Studio OpenAI-compatible base URL. Repeat for multiple.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms", description="Agent CLI for profiling and benchmarking LM Studio nodes.")
    parser.add_argument("--version", action="version", version="lms-agent-cli 0.4.0")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check scripts and endpoint reachability")
    add_endpoint_arg(doctor)
    doctor.add_argument("--timeout", type=int, default=8)
    doctor.set_defaults(func=cmd_doctor)

    probe = sub.add_parser("probe", help="List endpoint models")
    add_endpoint_arg(probe)
    probe.add_argument("--timeout", type=int, default=8)
    probe.add_argument("--json", action="store_true")
    probe.set_defaults(func=cmd_probe)

    inventory = sub.add_parser("inventory", help="Create benchmark inventory CSV")
    add_endpoint_arg(inventory)
    inventory.add_argument("--out", default="lmstudio_inventory.csv")
    inventory.add_argument("--models", default=None)
    inventory.add_argument("--max-models", type=int, default=0)
    inventory.add_argument("--timeout", type=int, default=8)
    inventory.set_defaults(func=cmd_inventory)

    profile = sub.add_parser("profile", help="Collect machine profile")
    add_endpoint_arg(profile)
    profile.add_argument("--output-dir", default="runs/profile")
    profile.add_argument("--inventory-csv", default=None)
    profile.add_argument("--timeout", type=int, default=8)
    profile.set_defaults(func=cmd_profile)

    quick = sub.add_parser("quick", help="One-command profile + manifest benchmark + recommendations")
    add_endpoint_arg(quick)
    quick.add_argument("--output-dir", default=DEFAULT_RUNS_DIR)
    quick.add_argument("--run-id", default=None)
    quick.add_argument("--suite-file", default=None)
    quick.add_argument("--models", default=None)
    quick.add_argument("--exclude-models", default=None)
    quick.add_argument("--max-models", type=int, default=3)
    quick.add_argument("--repeats", type=int, default=1)
    quick.add_argument("--timeout", type=int, default=900)
    quick.add_argument("--max-context-tokens", type=int, default=8192)
    quick.add_argument("--profile-only", action="store_true")
    quick.add_argument("--strict", action="store_true")
    quick.set_defaults(func=cmd_quick)

    runs = sub.add_parser("runs", help="List known runs")
    runs.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=cmd_runs)

    show = sub.add_parser("show", help="Show a run summary")
    show.add_argument("run_dir", nargs="?", default="latest")
    show.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    show.add_argument("--task", default="general")
    show.add_argument("--limit", type=int, default=10)
    show.set_defaults(func=cmd_show)

    recommend = sub.add_parser("recommend", help="Regenerate recommendations")
    recommend.add_argument("run_dir", nargs="?", default="latest")
    recommend.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    recommend.set_defaults(func=cmd_recommend)

    route = sub.add_parser("route", help="Print/export selected route with fallback")
    route.add_argument("run_dir", nargs="?", default="latest")
    route.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    route.add_argument("--task", default="general")
    route.add_argument("--json", action="store_true")
    route.add_argument("--write", action="store_true")
    route.set_defaults(func=cmd_route)

    compare = sub.add_parser("compare", help="Compare two LMS run directories")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    compare.add_argument("--output-dir", default=None)
    compare.add_argument("--show", action="store_true")
    compare.set_defaults(func=cmd_compare)

    evaluate = sub.add_parser("eval", help="Run deterministic evaluators")
    evaluate.add_argument("--output-file", default=None)
    evaluate.add_argument("--evaluators-json", default=None)
    evaluate.add_argument("--evaluators-file", default=None)
    evaluate.add_argument("--pretty", action="store_true")
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
