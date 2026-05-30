#!/usr/bin/env python3
"""
LMS agent CLI.

A small, dependency-light command line wrapper that lets agents use the LMS
benchmarking toolkit without hand-editing configs.

Common usage:
  python3 lms.py doctor
  python3 lms.py quick
  python3 lms.py quick --endpoint http://100.64.0.10:1234/v1 --repeats 1
  python3 lms.py inventory --endpoint http://127.0.0.1:1234/v1 --out lmstudio_inventory.csv
  python3 lms.py profile --endpoint http://127.0.0.1:1234/v1
  python3 lms.py recommend runs/<run_id>
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = "http://127.0.0.1:1234/v1"
DEFAULT_SUITE = REPO_ROOT / "benchmarks" / "agent_skill_suite.v1.json"
BENCHMARK_SCRIPT = REPO_ROOT / "benchmark_lmstudio_cross_machine_models.py"
PROFILE_SCRIPT = REPO_ROOT / "lms_machine_profile.py"
EVAL_SCRIPT = REPO_ROOT / "lms_eval.py"


# ----------------------------
# Basic helpers
# ----------------------------
def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_slug(value: str) -> str:
    out = []
    for ch in value.lower().strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "run"


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


def run_subprocess(cmd: Sequence[str], *, check: bool = False) -> int:
    print("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.run(list(cmd), check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return int(proc.returncode)


def http_get_json(url: str, timeout_s: int = 8) -> Tuple[Optional[Any], Optional[str], Optional[int], float]:
    started = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read(5_000_000).decode("utf-8", errors="replace")
            return json.loads(raw), None, getattr(resp, "status", None), time.perf_counter() - started
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return None, repr(exc), None, time.perf_counter() - started


# ----------------------------
# LM Studio probing / inventory
# ----------------------------
def probe_endpoint(base_url: str, timeout_s: int = 8) -> Dict[str, Any]:
    base_url = normalize_base_url(base_url)
    models_url = f"{base_url}/models"
    data, error, status, elapsed_s = http_get_json(models_url, timeout_s=timeout_s)
    models: List[str] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
    return {
        "base_url": base_url,
        "models_url": models_url,
        "reachable": error is None,
        "status": status,
        "elapsed_s": round(elapsed_s, 4),
        "model_count": len(models),
        "models": models,
        "error": error,
    }


def probe_endpoints(endpoints: Sequence[str], timeout_s: int = 8) -> List[Dict[str, Any]]:
    return [probe_endpoint(endpoint, timeout_s=timeout_s) for endpoint in endpoints]


def local_host_ip() -> str:
    # Best effort only. This avoids network calls; it asks the local socket stack
    # which address would be used for outbound traffic.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def write_inventory_csv(
    probes: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    host_name: Optional[str] = None,
    host_ip: Optional[str] = None,
    model_filter: Optional[Sequence[str]] = None,
    max_models: int = 0,
) -> List[Dict[str, Any]]:
    host_name = host_name or socket.gethostname()
    host_ip = host_ip or local_host_ip()
    include_models = set(model_filter or [])
    rows: List[Dict[str, Any]] = []
    endpoint_id = 1
    model_id = 1

    for probe in probes:
        base_url = probe["base_url"]
        models = list(probe.get("models") or [])
        if include_models:
            models = [m for m in models if m in include_models]
        if max_models > 0:
            models = models[:max_models]
        if not models and probe.get("reachable"):
            # A reachable endpoint with no model list is useful for diagnostics,
            # but not usable by the benchmark script. Do not emit a broken row.
            continue
        for model_key in models:
            rows.append(
                {
                    "host_name": host_name,
                    "host_ip": host_ip,
                    "endpoint_id": endpoint_id,
                    "base_url": base_url,
                    "reachable": 1 if probe.get("reachable") else 0,
                    "model_id": model_id,
                    "model_key": model_key,
                }
            )
            model_id += 1
        endpoint_id += 1

    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["host_name", "host_ip", "endpoint_id", "base_url", "reachable", "model_id", "model_key"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ----------------------------
# Recommendations
# ----------------------------
def float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def grade_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.90:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.55:
        return "C"
    if score > 0:
        return "D"
    return "F"


def grade_latency(ttft_s: Optional[float]) -> str:
    if ttft_s is None:
        return "unknown"
    if ttft_s <= 2:
        return "A"
    if ttft_s <= 5:
        return "B"
    if ttft_s <= 12:
        return "C"
    if ttft_s <= 25:
        return "D"
    return "F"


def grade_throughput(tps: Optional[float]) -> str:
    if tps is None:
        return "unknown"
    if tps >= 40:
        return "A"
    if tps >= 20:
        return "B"
    if tps >= 8:
        return "C"
    if tps > 0:
        return "D"
    return "F"


def load_summary_rows(run_dir: Path) -> List[Dict[str, str]]:
    candidates = [run_dir / "run_summary.csv", run_dir / "benchmark_summary.csv", run_dir / "bench" / "run_summary.csv"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
    return []


def load_profile(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "machine_profile.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def synthesize_recommendations(run_dir: Path) -> None:
    rows = load_summary_rows(run_dir)
    profile = load_profile(run_dir)

    capability_rows: List[Dict[str, Any]] = []
    ranked: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        ok_rate = float_or_none(row.get("ok_rate"))
        tps = float_or_none(row.get("tps_med"))
        ttft = float_or_none(row.get("ttft_med"))
        score = (ok_rate or 0.0) * 0.75 + min((tps or 0.0) / 40.0, 1.0) * 0.20 + (0.05 if ttft is not None and ttft <= 8 else 0.0)
        ranked.append((score, row))
        reliability_grade = grade_score(ok_rate)
        throughput_grade = grade_throughput(tps)
        latency_grade = grade_latency(ttft)
        capability_rows.append(
            {
                "run_id": row.get("run_id", ""),
                "host_name": row.get("host_name", ""),
                "host_ip": row.get("host_ip", ""),
                "base_url": row.get("base_url", ""),
                "model_key": row.get("model_key", ""),
                "context_tokens": "",
                "task_family": "general",
                "score": f"{score:.4f}",
                "grade": grade_score(score),
                "latency_grade": latency_grade,
                "throughput_grade": throughput_grade,
                "reliability_grade": reliability_grade,
                "recommended_use": recommended_use_for(row, score),
                "avoid_use": avoid_use_for(row, score),
                "evidence": f"ok_rate={row.get('ok_rate','')}; ttft_med={row.get('ttft_med','')}; tps_med={row.get('tps_med','')}",
                "notes": "Generated by lms.py recommend from benchmark summary.",
            }
        )

    capability_path = run_dir / "capability_matrix.csv"
    with capability_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "host_name",
            "host_ip",
            "base_url",
            "model_key",
            "context_tokens",
            "task_family",
            "score",
            "grade",
            "latency_grade",
            "throughput_grade",
            "reliability_grade",
            "recommended_use",
            "avoid_use",
            "evidence",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(capability_rows)

    ranked.sort(key=lambda item: item[0], reverse=True)
    md_path = run_dir / "agent_recommendations.md"
    lines: List[str] = []
    lines.append("# LMS Agent Recommendations")
    lines.append("")
    lines.append(f"- Generated UTC: `{utc_now_iso()}`")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append("")

    if profile:
        lines.append("## Machine synopsis")
        lines.append("")
        for rec in profile.get("recommendations", []):
            lines.append(f"- {rec}")
        lines.append("")

    lines.append("## Model routing")
    lines.append("")
    if not ranked:
        lines.append("No benchmark summary rows were found. Run `python3 lms.py quick` first.")
    else:
        best_score, best = ranked[0]
        lines.append(f"- Default local model candidate: `{best.get('model_key')}` on `{best.get('base_url')}`.")
        lines.append(f"- Evidence: OK rate `{best.get('ok_rate')}`, median TTFT `{best.get('ttft_med')}`, median TPS `{best.get('tps_med')}`.")
        lines.append("- Use this recommendation as a routing hint, not a guarantee, until task-family-specific scoring is wired into the benchmark runner.")
        lines.append("")
        lines.append("| Rank | Host | Model | OK rate | TTFT | TPS | Suggested use |")
        lines.append("|---:|---|---|---:|---:|---:|---|")
        for idx, (score, row) in enumerate(ranked[:20], start=1):
            lines.append(
                f"| {idx} | `{row.get('host_name','')}` | `{row.get('model_key','')}` | "
                f"{row.get('ok_rate','')} | {row.get('ttft_med','')} | {row.get('tps_med','')} | "
                f"{recommended_use_for(row, score)} |"
            )
    lines.append("")
    lines.append("## Operating rules for agents")
    lines.append("")
    lines.append("- Prefer high OK-rate models for autonomous work.")
    lines.append("- Prefer high TPS models for quick shell help, summaries, and low-risk drafts.")
    lines.append("- Prefer low TTFT models for interactive agents.")
    lines.append("- Do not route long-context or repository-wide work based on this summary alone until context-sweep scoring is enabled.")
    lines.append("- Fall back to a stronger or remote model when outputs are malformed, incomplete, or below task confidence thresholds.")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {capability_path}")
    print(f"wrote {md_path}")


def recommended_use_for(row: Dict[str, str], score: float) -> str:
    ok_rate = float_or_none(row.get("ok_rate")) or 0.0
    tps = float_or_none(row.get("tps_med")) or 0.0
    if score >= 0.85 and tps >= 20:
        return "default local agent model for routine coding, planning, and summaries"
    if ok_rate >= 0.8:
        return "reliable draft/review model; use with normal verification"
    if ok_rate > 0:
        return "limited use for drafts only; require review or fallback"
    return "not recommended"


def avoid_use_for(row: Dict[str, str], score: float) -> str:
    ok_rate = float_or_none(row.get("ok_rate")) or 0.0
    tps = float_or_none(row.get("tps_med"))
    if ok_rate <= 0:
        return "avoid all autonomous work until endpoint/model errors are resolved"
    if tps is not None and tps < 4:
        return "avoid interactive workflows and large outputs"
    if score < 0.55:
        return "avoid complex coding, long-context, and tool-call tasks"
    return "avoid only high-risk work until task-family benchmarks pass"


# ----------------------------
# Commands
# ----------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print("LMS doctor")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Repo root: {REPO_ROOT}")
    scripts = [PROFILE_SCRIPT, EVAL_SCRIPT, BENCHMARK_SCRIPT, DEFAULT_SUITE]
    for script in scripts:
        print(f"{'OK ' if script.exists() else 'MISS'} {script.relative_to(REPO_ROOT) if script.is_relative_to(REPO_ROOT) else script}")

    endpoints = args.endpoint or [default_endpoint()]
    print("\nEndpoint probes:")
    probes = probe_endpoints(endpoints, timeout_s=args.timeout)
    for probe in probes:
        status = "OK" if probe["reachable"] else "FAIL"
        print(f"{status} {probe['base_url']} models={probe['model_count']} elapsed={probe['elapsed_s']}s error={probe['error'] or ''}")
    return 0 if all(p["reachable"] for p in probes) else 1


def cmd_probe(args: argparse.Namespace) -> int:
    endpoints = args.endpoint or [default_endpoint()]
    probes = probe_endpoints(endpoints, timeout_s=args.timeout)
    if args.json:
        print(json.dumps(probes, indent=2))
        return 0 if all(p["reachable"] for p in probes) else 1

    for probe in probes:
        status = "reachable" if probe["reachable"] else "unreachable"
        print(f"{probe['base_url']} — {status} — {probe['model_count']} model(s)")
        if probe["error"]:
            print(f"  error: {probe['error']}")
        for model in probe.get("models", []):
            print(f"  - {model}")
    return 0 if all(p["reachable"] for p in probes) else 1


def cmd_inventory(args: argparse.Namespace) -> int:
    endpoints = args.endpoint or [default_endpoint()]
    probes = probe_endpoints(endpoints, timeout_s=args.timeout)
    rows = write_inventory_csv(
        probes,
        Path(args.out),
        model_filter=parse_csv_arg(args.models),
        max_models=args.max_models,
    )
    print(f"wrote {args.out} with {len(rows)} model row(s)")
    if not rows:
        print("No benchmarkable models found. Check LM Studio server mode and endpoint URL.")
        return 1
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(PROFILE_SCRIPT), "--output-dir", args.output_dir, "--timeout", str(args.timeout)]
    if args.inventory_csv:
        cmd += ["--inventory-csv", args.inventory_csv]
    for endpoint in args.endpoint or [default_endpoint()]:
        cmd += ["--probe-base-url", endpoint]
    return run_subprocess(cmd)


def cmd_quick(args: argparse.Namespace) -> int:
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_dir) / safe_slug(run_id)
    ensure_dir(run_dir)

    endpoints = args.endpoint or [default_endpoint()]
    inventory_csv = run_dir / "lmstudio_inventory.csv"

    print(f"Creating LMS run: {run_dir}")
    probes = probe_endpoints(endpoints, timeout_s=args.timeout)
    rows = write_inventory_csv(
        probes,
        inventory_csv,
        model_filter=parse_csv_arg(args.models),
        max_models=args.max_models,
    )
    print(f"Inventory rows: {len(rows)}")
    if not rows:
        print("No benchmarkable models found; wrote inventory/probe diagnostics only.")

    run_config = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "endpoints": [normalize_base_url(e) for e in endpoints],
        "inventory_csv": str(inventory_csv),
        "suite_file": str(DEFAULT_SUITE),
        "models_filter": parse_csv_arg(args.models),
        "max_models": args.max_models,
        "timeout": args.timeout,
        "repeats": args.repeats,
    }
    (run_dir / "lms_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if DEFAULT_SUITE.exists():
        shutil.copy2(DEFAULT_SUITE, run_dir / "agent_skill_suite.v1.json")

    profile_code = cmd_profile(
        argparse.Namespace(
            output_dir=str(run_dir),
            timeout=args.timeout,
            inventory_csv=str(inventory_csv),
            endpoint=endpoints,
        )
    )
    if profile_code != 0 and args.strict:
        return profile_code

    if args.profile_only or not rows:
        print(f"Run directory ready: {run_dir}")
        return 0 if rows else 1

    if not BENCHMARK_SCRIPT.exists():
        print(f"Benchmark script missing: {BENCHMARK_SCRIPT}")
        return 1

    bench_cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--inventory-csv",
        str(inventory_csv),
        "--output-dir",
        str(run_dir),
        "--sidecar-dir",
        str(run_dir / "sidecars"),
        "--timeout",
        str(args.timeout),
        "--repeats",
        str(args.repeats),
    ]
    if args.models:
        bench_cmd += ["--include-models", args.models]
    if args.exclude_models:
        bench_cmd += ["--exclude-models", args.exclude_models]

    bench_code = run_subprocess(bench_cmd)
    if bench_code != 0 and args.strict:
        return bench_code

    synthesize_recommendations(run_dir)
    print(f"\nDone. Agent artifacts are in: {run_dir}")
    print(f"Read: {run_dir / 'agent_recommendations.md'}")
    return bench_code


def cmd_recommend(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"run directory not found: {run_dir}")
        return 1
    synthesize_recommendations(run_dir)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cmd = [sys.executable, str(EVAL_SCRIPT)]
    if args.output_file:
        cmd += ["--output-file", args.output_file]
    if args.evaluators_json:
        cmd += ["--evaluators-json", args.evaluators_json]
    if args.evaluators_file:
        cmd += ["--evaluators-file", args.evaluators_file]
    if args.pretty:
        cmd += ["--pretty"]
    return run_subprocess(cmd)


# ----------------------------
# Parser
# ----------------------------
def add_endpoint_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help=f"LM Studio OpenAI-compatible base URL. Repeat for multiple. Default: {DEFAULT_ENDPOINT} or LMS_BASE_URL.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms",
        description="Simple CLI for agents to profile and benchmark LM Studio nodes.",
    )
    parser.add_argument("--version", action="version", version="lms-agent-cli 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local scripts and endpoint reachability")
    add_endpoint_arg(doctor)
    doctor.add_argument("--timeout", type=int, default=8)
    doctor.set_defaults(func=cmd_doctor)

    probe = sub.add_parser("probe", help="List models from LM Studio endpoint(s)")
    add_endpoint_arg(probe)
    probe.add_argument("--timeout", type=int, default=8)
    probe.add_argument("--json", action="store_true", help="Print machine-readable probe JSON")
    probe.set_defaults(func=cmd_probe)

    inventory = sub.add_parser("inventory", help="Create benchmark inventory CSV from endpoint model lists")
    add_endpoint_arg(inventory)
    inventory.add_argument("--out", default="lmstudio_inventory.csv")
    inventory.add_argument("--models", default=None, help="Comma-separated model IDs to include")
    inventory.add_argument("--max-models", type=int, default=0, help="Limit models per endpoint; 0 means all")
    inventory.add_argument("--timeout", type=int, default=8)
    inventory.set_defaults(func=cmd_inventory)

    profile = sub.add_parser("profile", help="Collect machine profile and LM Studio endpoint synopsis")
    add_endpoint_arg(profile)
    profile.add_argument("--output-dir", default="runs/profile")
    profile.add_argument("--inventory-csv", default=None)
    profile.add_argument("--timeout", type=int, default=8)
    profile.set_defaults(func=cmd_profile)

    quick = sub.add_parser("quick", help="One-command agent benchmark run with defaults")
    add_endpoint_arg(quick)
    quick.add_argument("--output-dir", default="runs")
    quick.add_argument("--run-id", default=None)
    quick.add_argument("--models", default=None, help="Comma-separated model IDs to include")
    quick.add_argument("--exclude-models", default=None, help="Comma-separated model IDs to exclude")
    quick.add_argument("--max-models", type=int, default=3, help="Limit models per endpoint for quick runs")
    quick.add_argument("--repeats", type=int, default=1)
    quick.add_argument("--timeout", type=int, default=900)
    quick.add_argument("--profile-only", action="store_true")
    quick.add_argument("--strict", action="store_true", help="Fail immediately when profile or benchmark subprocess fails")
    quick.set_defaults(func=cmd_quick)

    recommend = sub.add_parser("recommend", help="Generate capability matrix and agent recommendations from a run directory")
    recommend.add_argument("run_dir")
    recommend.set_defaults(func=cmd_recommend)

    evaluate = sub.add_parser("eval", help="Run deterministic evaluators against an output file/stdin")
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
