#!/usr/bin/env python3
"""
LMS agent CLI.

A small, dependency-light command line wrapper that lets agents use the LMS
benchmarking toolkit without hand-editing configs.

Common usage:
  lms doctor
  lms probe
  lms quick
  lms quick --endpoint http://100.64.0.10:1234/v1 --repeats 1
  lms inventory --endpoint http://127.0.0.1:1234/v1 --out lmstudio_inventory.csv
  lms profile --endpoint http://127.0.0.1:1234/v1
  lms runs
  lms show runs/<run_id>
  lms route runs/<run_id> --task coding
  lms recommend runs/<run_id>
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


REPO_ROOT = Path(__file__).resolve().parent
CWD = Path.cwd()
DEFAULT_ENDPOINT = "http://127.0.0.1:1234/v1"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_SUITE_REL = Path("benchmarks") / "agent_skill_suite.v1.json"
BENCHMARK_SCRIPT_REL = Path("benchmark_lmstudio_cross_machine_models.py")
PROFILE_SCRIPT_REL = Path("lms_machine_profile.py")
EVAL_SCRIPT_REL = Path("lms_eval.py")


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


def resolve_path(path: Path, required: bool = True) -> Path:
    """Resolve a repo asset for both editable and script-local installs."""
    candidates = [
        REPO_ROOT / path,
        CWD / path,
        CWD.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        # Return the primary candidate so error messages point to the intended path.
        return candidates[0]
    return candidates[0]


def suite_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env_suite = os.environ.get("LMS_SUITE_FILE")
    if env_suite:
        return Path(env_suite)
    return resolve_path(DEFAULT_SUITE_REL, required=False)


def benchmark_script_path() -> Path:
    return resolve_path(BENCHMARK_SCRIPT_REL, required=False)


def profile_script_path() -> Path:
    return resolve_path(PROFILE_SCRIPT_REL, required=False)


def eval_script_path() -> Path:
    return resolve_path(EVAL_SCRIPT_REL, required=False)


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


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def print_table(rows: List[List[str]], headers: List[str]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


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
    # Best effort only. This does not send packets; UDP connect asks the local
    # routing table which source address would be used.
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
# Run discovery / summaries
# ----------------------------
def discover_runs(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    candidates = []
    for path in runs_dir.iterdir():
        if path.is_dir() and any((path / marker).exists() for marker in ["lms_run_config.json", "machine_profile.json", "run_summary.csv", "agent_recommendations.md"]):
            candidates.append(path)
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_run(runs_dir: Path) -> Optional[Path]:
    runs = discover_runs(runs_dir)
    return runs[0] if runs else None


def load_summary_rows(run_dir: Path) -> List[Dict[str, str]]:
    candidates = [run_dir / "run_summary.csv", run_dir / "benchmark_summary.csv", run_dir / "bench" / "run_summary.csv"]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
    return []


def load_capability_rows(run_dir: Path) -> List[Dict[str, str]]:
    path = run_dir / "capability_matrix.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_profile(run_dir: Path) -> Optional[Dict[str, Any]]:
    return read_json(run_dir / "machine_profile.json")


def run_status(run_dir: Path) -> str:
    if (run_dir / "agent_recommendations.md").exists() and (run_dir / "capability_matrix.csv").exists():
        return "recommended"
    if (run_dir / "run_summary.csv").exists():
        return "benchmarked"
    if (run_dir / "machine_profile.json").exists():
        return "profiled"
    return "created"


def resolve_run_dir(value: str, runs_dir: str = DEFAULT_RUNS_DIR) -> Path:
    if value == "latest":
        latest = latest_run(Path(runs_dir))
        if not latest:
            raise SystemExit(f"no runs found under {runs_dir}")
        return latest
    return Path(value)


# ----------------------------
# Recommendations / routing
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


def score_summary_row(row: Dict[str, str]) -> float:
    ok_rate = float_or_none(row.get("ok_rate"))
    tps = float_or_none(row.get("tps_med"))
    ttft = float_or_none(row.get("ttft_med"))
    return (ok_rate or 0.0) * 0.75 + min((tps or 0.0) / 40.0, 1.0) * 0.20 + (0.05 if ttft is not None and ttft <= 8 else 0.0)


def synthesize_recommendations(run_dir: Path) -> None:
    rows = load_summary_rows(run_dir)
    profile = load_profile(run_dir)

    capability_rows: List[Dict[str, Any]] = []
    ranked: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        score = score_summary_row(row)
        ranked.append((score, row))
        ok_rate = float_or_none(row.get("ok_rate"))
        tps = float_or_none(row.get("tps_med"))
        ttft = float_or_none(row.get("ttft_med"))
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
        lines.append("No benchmark summary rows were found. Run `lms quick` first.")
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


def choose_route(run_dir: Path, task: str = "general") -> Optional[Dict[str, str]]:
    capability_rows = load_capability_rows(run_dir)
    if capability_rows:
        task_rows = [r for r in capability_rows if r.get("task_family") == task]
        if not task_rows and task != "general":
            task_rows = [r for r in capability_rows if r.get("task_family") == "general"]
        rows = task_rows or capability_rows
        rows.sort(key=lambda r: float_or_none(r.get("score")) or 0.0, reverse=True)
        return rows[0] if rows else None

    summary_rows = load_summary_rows(run_dir)
    if not summary_rows:
        return None
    ranked = sorted(summary_rows, key=score_summary_row, reverse=True)
    best = ranked[0]
    return {
        "run_id": best.get("run_id", ""),
        "task_family": task,
        "host_name": best.get("host_name", ""),
        "host_ip": best.get("host_ip", ""),
        "base_url": best.get("base_url", ""),
        "model_key": best.get("model_key", ""),
        "score": f"{score_summary_row(best):.4f}",
        "grade": grade_score(score_summary_row(best)),
        "evidence": f"ok_rate={best.get('ok_rate','')}; ttft_med={best.get('ttft_med','')}; tps_med={best.get('tps_med','')}",
        "recommended_use": recommended_use_for(best, score_summary_row(best)),
        "avoid_use": avoid_use_for(best, score_summary_row(best)),
    }


def render_route_yaml(route: Dict[str, str]) -> str:
    return "\n".join(
        [
            "routing:",
            f"  {route.get('task_family', 'general')}:",
            f"    preferred_model: {json.dumps(route.get('model_key', ''))}",
            f"    base_url: {json.dumps(route.get('base_url', ''))}",
            f"    host_name: {json.dumps(route.get('host_name', ''))}",
            f"    score: {json.dumps(route.get('score', ''))}",
            f"    grade: {json.dumps(route.get('grade', ''))}",
            f"    evidence: {json.dumps(route.get('evidence', ''))}",
            f"    recommended_use: {json.dumps(route.get('recommended_use', ''))}",
            f"    avoid_use: {json.dumps(route.get('avoid_use', ''))}",
            "    source: lms capability_matrix.csv",
            "",
        ]
    )


# ----------------------------
# Commands
# ----------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print("LMS doctor")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Repo root: {REPO_ROOT}")
    scripts = [profile_script_path(), eval_script_path(), benchmark_script_path(), suite_path()]
    for script in scripts:
        label = script.name if script.parent == REPO_ROOT else str(script)
        print(f"{'OK ' if script.exists() else 'MISS'} {label}")

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
    script = profile_script_path()
    if not script.exists():
        print(f"profile script missing: {script}")
        return 1
    cmd = [sys.executable, str(script), "--output-dir", args.output_dir, "--timeout", str(args.timeout)]
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
    selected_suite = suite_path(args.suite_file)

    print(f"Creating LMS run: {run_dir}")
    probes = probe_endpoints(endpoints, timeout_s=args.timeout)
    (run_dir / "endpoint_probes.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
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
        "suite_file": str(selected_suite),
        "models_filter": parse_csv_arg(args.models),
        "exclude_models": parse_csv_arg(args.exclude_models),
        "max_models": args.max_models,
        "timeout": args.timeout,
        "repeats": args.repeats,
    }
    (run_dir / "lms_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if selected_suite.exists():
        shutil.copy2(selected_suite, run_dir / selected_suite.name)
    else:
        print(f"warning: suite file not found: {selected_suite}")

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

    bench_script = benchmark_script_path()
    if not bench_script.exists():
        print(f"Benchmark script missing: {bench_script}")
        return 1

    bench_cmd = [
        sys.executable,
        str(bench_script),
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
    route = choose_route(run_dir, task="general")
    if route:
        (run_dir / "routing_rules.yaml").write_text(render_route_yaml(route), encoding="utf-8")
        (run_dir / "routing_rules.json").write_text(json.dumps({"routing": {"general": route}}, indent=2), encoding="utf-8")
        print(f"wrote {run_dir / 'routing_rules.yaml'}")

    print(f"\nDone. Agent artifacts are in: {run_dir}")
    print(f"Read: {run_dir / 'agent_recommendations.md'}")
    return bench_code


def cmd_recommend(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    if not run_dir.exists():
        print(f"run directory not found: {run_dir}")
        return 1
    synthesize_recommendations(run_dir)
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    if not run_dir.exists():
        print(f"run directory not found: {run_dir}")
        return 1
    if not (run_dir / "capability_matrix.csv").exists():
        synthesize_recommendations(run_dir)
    route = choose_route(run_dir, task=args.task)
    if not route:
        print("No route could be selected. Run `lms quick` first.")
        return 1

    if args.json:
        print(json.dumps(route, indent=2))
    else:
        print(render_route_yaml(route), end="")

    if args.write:
        yaml_path = run_dir / "routing_rules.yaml"
        json_path = run_dir / "routing_rules.json"
        yaml_path.write_text(render_route_yaml(route), encoding="utf-8")
        json_path.write_text(json.dumps({"routing": {args.task: route}}, indent=2), encoding="utf-8")
        print(f"wrote {yaml_path}")
        print(f"wrote {json_path}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    runs = discover_runs(Path(args.runs_dir))[: args.limit]
    if not runs:
        print(f"No runs found under {args.runs_dir}")
        return 1
    rows: List[List[str]] = []
    for run in runs:
        rows.append(
            [
                run.name,
                run_status(run),
                dt.datetime.fromtimestamp(run.stat().st_mtime).isoformat(timespec="seconds"),
                str(run),
            ]
        )
    print_table(rows, ["run_id", "status", "modified", "path"])
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    if not run_dir.exists():
        print(f"run directory not found: {run_dir}")
        return 1
    print(f"Run: {run_dir}")
    print(f"Status: {run_status(run_dir)}")

    config = read_json(run_dir / "lms_run_config.json")
    if config:
        print(f"Created: {config.get('created_at_utc')}")
        print(f"Endpoints: {', '.join(config.get('endpoints', []))}")

    profile = load_profile(run_dir)
    if profile:
        host = profile.get("host", {})
        mem = profile.get("memory", {})
        mem_gib = None
        if mem.get("mem_total_bytes") is not None:
            mem_gib = round(float(mem["mem_total_bytes"]) / (1024 ** 3), 2)
        print(f"Host: {host.get('hostname')} / {host.get('platform')}")
        if mem_gib is not None:
            print(f"RAM: {mem_gib} GiB")

    rows = load_summary_rows(run_dir)
    if rows:
        table_rows = []
        ranked = sorted(rows, key=score_summary_row, reverse=True)
        for row in ranked[: args.limit]:
            table_rows.append(
                [
                    row.get("host_name", ""),
                    row.get("model_key", ""),
                    row.get("ok_rate", ""),
                    row.get("ttft_med", ""),
                    row.get("tps_med", ""),
                    f"{score_summary_row(row):.3f}",
                ]
            )
        print("\nTop models")
        print_table(table_rows, ["host", "model", "ok", "ttft", "tps", "score"])

    route = choose_route(run_dir, args.task)
    if route:
        print("\nSelected route")
        print(f"Model: {route.get('model_key')} @ {route.get('base_url')}")
        print(f"Score: {route.get('score')} Grade: {route.get('grade')}")
        print(f"Evidence: {route.get('evidence')}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    script = eval_script_path()
    if not script.exists():
        print(f"eval script missing: {script}")
        return 1
    cmd = [sys.executable, str(script)]
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
    parser.add_argument("--version", action="version", version="lms-agent-cli 0.2.0")
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
    quick.add_argument("--output-dir", default=DEFAULT_RUNS_DIR)
    quick.add_argument("--run-id", default=None)
    quick.add_argument("--suite-file", default=None, help="Benchmark suite manifest to copy into the run directory")
    quick.add_argument("--models", default=None, help="Comma-separated model IDs to include")
    quick.add_argument("--exclude-models", default=None, help="Comma-separated model IDs to exclude")
    quick.add_argument("--max-models", type=int, default=3, help="Limit models per endpoint for quick runs")
    quick.add_argument("--repeats", type=int, default=1)
    quick.add_argument("--timeout", type=int, default=900)
    quick.add_argument("--profile-only", action="store_true")
    quick.add_argument("--strict", action="store_true", help="Fail immediately when profile or benchmark subprocess fails")
    quick.set_defaults(func=cmd_quick)

    runs = sub.add_parser("runs", help="List known LMS run directories")
    runs.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=cmd_runs)

    show = sub.add_parser("show", help="Show a compact run summary")
    show.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    show.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    show.add_argument("--task", default="general", help="Task family for route selection")
    show.add_argument("--limit", type=int, default=10)
    show.set_defaults(func=cmd_show)

    recommend = sub.add_parser("recommend", help="Generate capability matrix and agent recommendations from a run directory")
    recommend.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    recommend.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    recommend.set_defaults(func=cmd_recommend)

    route = sub.add_parser("route", help="Print/export the best route for an agent task")
    route.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    route.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    route.add_argument("--task", default="general", help="Task family, e.g. general, coding, long_context")
    route.add_argument("--json", action="store_true", help="Print JSON instead of YAML")
    route.add_argument("--write", action="store_true", help="Write routing_rules.yaml and routing_rules.json into the run directory")
    route.set_defaults(func=cmd_route)

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
