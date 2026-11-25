#!/usr/bin/env python3
"""Generate a concise agent-facing brief for an LMS run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def best_by_task(cap_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in cap_rows:
        grouped.setdefault(row.get("task_family", "general"), []).append(row)
    out = {}
    for task, rows in grouped.items():
        rows.sort(key=lambda r: fnum(r.get("score")), reverse=True)
        if rows:
            out[task] = rows[0]
    return out


def worst_fit(fit_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    order = {"poor": 0, "risky": 1, "borderline": 2, "unknown": 3, "good": 4}
    return sorted(fit_rows, key=lambda r: order.get(r.get("fit_grade", "unknown"), 3))


def generate_brief(run_dir: Path) -> str:
    config = read_json(run_dir / "lms_run_config.json")
    profile = read_json(run_dir / "machine_profile.json")
    routes = read_json(run_dir / "routing_rules.json")
    cap = read_csv(run_dir / "capability_matrix.csv")
    fit = read_csv(run_dir / "model_fit.csv")
    summary = read_csv(run_dir / "run_summary.csv")
    task_best = best_by_task(cap)

    lines = ["# LMS Agent Brief", "", f"- Run: `{run_dir}`"]
    if config:
        lines.append(f"- Created: `{config.get('created_at_utc', '')}`")
        lines.append(f"- Endpoints: `{', '.join(config.get('endpoints', []))}`")
    host = (profile.get("host") or {}).get("hostname")
    platform = (profile.get("host") or {}).get("platform")
    if host or platform:
        lines.append(f"- Machine: `{host}` / `{platform}`")
    lines.append("")

    lines += ["## Recommended routes", ""]
    if task_best:
        lines += ["| Task | Model | Score | Grade | Max reliable context | Evidence |", "|---|---|---:|---|---:|---|"]
        for task in sorted(task_best):
            row = task_best[task]
            lines.append(f"| `{task}` | `{row.get('model_key','')}` | {row.get('score','')} | {row.get('grade','')} | {row.get('max_reliable_context_tokens','')} | {row.get('evidence','')} |")
    else:
        lines.append("No capability matrix found. Run `lms recommend` first.")
    lines.append("")

    lines += ["## Hardware/model fit warnings", ""]
    if fit:
        lines += ["| Model | Fit | Estimated GiB | Notes |", "|---|---|---:|---|"]
        for row in worst_fit(fit)[:10]:
            lines.append(f"| `{row.get('model_key','')}` | {row.get('fit_grade','')} | {row.get('estimated_model_memory_gib','')} | {row.get('fit_notes','')} |")
    else:
        lines.append("No model fit report found. Run `lms fit latest`.")
    lines.append("")

    lines += ["## Operational guidance", ""]
    lines.append("- Use task-specific routes first; fall back to general only when the task route is missing.")
    lines.append("- Prefer models with high evaluator scores over models that are only fast.")
    lines.append("- Avoid long-context work beyond the reported max reliable context.")
    lines.append("- Treat `risky` or `poor` fit models as experimental even if they benchmark successfully once.")
    lines.append("- Use safety routes for shell commands, deployment advice, secrets handling, and network exposure reviews.")
    lines.append("- Re-run `lms compare` after model, driver, LM Studio, or hardware changes.")
    lines.append("")

    if summary:
        lines += ["## Top raw throughput models", "", "| Model | OK rate | Eval OK | Eval score | TTFT | TPS |", "|---|---:|---:|---:|---:|---:|"]
        ranked = sorted(summary, key=lambda r: fnum(r.get("tps_med")), reverse=True)
        for row in ranked[:8]:
            lines.append(f"| `{row.get('model_key','')}` | {row.get('ok_rate','')} | {row.get('eval_ok_rate','')} | {row.get('eval_score_avg','')} | {row.get('ttft_med','')} | {row.get('tps_med','')} |")
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a single agent-facing LMS run brief.")
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None, help="Default: run_dir/AGENT_BRIEF.md")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "AGENT_BRIEF.md"
    out.write_text(generate_brief(run_dir), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
