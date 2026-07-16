#!/usr/bin/env python3
"""Audit an LMS run directory for completeness and agent-readiness.

The audit is intentionally conservative. It does not replace benchmark scoring;
it checks whether the run artifacts are complete enough for an agent to rely on
and highlights low-quality or missing evidence before routing work.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_FILES = [
    "lms_run_config.json",
    "machine_profile.json",
    "lmstudio_inventory.csv",
    "run_results.csv",
    "run_summary.csv",
    "task_summary.csv",
    "capability_matrix.csv",
    "agent_recommendations.md",
    "routing_rules.json",
    "routing_rules.yaml",
    "model_fit.csv",
    "model_fit.md",
    "AGENT_BRIEF.md",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def boolish(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes", "ok"}


def audit_run(run_dir: Path, min_score: float, min_eval_ok: float, require_safety: bool) -> Dict[str, Any]:
    critical: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    missing = [name for name in REQUIRED_FILES if not (run_dir / name).exists()]
    if missing:
        critical.append("missing required artifacts: " + ", ".join(missing))

    inventory = read_csv(run_dir / "lmstudio_inventory.csv")
    results = read_csv(run_dir / "run_results.csv")
    summary = read_csv(run_dir / "run_summary.csv")
    task_summary = read_csv(run_dir / "task_summary.csv")
    capabilities = read_csv(run_dir / "capability_matrix.csv")
    fit = read_csv(run_dir / "model_fit.csv")
    routes = read_json(run_dir / "routing_rules.json")

    if not inventory:
        critical.append("inventory has no model rows")
    if not results:
        critical.append("run_results.csv has no benchmark rows")
    if not summary:
        critical.append("run_summary.csv has no model summary rows")
    if not capabilities:
        critical.append("capability_matrix.csv has no routing rows")

    run_rows = [r for r in results if r.get("phase") == "run"]
    if run_rows:
        ok_rate = sum(1 for r in run_rows if boolish(r.get("ok"))) / len(run_rows)
        eval_ok_rate = sum(1 for r in run_rows if boolish(r.get("eval_ok"))) / len(run_rows)
        info.append(f"benchmark case OK rate: {ok_rate:.2f}")
        info.append(f"deterministic evaluator OK rate: {eval_ok_rate:.2f}")
        if eval_ok_rate < min_eval_ok:
            warnings.append(f"evaluator OK rate {eval_ok_rate:.2f} is below threshold {min_eval_ok:.2f}")
    else:
        critical.append("no phase=run rows found in run_results.csv")

    task_families = sorted({r.get("task_family", "") for r in task_summary if r.get("task_family")})
    info.append("task families: " + (", ".join(task_families) if task_families else "none"))
    if require_safety and "safety" not in task_families:
        warnings.append("safety task family was not benchmarked")

    low_capabilities = [r for r in capabilities if fnum(r.get("score")) < min_score]
    if low_capabilities:
        warnings.append(f"{len(low_capabilities)} capability rows are below score threshold {min_score:.2f}")

    best_by_task: Dict[str, Dict[str, str]] = {}
    for row in capabilities:
        task = row.get("task_family", "general")
        if task not in best_by_task or fnum(row.get("score")) > fnum(best_by_task[task].get("score")):
            best_by_task[task] = row
    for task, row in sorted(best_by_task.items()):
        score = fnum(row.get("score"))
        if score < min_score:
            warnings.append(f"best route for {task} is low confidence: score={score:.2f}, model={row.get('model_key')}")

    poor_fit = [r for r in fit if r.get("fit_grade") in {"poor", "risky"}]
    if poor_fit:
        warnings.append(f"{len(poor_fit)} model fit rows are risky/poor")

    routing = routes.get("routing") if isinstance(routes, dict) else None
    if not routing:
        critical.append("routing_rules.json is missing routing object")
    elif require_safety and "safety" not in routing:
        warnings.append("routing_rules.json has no safety route")

    sidecars = run_dir / "sidecars"
    if not sidecars.exists():
        warnings.append("sidecars directory missing; raw output review may be incomplete")

    status = "fail" if critical else "warn" if warnings else "pass"
    return {
        "status": status,
        "ok": status == "pass",
        "critical": critical,
        "warnings": warnings,
        "info": info,
        "counts": {
            "inventory_rows": len(inventory),
            "result_rows": len(results),
            "run_rows": len(run_rows),
            "summary_rows": len(summary),
            "task_summary_rows": len(task_summary),
            "capability_rows": len(capabilities),
            "model_fit_rows": len(fit),
        },
        "thresholds": {
            "min_score": min_score,
            "min_eval_ok": min_eval_ok,
            "require_safety": require_safety,
        },
    }


def render_markdown(run_dir: Path, audit: Dict[str, Any]) -> str:
    lines = ["# LMS Run Audit", "", f"- Run: `{run_dir}`", f"- Status: `{audit.get('status')}`", ""]
    lines += ["## Counts", ""]
    for key, value in audit.get("counts", {}).items():
        lines.append(f"- {key}: `{value}`")
    if audit.get("critical"):
        lines += ["", "## Critical failures", ""]
        for item in audit["critical"]:
            lines.append(f"- {item}")
    if audit.get("warnings"):
        lines += ["", "## Warnings", ""]
        for item in audit["warnings"]:
            lines.append(f"- {item}")
    if audit.get("info"):
        lines += ["", "## Info", ""]
        for item in audit["info"]:
            lines.append(f"- {item}")
    lines += ["", "## Agent guidance", ""]
    if audit.get("status") == "pass":
        lines.append("- This run has complete artifacts and passes audit thresholds. Agents may use task-specific routes with normal verification.")
    elif audit.get("status") == "warn":
        lines.append("- This run is usable with caution. Agents should prefer high-scoring routes and avoid warned task families or risky fit models.")
    else:
        lines.append("- This run is not ready for autonomous routing. Re-run `lms quick` or repair missing artifacts before use.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit an LMS run directory for completeness and route-readiness.")
    parser.add_argument("run_dir")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-eval-ok", type=float, default=0.60)
    parser.add_argument("--no-require-safety", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    audit = audit_run(run_dir, min_score=args.min_score, min_eval_ok=args.min_eval_ok, require_safety=not args.no_require_safety)
    json_out = Path(args.json_out) if args.json_out else run_dir / "run_audit.json"
    md_out = Path(args.md_out) if args.md_out else run_dir / "RUN_AUDIT.md"
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(render_markdown(run_dir, audit), encoding="utf-8")
    print(json.dumps(audit, indent=2 if args.pretty else None, sort_keys=True))
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0 if audit.get("status") in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
