#!/usr/bin/env python3
"""Decision engine for agent model routing.

This module turns benchmark evidence into an actionable allow/fallback/block
answer that agents can consume before assigning work to a local LM Studio model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BLOCKING_FIT_GRADES = {"poor"}
RISKY_FIT_GRADES = {"risky", "borderline", "unknown"}


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


def int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def model_fit_map(run_dir: Path) -> Dict[str, Dict[str, str]]:
    return {row.get("model_key", ""): row for row in read_csv(run_dir / "model_fit.csv") if row.get("model_key")}


def sorted_task_capabilities(run_dir: Path, task: str) -> List[Dict[str, str]]:
    rows = read_csv(run_dir / "capability_matrix.csv")
    selected = [row for row in rows if row.get("task_family") == task]
    if not selected and task != "general":
        selected = [row for row in rows if row.get("task_family") == "general"]
    selected = selected or rows
    selected.sort(key=lambda row: fnum(row.get("score")), reverse=True)
    return selected


def route_violation(row: Dict[str, str], fit: Dict[str, str], context_tokens: Optional[int], min_score: float, require_audit_pass: bool, audit: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if require_audit_pass and audit.get("status") == "fail":
        reasons.append("run audit status is fail")
    if fnum(row.get("score")) < min_score:
        reasons.append(f"route score {fnum(row.get('score')):.3f} is below minimum {min_score:.3f}")
    fit_grade = fit.get("fit_grade")
    if fit_grade in BLOCKING_FIT_GRADES:
        reasons.append(f"model fit is {fit_grade}")
    max_ctx = int_or_none(row.get("max_reliable_context_tokens"))
    if context_tokens is not None and max_ctx is not None and max_ctx > 0 and context_tokens > max_ctx:
        reasons.append(f"requested context {context_tokens} exceeds max reliable context {max_ctx}")
    return reasons


def route_warnings(row: Dict[str, str], fit: Dict[str, str], context_tokens: Optional[int], audit: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    if audit.get("status") == "warn":
        warnings.extend([f"audit warning: {w}" for w in audit.get("warnings", [])[:5]])
    fit_grade = fit.get("fit_grade")
    if fit_grade in RISKY_FIT_GRADES:
        warnings.append(f"model fit is {fit_grade}: {fit.get('fit_notes', '')}")
    max_ctx = int_or_none(row.get("max_reliable_context_tokens"))
    if context_tokens is not None and not max_ctx:
        warnings.append("no measured max reliable context for this route")
    return warnings


def select_route(rows: List[Dict[str, str]], fits: Dict[str, Dict[str, str]], context_tokens: Optional[int], min_score: float, require_audit_pass: bool, audit: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], List[Dict[str, Any]]]:
    evaluated: List[Dict[str, Any]] = []
    for row in rows:
        fit = fits.get(row.get("model_key", ""), {})
        violations = route_violation(row, fit, context_tokens, min_score, require_audit_pass, audit)
        warnings = route_warnings(row, fit, context_tokens, audit)
        evaluated.append({"route": row, "fit": fit, "violations": violations, "warnings": warnings})
        if not violations:
            return row, evaluated
    return None, evaluated


def decide(run_dir: Path, task: str, context_tokens: Optional[int], min_score: float, require_audit_pass: bool = True, allow_warn_audit: bool = True) -> Dict[str, Any]:
    audit = read_json(run_dir / "run_audit.json")
    rows = sorted_task_capabilities(run_dir, task)
    fits = model_fit_map(run_dir)
    if not rows:
        return {
            "decision": "block",
            "reason": "no capability rows available",
            "task_family": task,
            "run_dir": str(run_dir),
            "selected": None,
            "fallback": None,
            "warnings": [],
            "evaluated": [],
        }

    if audit.get("status") == "fail" and require_audit_pass:
        return {
            "decision": "block",
            "reason": "run audit failed",
            "task_family": task,
            "run_dir": str(run_dir),
            "selected": None,
            "fallback": None,
            "warnings": audit.get("critical", []) + audit.get("warnings", []),
            "evaluated": [],
        }

    selected, evaluated = select_route(rows, fits, context_tokens, min_score, require_audit_pass, audit)
    if selected:
        fallback = None
        for item in evaluated:
            route = item["route"]
            if route is selected:
                continue
            if not item["violations"] and (route.get("model_key") != selected.get("model_key") or route.get("base_url") != selected.get("base_url")):
                fallback = route
                break
        decision = "allow"
        warnings = route_warnings(selected, fits.get(selected.get("model_key", ""), {}), context_tokens, audit)
        if audit.get("status") == "warn" and not allow_warn_audit:
            decision = "fallback" if fallback else "block"
            warnings.append("audit status is warn and allow_warn_audit is false")
        return {
            "decision": decision,
            "reason": "selected route satisfies policy" if decision == "allow" else "selected route requires fallback/block due to warnings policy",
            "task_family": task,
            "run_dir": str(run_dir),
            "selected": selected,
            "fallback": fallback,
            "warnings": warnings,
            "evaluated": evaluated,
            "policy": {"min_score": min_score, "context_tokens": context_tokens, "require_audit_pass": require_audit_pass, "allow_warn_audit": allow_warn_audit},
        }

    return {
        "decision": "block",
        "reason": "no route satisfies policy",
        "task_family": task,
        "run_dir": str(run_dir),
        "selected": None,
        "fallback": None,
        "warnings": [],
        "evaluated": evaluated,
        "policy": {"min_score": min_score, "context_tokens": context_tokens, "require_audit_pass": require_audit_pass, "allow_warn_audit": allow_warn_audit},
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# LMS Routing Decision", "", f"- Decision: `{result.get('decision')}`", f"- Reason: {result.get('reason')}", f"- Task: `{result.get('task_family')}`", f"- Run: `{result.get('run_dir')}`", ""]
    selected = result.get("selected") or {}
    if selected:
        lines += ["## Selected route", ""]
        lines.append(f"- Model: `{selected.get('model_key')}`")
        lines.append(f"- Endpoint: `{selected.get('base_url')}`")
        lines.append(f"- Score: `{selected.get('score')}`")
        lines.append(f"- Grade: `{selected.get('grade')}`")
        lines.append(f"- Evidence: {selected.get('evidence')}")
    fallback = result.get("fallback") or {}
    if fallback:
        lines += ["", "## Fallback route", ""]
        lines.append(f"- Model: `{fallback.get('model_key')}`")
        lines.append(f"- Endpoint: `{fallback.get('base_url')}`")
        lines.append(f"- Score: `{fallback.get('score')}`")
    if result.get("warnings"):
        lines += ["", "## Warnings", ""]
        for warning in result["warnings"]:
            lines.append(f"- {warning}")
    if result.get("evaluated"):
        lines += ["", "## Evaluated routes", "", "| Model | Score | Violations | Warnings |", "|---|---:|---|---|"]
        for item in result["evaluated"][:20]:
            route = item.get("route") or {}
            lines.append(f"| `{route.get('model_key','')}` | {route.get('score','')} | {'; '.join(item.get('violations') or [])} | {'; '.join(item.get('warnings') or [])} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make an allow/fallback/block routing decision from LMS run evidence.")
    parser.add_argument("run_dir")
    parser.add_argument("--task", default="general")
    parser.add_argument("--context-tokens", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--no-require-audit-pass", action="store_true")
    parser.add_argument("--no-allow-warn-audit", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = decide(
        Path(args.run_dir),
        task=args.task,
        context_tokens=args.context_tokens,
        min_score=args.min_score,
        require_audit_pass=not args.no_require_audit_pass,
        allow_warn_audit=not args.no_allow_warn_audit,
    )
    json_out = Path(args.json_out) if args.json_out else Path(args.run_dir) / "routing_decision.json"
    md_out = Path(args.md_out) if args.md_out else Path(args.run_dir) / "ROUTING_DECISION.md"
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0 if result.get("decision") in {"allow", "fallback"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
