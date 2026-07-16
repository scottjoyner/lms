#!/usr/bin/env python3
"""History-backed routing decisions for lms-bench.

This command answers the agent question: "Across everything I have measured and
indexed, what route may I use for this task right now?"

Unlike lms_decide.py, this module does not require a run directory. It reads the
SQLite history database populated by `lms-bench history ingest` or automatic
post-run ingestion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lms_history


def int_or_none(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def route_violations(row: Dict[str, Any], context_tokens: Optional[int], min_score: float, require_audit_pass: bool) -> List[str]:
    violations: List[str] = []
    score = fnum(row.get("score"))
    if score < min_score:
        violations.append(f"route score {score:.3f} is below minimum {min_score:.3f}")
    if require_audit_pass and not bool(row.get("audit_ok")):
        violations.append(f"source run audit is not pass: {row.get('audit_status')}")
    max_ctx = int_or_none(row.get("max_reliable_context_tokens"))
    if context_tokens is not None and max_ctx is not None and max_ctx > 0 and context_tokens > max_ctx:
        violations.append(f"requested context {context_tokens} exceeds max reliable context {max_ctx}")
    return violations


def route_warnings(row: Dict[str, Any], context_tokens: Optional[int]) -> List[str]:
    warnings: List[str] = []
    if row.get("audit_status") == "warn":
        warnings.append("source run audit status is warn")
    if context_tokens is not None and not int_or_none(row.get("max_reliable_context_tokens")):
        warnings.append("route has no measured max reliable context")
    return warnings


def decide_from_history(
    db: Path,
    task: str,
    context_tokens: Optional[int] = None,
    min_score: float = 0.55,
    require_audit_pass: bool = True,
    limit: int = 50,
) -> Dict[str, Any]:
    with lms_history.connect(db) as conn:
        rows = lms_history.best_routes(conn, task=task, limit=limit, min_score=0.0)
        if not rows and task != "general":
            rows = lms_history.best_routes(conn, task="general", limit=limit, min_score=0.0)
    evaluated: List[Dict[str, Any]] = []
    selected: Optional[Dict[str, Any]] = None
    fallback: Optional[Dict[str, Any]] = None

    for row in rows:
        violations = route_violations(row, context_tokens, min_score, require_audit_pass)
        warnings = route_warnings(row, context_tokens)
        item = {"route": row, "violations": violations, "warnings": warnings}
        evaluated.append(item)
        if not violations and selected is None:
            selected = row
        elif not violations and selected is not None and fallback is None:
            if row.get("model_key") != selected.get("model_key") or row.get("base_url") != selected.get("base_url"):
                fallback = row

    if selected:
        return {
            "decision": "allow",
            "reason": "selected route satisfies history policy",
            "source": "history",
            "db": str(db),
            "task_family": task,
            "selected": selected,
            "fallback": fallback,
            "warnings": route_warnings(selected, context_tokens),
            "evaluated": evaluated,
            "policy": {
                "context_tokens": context_tokens,
                "min_score": min_score,
                "require_audit_pass": require_audit_pass,
                "limit": limit,
            },
        }

    return {
        "decision": "block",
        "reason": "no historical route satisfies policy",
        "source": "history",
        "db": str(db),
        "task_family": task,
        "selected": None,
        "fallback": None,
        "warnings": [],
        "evaluated": evaluated,
        "policy": {
            "context_tokens": context_tokens,
            "min_score": min_score,
            "require_audit_pass": require_audit_pass,
            "limit": limit,
        },
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# LMS History Routing Decision",
        "",
        f"- Decision: `{result.get('decision')}`",
        f"- Reason: {result.get('reason')}",
        f"- Task: `{result.get('task_family')}`",
        f"- DB: `{result.get('db')}`",
        "",
    ]
    selected = result.get("selected") or {}
    if selected:
        lines += ["## Selected historical route", ""]
        lines.append(f"- Model: `{selected.get('model_key')}`")
        lines.append(f"- Endpoint: `{selected.get('base_url')}`")
        lines.append(f"- Score: `{selected.get('score')}`")
        lines.append(f"- Grade: `{selected.get('grade')}`")
        lines.append(f"- Audit: `{selected.get('audit_status')}`")
        lines.append(f"- Run: `{selected.get('run_id')}` / `{selected.get('run_dir')}`")
        lines.append(f"- Evidence: {selected.get('evidence')}")
    fallback = result.get("fallback") or {}
    if fallback:
        lines += ["", "## Fallback historical route", ""]
        lines.append(f"- Model: `{fallback.get('model_key')}`")
        lines.append(f"- Endpoint: `{fallback.get('base_url')}`")
        lines.append(f"- Score: `{fallback.get('score')}`")
        lines.append(f"- Run: `{fallback.get('run_id')}`")
    if result.get("warnings"):
        lines += ["", "## Warnings", ""]
        for warning in result["warnings"]:
            lines.append(f"- {warning}")
    if result.get("evaluated"):
        lines += ["", "## Evaluated historical routes", "", "| Model | Score | Audit | Run | Violations |", "|---|---:|---|---|---|"]
        for item in result["evaluated"][:30]:
            route = item.get("route") or {}
            lines.append(
                f"| `{route.get('model_key','')}` | {route.get('score','')} | {route.get('audit_status','')} | "
                f"`{route.get('run_id','')}` | {'; '.join(item.get('violations') or [])} |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Make an allow/block routing decision from the LMS history DB.")
    parser.add_argument("--db", default=None, help=f"History DB path; default {lms_history.DEFAULT_DB}")
    parser.add_argument("--task", default="general")
    parser.add_argument("--context-tokens", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--no-require-audit-pass", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    db = lms_history.db_path(args.db)
    result = decide_from_history(
        db,
        task=args.task,
        context_tokens=args.context_tokens,
        min_score=args.min_score,
        require_audit_pass=not args.no_require_audit_pass,
        limit=args.limit,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result.get("decision") == "allow" else 1


if __name__ == "__main__":
    raise SystemExit(main())
