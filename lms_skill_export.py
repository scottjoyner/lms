#!/usr/bin/env python3
"""Export an LMS run as a compact agent skill contract.

The skill export is meant for downstream agents that should not parse Markdown.
It provides task routes, thresholds, fit warnings, audit status, and commands for
refreshing the evidence.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def best_routes(capabilities: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in capabilities:
        grouped.setdefault(row.get("task_family", "general"), []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for task, rows in grouped.items():
        rows = sorted(rows, key=lambda r: fnum(r.get("score")), reverse=True)
        if not rows:
            continue
        preferred = rows[0]
        fallback = None
        for candidate in rows[1:]:
            if candidate.get("model_key") != preferred.get("model_key") or candidate.get("base_url") != preferred.get("base_url"):
                fallback = candidate
                break
        out[task] = {
            "preferred": preferred,
            "fallback": fallback,
            "min_score": 0.55,
            "refresh_command": "lms quick",
            "route_command": f"lms route latest --task {task}",
        }
    return out


def fit_warnings(fit_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    warnings = []
    for row in fit_rows:
        if row.get("fit_grade") in {"poor", "risky", "borderline", "unknown"}:
            warnings.append(row)
    return warnings


def export_skill(run_dir: Path) -> Dict[str, Any]:
    config = read_json(run_dir / "lms_run_config.json")
    profile = read_json(run_dir / "machine_profile.json")
    audit = read_json(run_dir / "run_audit.json")
    routes = read_json(run_dir / "routing_rules.json")
    capabilities = read_csv(run_dir / "capability_matrix.csv")
    fit = read_csv(run_dir / "model_fit.csv")

    host = profile.get("host") or {}
    skill = {
        "schema_version": "lms_agent_skill.v1",
        "generated_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "run_id": config.get("run_id") or run_dir.name,
        "machine": {
            "hostname": host.get("hostname"),
            "platform": host.get("platform"),
            "python_version": host.get("python_version"),
        },
        "audit": {
            "status": audit.get("status", "unknown"),
            "ok": audit.get("ok", False),
            "critical": audit.get("critical", []),
            "warnings": audit.get("warnings", []),
        },
        "routes": routes.get("routing") or best_routes(capabilities),
        "fit_warnings": fit_warnings(fit),
        "operating_rules": [
            "Use task-specific routes before general routes.",
            "Do not use a route if the audit status is fail unless a human overrides it.",
            "Prefer evaluator score over raw throughput for autonomous work.",
            "Do not exceed max_reliable_context_tokens when present.",
            "Use fallback_model when preferred route is unavailable, slow, malformed, or below threshold.",
            "Re-run lms quick after model, driver, hardware, or LM Studio changes.",
        ],
        "commands": {
            "refresh": "lms quick",
            "audit": "lms audit latest --pretty",
            "brief": "lms brief latest",
            "fit": "lms fit latest",
            "compare": "lms compare <old_run> <new_run> --show",
        },
        "artifact_paths": {
            "agent_brief": str(run_dir / "AGENT_BRIEF.md"),
            "audit_json": str(run_dir / "run_audit.json"),
            "capability_matrix": str(run_dir / "capability_matrix.csv"),
            "routing_rules": str(run_dir / "routing_rules.json"),
            "model_fit": str(run_dir / "model_fit.csv"),
        },
    }
    return skill


def render_markdown(skill: Dict[str, Any]) -> str:
    lines = ["# LMS Agent Skill Export", "", f"- Schema: `{skill.get('schema_version')}`", f"- Generated: `{skill.get('generated_at_utc')}`", f"- Run: `{skill.get('run_dir')}`", f"- Audit: `{skill.get('audit', {}).get('status')}`", ""]
    lines += ["## Operating rules", ""]
    for rule in skill.get("operating_rules", []):
        lines.append(f"- {rule}")
    lines += ["", "## Routes", ""]
    routes = skill.get("routes") or {}
    if isinstance(routes, dict):
        lines += ["| Task | Preferred | Fallback | Score |", "|---|---|---|---:|"]
        for task, bundle in sorted(routes.items()):
            preferred = bundle.get("preferred") if isinstance(bundle, dict) else None
            fallback = bundle.get("fallback") if isinstance(bundle, dict) else None
            if preferred is None and isinstance(bundle, dict):
                preferred = bundle
            lines.append(
                f"| `{task}` | `{(preferred or {}).get('model_key','')}` | "
                f"`{(fallback or {}).get('model_key','') if fallback else ''}` | {(preferred or {}).get('score','')} |"
            )
    if skill.get("fit_warnings"):
        lines += ["", "## Fit warnings", ""]
        for row in skill["fit_warnings"][:20]:
            lines.append(f"- `{row.get('model_key')}`: {row.get('fit_grade')} — {row.get('fit_notes')}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an LMS run as an agent-readable skill contract.")
    parser.add_argument("run_dir")
    parser.add_argument("--json-out", default=None, help="Default: run_dir/lms_agent_skill.json")
    parser.add_argument("--md-out", default=None, help="Default: run_dir/LMS_AGENT_SKILL.md")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    skill = export_skill(run_dir)
    json_out = Path(args.json_out) if args.json_out else run_dir / "lms_agent_skill.json"
    md_out = Path(args.md_out) if args.md_out else run_dir / "LMS_AGENT_SKILL.md"
    json_out.write_text(json.dumps(skill, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(render_markdown(skill), encoding="utf-8")
    print(json.dumps(skill, indent=2 if args.pretty else None, sort_keys=True))
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
