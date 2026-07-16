#!/usr/bin/env python3
"""Validate LMS benchmark suite manifests.

This validator catches the common mistakes that make agent benchmarks hard to
trust: duplicate case keys, missing evaluator fields, unknown evaluator types,
invalid context sweeps, and missing task metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from lms_eval import EVALUATORS
except Exception:  # pragma: no cover
    EVALUATORS = {}


REQUIRED_CASE_FIELDS = ["case_key", "priority", "task_family", "system", "max_output_tokens", "evaluators", "recommendation_signal"]
PROMPT_FIELDS = ["prompt", "prompt_template"]


def load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest root must be an object")
    return data


def validate_manifest(data: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("manifest must include a non-empty cases array")
        cases = []

    seen = set()
    task_families = set()
    case_count = 0
    evaluator_count = 0

    for idx, case in enumerate(cases):
        case_count += 1
        prefix = f"cases[{idx}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        case_key = str(case.get("case_key", "")).strip()
        if not case_key:
            errors.append(f"{prefix}.case_key is required")
        elif case_key in seen:
            errors.append(f"duplicate case_key: {case_key}")
        else:
            seen.add(case_key)

        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                errors.append(f"{case_key or prefix}.{field} is required")
        if not any(field in case for field in PROMPT_FIELDS):
            errors.append(f"{case_key or prefix} must include prompt or prompt_template")
        if "prompt" in case and "prompt_template" in case:
            warnings.append(f"{case_key} includes both prompt and prompt_template; prompt_template will be ignored by some tooling")

        task_family = str(case.get("task_family", "")).strip()
        if task_family:
            task_families.add(task_family)
        else:
            errors.append(f"{case_key or prefix}.task_family must be non-empty")

        try:
            max_tokens = int(case.get("max_output_tokens", 0))
            if max_tokens <= 0:
                errors.append(f"{case_key or prefix}.max_output_tokens must be > 0")
        except (TypeError, ValueError):
            errors.append(f"{case_key or prefix}.max_output_tokens must be an integer")

        evaluators = case.get("evaluators")
        if not isinstance(evaluators, list):
            errors.append(f"{case_key or prefix}.evaluators must be an array")
            evaluators = []
        if not evaluators:
            warnings.append(f"{case_key or prefix} has no evaluators; it will score as pass-through")
        for eidx, evaluator in enumerate(evaluators):
            evaluator_count += 1
            if not isinstance(evaluator, dict):
                errors.append(f"{case_key}.evaluators[{eidx}] must be an object")
                continue
            etype = str(evaluator.get("type", ""))
            if not etype:
                errors.append(f"{case_key}.evaluators[{eidx}].type is required")
            elif EVALUATORS and etype not in EVALUATORS:
                errors.append(f"{case_key}.evaluators[{eidx}] unknown evaluator type: {etype}")

        if "context_sweep_tokens" in case:
            sweep = case.get("context_sweep_tokens")
            if not isinstance(sweep, list) or not sweep:
                errors.append(f"{case_key}.context_sweep_tokens must be a non-empty array")
            else:
                last = 0
                for value in sweep:
                    try:
                        intval = int(value)
                        if intval <= 0:
                            errors.append(f"{case_key}.context_sweep_tokens contains non-positive value: {value}")
                        if intval < last:
                            warnings.append(f"{case_key}.context_sweep_tokens is not sorted ascending")
                        last = intval
                    except (TypeError, ValueError):
                        errors.append(f"{case_key}.context_sweep_tokens contains non-integer value: {value}")
            if "prompt_template" not in case:
                warnings.append(f"{case_key} has context_sweep_tokens but no prompt_template")
            if "synthetic_context" not in case:
                warnings.append(f"{case_key} has context_sweep_tokens but no synthetic_context")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "case_count": case_count,
        "task_families": sorted(task_families),
        "evaluator_count": evaluator_count,
        "suite_id": data.get("suite_id"),
        "version": data.get("version"),
    }


def render_markdown(path: Path, result: Dict[str, Any]) -> str:
    lines = ["# LMS Suite Validation", "", f"- Manifest: `{path}`", f"- Suite: `{result.get('suite_id')}`", f"- Version: `{result.get('version')}`", f"- OK: `{result.get('ok')}`", f"- Cases: `{result.get('case_count')}`", f"- Evaluators: `{result.get('evaluator_count')}`", ""]
    lines += ["## Task families", ""]
    for family in result.get("task_families", []):
        lines.append(f"- `{family}`")
    if result.get("errors"):
        lines += ["", "## Errors", ""]
        for item in result["errors"]:
            lines.append(f"- {item}")
    if result.get("warnings"):
        lines += ["", "## Warnings", ""]
        for item in result["warnings"]:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an LMS benchmark suite manifest.")
    parser.add_argument("suite_file", help="Path to benchmark suite JSON")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.suite_file)
    try:
        result = validate_manifest(load_manifest(path))
    except Exception as exc:
        result = {"ok": False, "errors": [repr(exc)], "warnings": [], "case_count": 0, "task_families": [], "evaluator_count": 0, "suite_id": None, "version": None}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(path, result), encoding="utf-8")
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
