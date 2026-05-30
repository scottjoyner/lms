#!/usr/bin/env python3
"""
Deterministic evaluators for LMS agent benchmark outputs.

These evaluators are intentionally simple, local, and dependency-free. They are
used before any optional LLM-as-judge pass so core quality signals remain
reproducible and cheap.

Supported evaluator types:
  - exact_contains
  - contains_all
  - max_chars
  - min_chars
  - json_parse
  - json_required_keys
  - json_forbidden_extra_keys
  - no_markdown_fence
  - regex_contains
  - regex_not_contains

CLI examples:
  echo '{"x": 1}' | python3 lms_eval.py --evaluators-json '[{"type":"json_parse"}]'
  python3 lms_eval.py --output-file raw.txt --evaluators-file evals.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class EvalResult:
    evaluator_type: str
    ok: bool
    score: float
    message: str
    details: Dict[str, Any]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_json_output(output_text: str) -> Tuple[Optional[Any], Optional[str]]:
    text = output_text.strip()
    if not text:
        return None, "empty output"

    # Models sometimes wrap strict JSON in fences. json_parse should be strict,
    # but this helper includes a second extraction attempt so reports can explain
    # whether JSON existed but was surrounded by invalid wrapper text.
    try:
        return json.loads(text), None
    except json.JSONDecodeError as strict_exc:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                return parsed, f"json was fenced or surrounded by markdown: {strict_exc}"
            except json.JSONDecodeError:
                pass
        return None, str(strict_exc)


def eval_exact_contains(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    expected = str(spec.get("value", ""))
    ok = expected in output_text
    return EvalResult(
        evaluator_type="exact_contains",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="expected text found" if ok else "expected text missing",
        details={"value": expected},
    )


def eval_contains_all(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    values = [str(v) for v in _as_list(spec.get("value"))]
    missing = [v for v in values if v not in output_text]
    ok = not missing
    score = 1.0 if ok else (len(values) - len(missing)) / len(values) if values else 1.0
    return EvalResult(
        evaluator_type="contains_all",
        ok=ok,
        score=score,
        message="all required strings found" if ok else "some required strings missing",
        details={"required": values, "missing": missing},
    )


def eval_max_chars(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    limit = int(spec.get("value", 0))
    actual = len(output_text)
    ok = actual <= limit
    return EvalResult(
        evaluator_type="max_chars",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="output within max char limit" if ok else "output exceeded max char limit",
        details={"limit": limit, "actual": actual},
    )


def eval_min_chars(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    limit = int(spec.get("value", 0))
    actual = len(output_text)
    ok = actual >= limit
    return EvalResult(
        evaluator_type="min_chars",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="output met min char limit" if ok else "output below min char limit",
        details={"limit": limit, "actual": actual},
    )


def eval_json_parse(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    parsed, error = _parse_json_output(output_text)
    ok = parsed is not None and error is None
    recoverable = parsed is not None and error is not None
    return EvalResult(
        evaluator_type="json_parse",
        ok=ok,
        score=1.0 if ok else 0.5 if recoverable else 0.0,
        message="valid strict JSON" if ok else "JSON found but not strict output" if recoverable else "invalid JSON",
        details={"error": error, "parsed_type": type(parsed).__name__ if parsed is not None else None},
    )


def eval_json_required_keys(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    parsed, error = _parse_json_output(output_text)
    required = [str(v) for v in _as_list(spec.get("value"))]
    if not isinstance(parsed, dict):
        return EvalResult(
            evaluator_type="json_required_keys",
            ok=False,
            score=0.0,
            message="output is not a JSON object",
            details={"error": error, "required": required},
        )
    missing = [key for key in required if key not in parsed]
    ok = not missing
    score = 1.0 if ok else (len(required) - len(missing)) / len(required) if required else 1.0
    return EvalResult(
        evaluator_type="json_required_keys",
        ok=ok,
        score=score,
        message="all required JSON keys found" if ok else "required JSON keys missing",
        details={"required": required, "missing": missing},
    )


def eval_json_forbidden_extra_keys(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    parsed, error = _parse_json_output(output_text)
    allowed = [str(v) for v in _as_list(spec.get("value"))]
    if not isinstance(parsed, dict):
        return EvalResult(
            evaluator_type="json_forbidden_extra_keys",
            ok=False,
            score=0.0,
            message="output is not a JSON object",
            details={"error": error, "allowed": allowed},
        )
    extra = [key for key in parsed.keys() if key not in allowed]
    ok = not extra
    return EvalResult(
        evaluator_type="json_forbidden_extra_keys",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="no extra JSON keys found" if ok else "extra JSON keys found",
        details={"allowed": allowed, "extra": extra},
    )


def eval_no_markdown_fence(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    ok = "```" not in output_text
    return EvalResult(
        evaluator_type="no_markdown_fence",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="no markdown fence found" if ok else "markdown fence found",
        details={},
    )


def eval_regex_contains(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    pattern = str(spec.get("value", ""))
    flags = re.DOTALL | (re.IGNORECASE if spec.get("ignore_case") else 0)
    ok = re.search(pattern, output_text, flags=flags) is not None
    return EvalResult(
        evaluator_type="regex_contains",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="regex matched" if ok else "regex did not match",
        details={"pattern": pattern},
    )


def eval_regex_not_contains(output_text: str, spec: Dict[str, Any]) -> EvalResult:
    pattern = str(spec.get("value", ""))
    flags = re.DOTALL | (re.IGNORECASE if spec.get("ignore_case") else 0)
    ok = re.search(pattern, output_text, flags=flags) is None
    return EvalResult(
        evaluator_type="regex_not_contains",
        ok=ok,
        score=1.0 if ok else 0.0,
        message="regex absent" if ok else "forbidden regex matched",
        details={"pattern": pattern},
    )


EVALUATORS = {
    "exact_contains": eval_exact_contains,
    "contains_all": eval_contains_all,
    "max_chars": eval_max_chars,
    "min_chars": eval_min_chars,
    "json_parse": eval_json_parse,
    "json_required_keys": eval_json_required_keys,
    "json_forbidden_extra_keys": eval_json_forbidden_extra_keys,
    "no_markdown_fence": eval_no_markdown_fence,
    "regex_contains": eval_regex_contains,
    "regex_not_contains": eval_regex_not_contains,
}


def evaluate_output(output_text: str, evaluator_specs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    results: List[EvalResult] = []
    for spec in evaluator_specs:
        evaluator_type = spec.get("type")
        fn = EVALUATORS.get(str(evaluator_type))
        if not fn:
            results.append(
                EvalResult(
                    evaluator_type=str(evaluator_type),
                    ok=False,
                    score=0.0,
                    message="unknown evaluator type",
                    details={"spec": spec},
                )
            )
            continue
        try:
            results.append(fn(output_text, spec))
        except Exception as exc:
            results.append(
                EvalResult(
                    evaluator_type=str(evaluator_type),
                    ok=False,
                    score=0.0,
                    message="evaluator raised exception",
                    details={"error": repr(exc), "spec": spec},
                )
            )

    result_dicts = [asdict(r) for r in results]
    if not result_dicts:
        return {"ok": True, "score": 1.0, "results": [], "failed": []}

    failed = [r for r in result_dicts if not r["ok"]]
    avg_score = sum(float(r["score"]) for r in result_dicts) / len(result_dicts)
    return {
        "ok": not failed,
        "score": round(avg_score, 4),
        "results": result_dicts,
        "failed": failed,
    }


def load_evaluator_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.evaluators_file:
        data = json.loads(Path(args.evaluators_file).read_text(encoding="utf-8"))
    elif args.evaluators_json:
        data = json.loads(args.evaluators_json)
    else:
        raise SystemExit("provide --evaluators-json or --evaluators-file")

    if not isinstance(data, list):
        raise SystemExit("evaluator spec must be a JSON array")
    for item in data:
        if not isinstance(item, dict) or "type" not in item:
            raise SystemExit("each evaluator spec must be an object with a type")
    return data


def load_output_text(args: argparse.Namespace) -> str:
    if args.output_file:
        return Path(args.output_file).read_text(encoding="utf-8", errors="replace")
    return sys.stdin.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic LMS evaluators against model output.")
    parser.add_argument("--output-file", default=None, help="File containing model output. Defaults to stdin.")
    parser.add_argument("--evaluators-json", default=None, help="JSON array of evaluator specs")
    parser.add_argument("--evaluators-file", default=None, help="Path to JSON array of evaluator specs")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON result")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    specs = load_evaluator_specs(args)
    output_text = load_output_text(args)
    result = evaluate_output(output_text, specs)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
