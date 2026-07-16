#!/usr/bin/env python3
"""Validate LMS generated artifacts against versioned JSON schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None


SCHEMA_MAP = {
    "endpoint_registry": ("schemas/endpoint_registry.schema.json", "endpoints.json"),
    "run_audit": ("schemas/run_audit.schema.json", "run_audit.json"),
    "lms_agent_skill": ("schemas/lms_agent_skill.schema.json", "lms_agent_skill.json"),
    "routing_rules": ("schemas/routing_rules.schema.json", "routing_rules.json"),
    "machine_profile": ("schemas/machine_profile.schema.json", "machine_profile.json"),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema(schema_path: Path) -> Dict[str, Any]:
    return read_json(schema_path)


def validate_one(schema_path: Path, artifact_path: Path) -> Dict[str, Any]:
    result = {
        "schema": str(schema_path),
        "artifact": str(artifact_path),
        "ok": False,
        "errors": [],
    }
    if not schema_path.exists():
        result["errors"].append(f"schema not found: {schema_path}")
        return result
    if not artifact_path.exists():
        result["errors"].append(f"artifact not found: {artifact_path}")
        return result
    try:
        schema = load_schema(schema_path)
        artifact = read_json(artifact_path)
        if jsonschema is None:
            # Minimal fallback: JSON is parseable and required fields exist.
            for key in schema.get("required", []):
                if key not in artifact:
                    result["errors"].append(f"missing required key: {key}")
        else:
            validator = jsonschema.Draft202012Validator(schema)
            for err in sorted(validator.iter_errors(artifact), key=lambda e: list(e.path)):
                path = ".".join(str(p) for p in err.path) or "$"
                result["errors"].append(f"{path}: {err.message}")
        result["ok"] = not result["errors"]
        return result
    except Exception as exc:
        result["errors"].append(repr(exc))
        return result


def validate_run(run_dir: Path, schema_root: Optional[Path] = None) -> Dict[str, Any]:
    schema_root = schema_root or repo_root()
    checks: List[Dict[str, Any]] = []
    for name in ["run_audit", "lms_agent_skill", "routing_rules", "machine_profile"]:
        schema_rel, artifact_name = SCHEMA_MAP[name]
        checks.append(validate_one(schema_root / schema_rel, run_dir / artifact_name))
    return {
        "ok": all(c["ok"] for c in checks),
        "run_dir": str(run_dir),
        "checks": checks,
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# LMS Artifact Validation", "", f"- OK: `{result.get('ok')}`"]
    if result.get("run_dir"):
        lines.append(f"- Run: `{result.get('run_dir')}`")
    lines.append("")
    lines += ["| Artifact | Schema | OK | Errors |", "|---|---|:---:|---|"]
    for check in result.get("checks", []):
        errors = "<br>".join(check.get("errors") or [])
        lines.append(f"| `{check.get('artifact')}` | `{check.get('schema')}` | {'yes' if check.get('ok') else 'no'} | {errors} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LMS artifacts against JSON schemas.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Validate generated artifacts in a run directory")
    run.add_argument("run_dir")
    run.add_argument("--schema-root", default=None)
    run.add_argument("--json-out", default=None)
    run.add_argument("--md-out", default=None)
    run.add_argument("--pretty", action="store_true")

    one = sub.add_parser("one", help="Validate one artifact against one schema")
    one.add_argument("schema")
    one.add_argument("artifact")
    one.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "one":
        result = {"ok": False, "checks": [validate_one(Path(args.schema), Path(args.artifact))]}
        result["ok"] = result["checks"][0]["ok"]
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0 if result["ok"] else 1

    schema_root = Path(args.schema_root) if args.schema_root else None
    result = validate_run(Path(args.run_dir), schema_root=schema_root)
    json_out = Path(args.json_out) if args.json_out else Path(args.run_dir) / "artifact_validation.json"
    md_out = Path(args.md_out) if args.md_out else Path(args.run_dir) / "ARTIFACT_VALIDATION.md"
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
