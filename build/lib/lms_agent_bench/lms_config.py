#!/usr/bin/env python3
"""Configuration helper for lms-bench.

This module keeps configuration simple and explainable. It supports defaults,
optional JSON config files, and environment variable overrides. CLI flags still
belong to the top-level commands; this utility tells agents what baseline config
will be used before flags are applied.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CONFIG_PATH = Path(os.environ.get("LMS_BENCH_CONFIG", "~/.config/lms-bench/config.json")).expanduser()

DEFAULTS: Dict[str, Any] = {
    "schema_version": "lms_bench_config.v1",
    "default_endpoint": "http://127.0.0.1:1234/v1",
    "runs_dir": "runs",
    "max_models": 3,
    "repeats": 1,
    "timeout": 900,
    "max_context_tokens": 8192,
    "registry_path": "~/.config/lms-bench/endpoints.json",
    "suite_file": "benchmarks/agent_skill_suite.v1.json",
    "require_safety": True,
    "min_score": 0.55,
    "min_eval_ok": 0.60,
}

ENV_MAP = {
    "default_endpoint": ["LMS_BENCH_ENDPOINT", "LMS_BASE_URL", "LMSTUDIO_BASE_URL"],
    "runs_dir": ["LMS_BENCH_RUNS_DIR"],
    "max_models": ["LMS_BENCH_MAX_MODELS"],
    "repeats": ["LMS_BENCH_REPEATS"],
    "timeout": ["LMS_BENCH_TIMEOUT"],
    "max_context_tokens": ["LMS_BENCH_MAX_CONTEXT_TOKENS"],
    "registry_path": ["LMS_BENCH_ENDPOINTS"],
    "suite_file": ["LMS_BENCH_SUITE_FILE"],
    "min_score": ["LMS_BENCH_MIN_SCORE"],
    "min_eval_ok": ["LMS_BENCH_MIN_EVAL_OK"],
}

INT_FIELDS = {"max_models", "repeats", "timeout", "max_context_tokens"}
FLOAT_FIELDS = {"min_score", "min_eval_ok"}
BOOL_FIELDS = {"require_safety"}


def config_path(path: Optional[str]) -> Path:
    return Path(path).expanduser() if path else DEFAULT_CONFIG_PATH


def load_file_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return value
    if key in INT_FIELDS:
        return int(value)
    if key in FLOAT_FIELDS:
        return float(value)
    if key in BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    return value


def env_overrides() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, env_names in ENV_MAP.items():
        for env_name in env_names:
            if env_name in os.environ and os.environ[env_name] != "":
                out[key] = {"value": coerce_value(key, os.environ[env_name]), "source": f"env:{env_name}"}
                break
    return out


def effective_config(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or DEFAULT_CONFIG_PATH
    file_config = load_file_config(path)
    effective: Dict[str, Any] = {}
    sources: Dict[str, str] = {}

    for key, value in DEFAULTS.items():
        effective[key] = value
        sources[key] = "default"

    for key, value in file_config.items():
        if key in DEFAULTS:
            effective[key] = coerce_value(key, value)
            sources[key] = str(path)

    for key, payload in env_overrides().items():
        effective[key] = payload["value"]
        sources[key] = payload["source"]

    return {
        "schema_version": "lms_bench_effective_config.v1",
        "config_path": str(path),
        "config_file_exists": path.exists(),
        "values": effective,
        "sources": sources,
    }


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    values = config.get("values", config)
    errors: List[str] = []
    warnings: List[str] = []
    if not str(values.get("default_endpoint", "")).startswith(("http://", "https://")):
        errors.append("default_endpoint must start with http:// or https://")
    if int(values.get("max_models", 0)) < 0:
        errors.append("max_models must be >= 0")
    if int(values.get("repeats", 0)) <= 0:
        errors.append("repeats must be > 0")
    if int(values.get("timeout", 0)) <= 0:
        errors.append("timeout must be > 0")
    if int(values.get("max_context_tokens", 0)) <= 0:
        errors.append("max_context_tokens must be > 0")
    if not (0 <= float(values.get("min_score", 0)) <= 1):
        errors.append("min_score must be between 0 and 1")
    if not (0 <= float(values.get("min_eval_ok", 0)) <= 1):
        errors.append("min_eval_ok must be between 0 and 1")
    suite = Path(str(values.get("suite_file", "")))
    if not suite.exists():
        warnings.append(f"suite_file does not exist from current directory: {suite}")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def render_markdown(config: Dict[str, Any], validation: Dict[str, Any]) -> str:
    lines = ["# LMS Bench Effective Config", "", f"- Config path: `{config.get('config_path')}`", f"- Config file exists: `{config.get('config_file_exists')}`", f"- Valid: `{validation.get('ok')}`", ""]
    lines += ["| Key | Value | Source |", "|---|---|---|"]
    values = config.get("values", {})
    sources = config.get("sources", {})
    for key in sorted(values):
        lines.append(f"| `{key}` | `{values[key]}` | `{sources.get(key, '')}` |")
    if validation.get("errors"):
        lines += ["", "## Errors", ""]
        for item in validation["errors"]:
            lines.append(f"- {item}")
    if validation.get("warnings"):
        lines += ["", "## Warnings", ""]
        for item in validation["warnings"]:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def cmd_show(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    cfg = effective_config(path)
    validation = validate_config(cfg)
    if args.md_out:
        Path(args.md_out).write_text(render_markdown(cfg, validation), encoding="utf-8")
    if args.pretty:
        print(json.dumps({"config": cfg, "validation": validation}, indent=2, sort_keys=True))
    else:
        print(json.dumps({"config": cfg, "validation": validation}, sort_keys=True))
    return 0 if validation.get("ok") else 1


def cmd_init(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    if path.exists() and not args.force:
        print(f"config already exists: {path}; use --force to overwrite")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULTS, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    path = config_path(args.config)
    cfg = effective_config(path)
    validation = validate_config(cfg)
    print(json.dumps(validation, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if validation.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show, initialize, and validate lms-bench configuration.")
    parser.add_argument("--config", default=None, help=f"Config file path; default {DEFAULT_CONFIG_PATH}")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show")
    show.add_argument("--pretty", action="store_true")
    show.add_argument("--md-out", default=None)
    show.set_defaults(func=cmd_show)

    init = sub.add_parser("init")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    validate = sub.add_parser("validate")
    validate.add_argument("--pretty", action="store_true")
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
