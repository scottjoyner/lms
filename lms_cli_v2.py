#!/usr/bin/env python3
"""Upgraded LMS CLI wrapper.

This wrapper keeps the existing manifest-aware CLI intact while adding the next
agent-facing command layer:

  lms-bench fit latest
  lms-bench validate-suite
  lms-bench validate-run latest
  lms-bench brief latest
  lms-bench audit latest
  lms-bench export-skill latest
  lms-bench quick --from-registry --tags gpu

The wrapper delegates most existing commands to `lms_cli.main`.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path
from typing import List, Optional

import lms_agent_brief
import lms_artifact_validate
import lms_cli
import lms_endpoint_registry
import lms_manifest_validate
import lms_model_fit
import lms_run_audit
import lms_skill_export


VERSION = "lms-agent-cli 0.12.0"
MODULE_FILES = [
    "lms_cli_v2.py",
    "lms_cli.py",
    "lms_endpoint_registry.py",
    "lmstudio_cli_bridge.py",
    "lms_skill_export.py",
    "lms_run_audit.py",
    "lms_manifest_validate.py",
    "lms_artifact_validate.py",
    "lms_agent_brief.py",
    "lms_model_fit.py",
    "lms_machine_profile.py",
    "lms_eval.py",
    "benchmark_lmstudio_cross_machine_models.py",
]


def build_fit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms-bench fit", description="Estimate model-to-hardware fit for an LMS run directory.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
    return parser


def build_brief_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms-bench brief", description="Generate a single agent-facing LMS run brief.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--out", default=None)
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms-bench audit", description="Audit an LMS run directory for completeness and route-readiness.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-eval-ok", type=float, default=0.60)
    parser.add_argument("--no-require-safety", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_export_skill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms-bench export-skill", description="Export an LMS run as an agent-readable skill contract.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_validate_suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms-bench validate-suite", description="Validate an LMS benchmark suite manifest.")
    parser.add_argument("suite_file", nargs="?", default=None, help="Defaults to the bundled agent suite")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_validate_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms-bench validate-run", description="Validate generated run artifacts against JSON schemas.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--pretty", action="store_true")
    return parser


def run_fit(argv: List[str]) -> int:
    args = build_fit_parser().parse_args(argv)
    run_dir = lms_cli.resolve_run_dir(args.run_dir, args.runs_dir)
    cmd = [str(run_dir)]
    if args.csv_out:
        cmd += ["--csv-out", args.csv_out]
    if args.md_out:
        cmd += ["--md-out", args.md_out]
    return int(lms_model_fit.main(cmd))


def run_brief(argv: List[str]) -> int:
    args = build_brief_parser().parse_args(argv)
    run_dir = lms_cli.resolve_run_dir(args.run_dir, args.runs_dir)
    cmd = [str(run_dir)]
    if args.out:
        cmd += ["--out", args.out]
    return int(lms_agent_brief.main(cmd))


def run_audit(argv: List[str]) -> int:
    args = build_audit_parser().parse_args(argv)
    run_dir = lms_cli.resolve_run_dir(args.run_dir, args.runs_dir)
    cmd = [str(run_dir), "--min-score", str(args.min_score), "--min-eval-ok", str(args.min_eval_ok)]
    if args.no_require_safety:
        cmd.append("--no-require-safety")
    if args.pretty:
        cmd.append("--pretty")
    return int(lms_run_audit.main(cmd))


def run_export_skill(argv: List[str]) -> int:
    args = build_export_skill_parser().parse_args(argv)
    run_dir = lms_cli.resolve_run_dir(args.run_dir, args.runs_dir)
    cmd = [str(run_dir)]
    if args.json_out:
        cmd += ["--json-out", args.json_out]
    if args.md_out:
        cmd += ["--md-out", args.md_out]
    if args.pretty:
        cmd += ["--pretty"]
    return int(lms_skill_export.main(cmd))


def run_validate_suite(argv: List[str]) -> int:
    args = build_validate_suite_parser().parse_args(argv)
    suite = Path(args.suite_file) if args.suite_file else lms_cli.resolve_asset(lms_cli.DEFAULT_SUITE)
    cmd = [str(suite)]
    if args.json_out:
        cmd += ["--json-out", args.json_out]
    if args.md_out:
        cmd += ["--md-out", args.md_out]
    if args.pretty:
        cmd += ["--pretty"]
    return int(lms_manifest_validate.main(cmd))


def run_validate_run(argv: List[str]) -> int:
    args = build_validate_run_parser().parse_args(argv)
    run_dir = lms_cli.resolve_run_dir(args.run_dir, args.runs_dir)
    cmd = ["run", str(run_dir)]
    if args.pretty:
        cmd.append("--pretty")
    return int(lms_artifact_validate.main(cmd))


def parse_quick_output_dir(argv: List[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    try:
        args, _ = parser.parse_known_args(argv)
        return args.output_dir
    except SystemExit:
        return lms_cli.DEFAULT_RUNS_DIR


def quick_is_profile_only(argv: List[str]) -> bool:
    return "--profile-only" in argv


def registry_quick_args(argv: List[str]) -> Optional[List[str]]:
    if "--from-registry" not in argv:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--from-registry", action="store_true")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--registry-name", action="append", default=[])
    parser.add_argument("--tags", default=None)
    parser.add_argument("--all-endpoints", action="store_true")
    parsed, remainder = parser.parse_known_args(argv)
    registry = lms_endpoint_registry.load_registry(lms_endpoint_registry.registry_path(parsed.registry))
    endpoints = lms_endpoint_registry.select_endpoints(
        registry,
        parsed.registry_name,
        lms_endpoint_registry.split_tags(parsed.tags),
        enabled_only=not parsed.all_endpoints,
    )
    if not endpoints:
        raise SystemExit("no registry endpoints selected; add endpoints with lms-bench-endpoints add")
    translated = ["quick"]
    for endpoint in endpoints:
        translated += ["--endpoint", endpoint["base_url"]]
    translated += [arg for arg in remainder if arg != "--from-registry"]
    return translated


def postprocess_latest_run(runs_dir: str) -> None:
    run_dir = lms_cli.resolve_run_dir("latest", runs_dir)
    print("\nRunning model-fit analysis for latest run...")
    fit_rc = int(lms_model_fit.main([str(run_dir)]))
    if fit_rc != 0:
        print("model-fit analysis completed with warnings or no models found")
    print("\nGenerating agent brief...")
    lms_agent_brief.main([str(run_dir)])
    print("\nAuditing run artifacts...")
    audit_rc = int(lms_run_audit.main([str(run_dir)]))
    if audit_rc != 0:
        print("run audit reported critical issues")
    print("\nExporting agent skill contract...")
    lms_skill_export.main([str(run_dir)])
    print("\nValidating generated JSON artifacts...")
    validation_rc = int(lms_artifact_validate.main(["run", str(run_dir)]))
    if validation_rc != 0:
        print("artifact validation reported schema issues")


def run_selftest() -> int:
    failures = []
    for rel in MODULE_FILES:
        path = lms_cli.resolve_asset(Path(rel))
        if not path.exists():
            failures.append(f"missing module: {rel}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            failures.append(f"compile failed for {rel}: {exc}")
    suite_rc = run_validate_suite(["--pretty"])
    if suite_rc != 0:
        failures.append("suite validation failed")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("selftest passed: modules compile, suite manifest is valid, evaluator registry is loadable")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return lms_cli.main(args)
    if args[0] in {"--version", "-V"}:
        print(VERSION)
        return 0
    if args[0] == "fit":
        return run_fit(args[1:])
    if args[0] == "brief":
        return run_brief(args[1:])
    if args[0] == "audit":
        return run_audit(args[1:])
    if args[0] in {"export-skill", "skill"}:
        return run_export_skill(args[1:])
    if args[0] in {"validate-run", "validate-artifacts"}:
        return run_validate_run(args[1:])
    if args[0] in {"validate-suite", "validate"}:
        return run_validate_suite(args[1:])
    if args[0] == "selftest":
        return run_selftest()

    delegated_args = args
    if args[0] == "quick":
        registry_args = registry_quick_args(args[1:])
        if registry_args:
            delegated_args = registry_args

    rc = int(lms_cli.main(delegated_args))
    if delegated_args[0] == "quick" and rc == 0 and not quick_is_profile_only(delegated_args[1:]):
        runs_dir = parse_quick_output_dir(delegated_args[1:])
        try:
            postprocess_latest_run(runs_dir)
        except Exception as exc:
            print(f"post-run analysis skipped: {exc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
