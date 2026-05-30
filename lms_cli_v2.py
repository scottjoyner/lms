#!/usr/bin/env python3
"""Upgraded LMS CLI wrapper.

This wrapper keeps the existing manifest-aware CLI intact while adding the next
agent-facing command layer:

  lms fit latest
  lms validate-suite
  lms brief latest
  lms audit latest
  lms export-skill latest
  lms quick ...   # also emits model_fit + AGENT_BRIEF + RUN_AUDIT + skill export

The wrapper delegates all existing commands to `lms_cli.main`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import lms_agent_brief
import lms_cli
import lms_manifest_validate
import lms_model_fit
import lms_run_audit
import lms_skill_export


VERSION = "lms-agent-cli 0.8.0"


def build_fit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms fit", description="Estimate model-to-hardware fit for an LMS run directory.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
    return parser


def build_brief_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms brief", description="Generate a single agent-facing LMS run brief.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--out", default=None)
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms audit", description="Audit an LMS run directory for completeness and route-readiness.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-eval-ok", type=float, default=0.60)
    parser.add_argument("--no-require-safety", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_export_skill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms export-skill", description="Export an LMS run as an agent-readable skill contract.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser


def build_validate_suite_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms validate-suite", description="Validate an LMS benchmark suite manifest.")
    parser.add_argument("suite_file", nargs="?", default=None, help="Defaults to the bundled agent suite")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
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
    if args[0] in {"validate-suite", "validate"}:
        return run_validate_suite(args[1:])
    if args[0] == "selftest":
        rc = run_validate_suite(["--pretty"])
        if rc != 0:
            return rc
        print("selftest passed: suite manifest is valid and evaluator registry is loadable")
        return 0

    rc = int(lms_cli.main(args))
    if args[0] == "quick" and rc == 0 and not quick_is_profile_only(args[1:]):
        runs_dir = parse_quick_output_dir(args[1:])
        try:
            postprocess_latest_run(runs_dir)
        except Exception as exc:
            print(f"post-run analysis skipped: {exc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
