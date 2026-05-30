#!/usr/bin/env python3
"""Upgraded LMS CLI wrapper.

This wrapper keeps the existing manifest-aware CLI intact while adding the next
agent-facing command layer:

  lms fit latest
  lms quick ...   # also emits model_fit.csv/model_fit.md after successful runs

The wrapper delegates all existing commands to `lms_cli.main`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import lms_cli
import lms_model_fit


VERSION = "lms-agent-cli 0.5.0"


def build_fit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lms fit", description="Estimate model-to-hardware fit for an LMS run directory.")
    parser.add_argument("run_dir", nargs="?", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--runs-dir", default=lms_cli.DEFAULT_RUNS_DIR)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
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


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return lms_cli.main(args)
    if args[0] in {"--version", "-V"}:
        print(VERSION)
        return 0
    if args[0] == "fit":
        return run_fit(args[1:])

    rc = int(lms_cli.main(args))
    if args[0] == "quick" and rc == 0 and not quick_is_profile_only(args[1:]):
        runs_dir = parse_quick_output_dir(args[1:])
        try:
            run_dir = lms_cli.resolve_run_dir("latest", runs_dir)
            print("\nRunning model-fit analysis for latest run...")
            fit_rc = int(lms_model_fit.main([str(run_dir)]))
            if fit_rc != 0:
                print("model-fit analysis completed with warnings or no models found")
        except Exception as exc:
            print(f"model-fit analysis skipped: {exc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
