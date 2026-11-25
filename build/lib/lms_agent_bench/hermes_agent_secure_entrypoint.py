"""Secret-safe wrapper for the Hermes benchmark CLI.

API keys are resolved from an inherited environment variable and are never
placed in the trial subprocess argument vector. The legacy ``--api-key`` option
remains accepted only for the nonsecret default value ``local-benchmark``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from lms_agent_bench import hermes_agent_bench as _base

DEFAULT_API_KEY_ENV = "HERMES_BENCH_API_KEY"
_PATCHED = False


def _parser() -> argparse.ArgumentParser:
    parser = _base.build_parser()
    sub_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    run = sub_action.choices["run"]
    hidden = sub_action.choices["_trial"]
    run.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    hidden.add_argument("--api-key-env", required=True)
    hidden_api_key = next(
        action
        for action in hidden._actions
        if "--api-key" in action.option_strings
    )
    hidden_api_key.required = False
    hidden_api_key.default = None
    return parser


def _safe_trial_command(
    args: argparse.Namespace,
    *,
    case_path: Path,
    loadout_path: Path,
    result_path: Path,
    home: Path,
    workspace: Path,
    state_dir: Path,
) -> List[str]:
    return [
        args.hermes_python,
        str(Path(__file__).resolve()),
        "_trial",
        "--hermes-repo",
        str(Path(args.hermes_repo).resolve()),
        "--case",
        str(case_path),
        "--loadout",
        str(loadout_path),
        "--endpoint",
        args.endpoint,
        "--api-key-env",
        args.api_key_env,
        "--result",
        str(result_path),
        "--hermes-home",
        str(home),
        "--workspace",
        str(workspace),
        "--state-dir",
        str(state_dir),
    ]


def _resolve_run_key(args: argparse.Namespace) -> None:
    environment_name = str(args.api_key_env or "")
    if not environment_name:
        raise ValueError("--api-key-env must not be empty")
    environment_value = os.getenv(environment_name)
    explicit = str(args.api_key or "")
    if environment_value:
        args.api_key = environment_value
        return
    if explicit not in {"", "local-benchmark"}:
        raise ValueError(
            "nondefault Hermes API keys must be supplied through --api-key-env"
        )
    args.api_key = "local-benchmark"


def _resolve_trial_key(args: argparse.Namespace) -> None:
    environment_name = str(args.api_key_env or "")
    if not environment_name:
        raise ValueError("hidden Hermes trial requires an API-key environment name")
    args.api_key = os.getenv(environment_name, "local-benchmark")


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _base.trial_command = _safe_trial_command
    _PATCHED = True


def build_parser() -> argparse.ArgumentParser:
    return _parser()


def main(argv: Optional[List[str]] = None) -> int:
    apply_patches()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            _resolve_run_key(args)
            return _base.run_benchmark(args)
        if args.command == "gate":
            return _base.gate_report(args)
        _resolve_trial_key(args)
        return _base.trial_process(args)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Hermes benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
