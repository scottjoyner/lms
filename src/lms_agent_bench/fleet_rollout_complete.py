"""Public rollout wrapper with census-backed full-fleet coverage enforcement."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from lms_agent_bench import fleet_rollout_command as _command
from lms_agent_bench import fleet_rollout_entrypoint as _entrypoint
from lms_agent_bench.fleet_coverage import validate_rollout_coverage


def _option_value(argv: Sequence[str], name: str) -> Optional[str]:
    values = list(argv)
    for index, value in enumerate(values):
        if value == name and index + 1 < len(values):
            return values[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _write_validation_coverage(
    output_path: str, coverage: Dict[str, Any], base_returncode: int
) -> int:
    path = Path(output_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {
            "schema_version": "fleet_rollout_validation.v1",
            "ready_for_observation": False,
            "admission": {"admitted": False},
        }
    payload["coverage"] = coverage
    payload["ready_for_observation"] = bool(
        payload.get("ready_for_observation") and coverage.get("ready")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not coverage.get("ready"):
        return 1
    return base_returncode


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _option_value(actual_argv, "--config")
    env_file = _option_value(actual_argv, "--env-file")
    command_name = actual_argv[0] if actual_argv else ""
    captured: Dict[str, Any] = {}
    active_loader = _entrypoint.load_rollout_config

    def covered_load(path: str, selected_env_file: Optional[str]):
        config = active_loader(path, selected_env_file)
        coverage = validate_rollout_coverage(config, path)
        captured["coverage"] = coverage
        if command_name != "validate" and not coverage.get("ready"):
            raise ValueError(
                "; ".join(
                    coverage.get("errors")
                    or ["fleet benchmark coverage is incomplete"]
                )
            )
        return config

    _entrypoint.load_rollout_config = covered_load
    try:
        returncode = _command.main(actual_argv)
    finally:
        _entrypoint.load_rollout_config = active_loader

    if command_name == "validate":
        output_path = _option_value(actual_argv, "--out")
        coverage = captured.get("coverage")
        if output_path and isinstance(coverage, dict):
            return _write_validation_coverage(output_path, coverage, returncode)
        if config_path:
            try:
                config = active_loader(config_path, env_file)
                coverage = validate_rollout_coverage(config, config_path)
            except (OSError, ValueError, json.JSONDecodeError):
                return 1
            if output_path:
                return _write_validation_coverage(output_path, coverage, returncode)
            if not coverage.get("ready"):
                return 1
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
