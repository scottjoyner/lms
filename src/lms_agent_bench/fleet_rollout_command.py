"""Production command surface for guarded physical fleet rollout.

The compatibility implementation in :mod:`fleet_rollout_entrypoint` owns
configuration expansion, validation, collection, and failure-safe packaging.
This module makes the generated remote script invoke the hardened planner and
benchmark entrypoints without requiring an execution-only selection command to
exist during observation-only runs.
"""
from __future__ import annotations

import sys
from typing import Any, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_rollout as _base
from lms_agent_bench import fleet_rollout_entrypoint as _support

_ORIGINAL_BUILD_REMOTE_SCRIPT = _base.build_remote_script


def route_hardened_entrypoints(script: str) -> str:
    required = {
        "lms_agent_bench.fleet_loadout discover": (
            "lms_agent_bench.fleet_loadout_entrypoint discover"
        ),
        "lms_agent_bench.fleet_loadout plan": (
            "lms_agent_bench.fleet_loadout_entrypoint plan"
        ),
        "lms_agent_bench.fleet_bench_plan": (
            "lms_agent_bench.fleet_bench_entrypoint"
        ),
    }
    optional = {
        "lms_agent_bench.fleet_loadout select": (
            "lms_agent_bench.fleet_loadout_entrypoint select"
        )
    }
    for original, hardened in required.items():
        if original not in script:
            raise RuntimeError(
                f"rollout script no longer contains expected command: {original}"
            )
        script = script.replace(original, hardened)
    for original, hardened in optional.items():
        if original in script:
            script = script.replace(original, hardened)
    return script


def build_remote_script(
    node: Mapping[str, Any],
    run_id: str,
    execute_candidates: Sequence[str] = (),
    update_code: bool = False,
    dry_run_limit: int = 4,
) -> str:
    script = _ORIGINAL_BUILD_REMOTE_SCRIPT(
        node,
        run_id,
        execute_candidates=execute_candidates,
        update_code=update_code,
        dry_run_limit=dry_run_limit,
    )
    script = route_hardened_entrypoints(script)
    script = _support._remove_legacy_packaging(script)
    marker = 'mkdir -p "$ARTIFACT_DIR"\n'
    if marker not in script:
        raise RuntimeError(
            "rollout script no longer exposes the artifact-directory marker"
        )
    return script.replace(
        marker, marker + _support._exit_packaging_snippet(), 1
    )


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    if actual_argv and actual_argv[0] == "validate":
        return _support.validate_command(actual_argv[1:])

    cleaned_argv, env_file = _support._extract_env_file(actual_argv)
    _base.build_remote_script = build_remote_script
    _base.execute_remote = _support.execute_remote
    _base.load_json = lambda path: _support.load_rollout_config(path, env_file)
    return _base.main(cleaned_argv)


if __name__ == "__main__":
    raise SystemExit(main())
