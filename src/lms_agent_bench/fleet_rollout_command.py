"""Command wrapper that makes evidence collection part of rollout success."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_rollout_entrypoint as _entrypoint


def _option_value(argv: Sequence[str], name: str) -> Optional[str]:
    values = list(argv)
    for index, value in enumerate(values):
        if value == name and index + 1 < len(values):
            return values[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _collection_failed(result: Mapping[str, Any]) -> bool:
    try:
        remote_returncode = int(result.get("returncode"))
    except (TypeError, ValueError):
        remote_returncode = -1
    if remote_returncode != 0 or result.get("timed_out") is True:
        return True
    try:
        scp_returncode = int(result.get("scp_returncode"))
    except (TypeError, ValueError):
        scp_returncode = -1
    return (
        scp_returncode != 0
        or result.get("scp_timed_out") is True
        or not result.get("collected_artifact")
        or not result.get("collected_artifact_sha256")
    )


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    returncode = _entrypoint.main(actual_argv)
    if not actual_argv or actual_argv[0] != "run" or "--no-collect" in actual_argv:
        return returncode

    output_dir = _option_value(actual_argv, "--output-dir")
    if not output_dir:
        return returncode
    results_path = Path(output_dir) / "rollout_results.json"
    if not results_path.is_file():
        return 1
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(results, list) or not results:
        return 1
    return 1 if any(
        not isinstance(item, Mapping) or _collection_failed(item)
        for item in results
    ) else returncode


if __name__ == "__main__":
    raise SystemExit(main())
