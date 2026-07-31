"""Installed entrypoint for safe fleet plan execution."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from lms_agent_bench import fleet_bench_plan as _base


def default_suite_file() -> str:
    path = (
        Path(__file__).resolve().parent
        / "benchmarks"
        / "agent_skill_suite.v1.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"packaged benchmark suite is missing: {path}")
    return str(path)


def inject_default_suite(argv: List[str]) -> List[str]:
    if "--suite-file" in argv:
        return list(argv)
    return [*argv, "--suite-file", default_suite_file()]


def run_lms_suite(
    candidate_dir: Path,
    base_url: str,
    model: str,
    candidate_id: str,
    suite_file: str,
    timeout_s: float,
    repeats: int,
    max_context_tokens: int,
) -> int:
    inventory = candidate_dir / "inventory.csv"
    _base.write_inventory(inventory, base_url, model, candidate_id)
    output_dir = candidate_dir / "suite"
    sidecar_dir = candidate_dir / "sidecars"
    command = [
        sys.executable,
        "-m",
        "lms_agent_bench.benchmark_entrypoint",
        "--inventory-csv",
        str(inventory),
        "--cases-file",
        suite_file,
        "--output-dir",
        str(output_dir),
        "--sidecar-dir",
        str(sidecar_dir),
        "--timeout",
        str(timeout_s),
        "--repeats",
        str(repeats),
        "--max-context-tokens",
        str(max_context_tokens),
    ]
    (candidate_dir / "suite_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with (candidate_dir / "suite.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    return int(proc.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    _base.run_lms_suite = run_lms_suite
    return _base.main(inject_default_suite(actual_argv))


if __name__ == "__main__":
    raise SystemExit(main())
