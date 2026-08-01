"""Installed entrypoint for safe fleet plan execution.

The wrapper owns dry-run semantics and enforces that mapped physical benchmark
endpoints are loopback-local, matching the execution manifest's isolation claim.
"""
from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from lms_agent_bench import fleet_bench_plan as _base

_ORIGINAL_EXECUTE_CANDIDATE = _base.execute_candidate
_ORIGINAL_PARSE_ENDPOINT_MAP = _base.parse_endpoint_map


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


def is_loopback_url(value: str) -> bool:
    parsed = urlsplit(value)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def parse_endpoint_map(values: Sequence[str]) -> Dict[str, str]:
    parsed = _ORIGINAL_PARSE_ENDPOINT_MAP(values)
    remote = {
        candidate_id: url
        for candidate_id, url in parsed.items()
        if not is_loopback_url(url)
    }
    if remote:
        rendered = ", ".join(
            f"{candidate_id}={url}" for candidate_id, url in sorted(remote.items())
        )
        raise ValueError(
            "physical plan execution accepts loopback endpoint mappings only; "
            f"rejected: {rendered}"
        )
    return parsed


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


def _dry_run_base(
    candidate: Mapping[str, Any], candidate_dir: Path
) -> Dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "engine": candidate.get("engine"),
        "backend": candidate.get("backend"),
        "model_id": candidate.get("model", {}).get("id", ""),
        "base_url": "",
        "ok_rate": "",
        "eval_ok_rate": "",
        "eval_score_avg": "",
        "tps_med": "",
        "ttft_med": "",
        "memory_peak_bytes": "",
        "memory_headroom_ratio": "",
        "concurrency_ok": "",
        "streaming_ok": "",
        "cancellation_ok": "",
        "crash_count": 0,
        "benchmark_exit_code": 0,
        "error": "",
        "candidate_dir": str(candidate_dir),
        "dry_run": True,
    }


def execute_candidate(
    candidate: Mapping[str, Any],
    args: Any,
    endpoint_map: Mapping[str, str],
    help_cache: Dict[str, str],
) -> Dict[str, Any]:
    """Render dry runs safely; delegate real execution to the base engine."""
    if not args.dry_run:
        return _ORIGINAL_EXECUTE_CANDIDATE(
            candidate, args, endpoint_map, help_cache
        )

    candidate_id = str(candidate["candidate_id"])
    candidate_dir = Path(args.output_dir) / _base.safe_slug(candidate_id)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result = _dry_run_base(candidate, candidate_dir)
    try:
        mapped_url = endpoint_map.get(candidate_id)
        if mapped_url:
            if not is_loopback_url(mapped_url):
                raise ValueError(
                    f"mapped endpoint is not loopback-local: {mapped_url}"
                )
            result.update(
                {
                    "base_url": _base.normalize_base_url(mapped_url),
                    "launch_mode": "mapped_existing_endpoint",
                    "requires_endpoint_map": False,
                }
            )
        elif candidate.get("engine") == "llama.cpp":
            configured_binary = (
                args.llama_server_bin
                or os.environ.get("LLAMA_SERVER_BIN")
                or shutil.which("llama-server")
                or "llama-server"
            )
            available_binary = (
                str(configured_binary)
                if Path(str(configured_binary)).exists()
                else shutil.which(str(configured_binary))
            )
            help_text = (
                help_cache.setdefault(
                    str(configured_binary),
                    _base.command_supported(str(configured_binary)),
                )
                if available_binary
                else ""
            )
            launch_command = _base.build_llama_server_command(
                candidate,
                str(configured_binary),
                help_text=help_text,
            )
            launch = {
                "candidate_id": candidate_id,
                "command": launch_command,
                "binary_available": bool(available_binary),
                "environment_overrides": candidate.get("environment", {}),
                "created_at_utc": _base.utc_now_iso(),
                "dry_run": True,
            }
            _base.write_json(str(candidate_dir / "launch.json"), launch)
            result.update(
                {
                    "launch_mode": "ephemeral_loopback_only",
                    "launch_command": launch_command,
                    "binary_available": bool(available_binary),
                    "requires_endpoint_map": False,
                }
            )
        else:
            result.update(
                {
                    "launch_mode": "existing_or_adapter",
                    "requires_endpoint_map": True,
                    "render_note": (
                        "real execution requires a loopback --endpoint-map "
                        f"{candidate_id}=URL"
                    ),
                }
            )
    except Exception as exc:
        result.update(
            {
                "benchmark_exit_code": 1,
                "error": repr(exc),
            }
        )
    _base.write_json(str(candidate_dir / "result.json"), result)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    _base.run_lms_suite = run_lms_suite
    _base.execute_candidate = execute_candidate
    _base.parse_endpoint_map = parse_endpoint_map
    return _base.main(inject_default_suite(actual_argv))


if __name__ == "__main__":
    raise SystemExit(main())
