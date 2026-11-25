"""Installed entrypoint for safe, reliability-gated fleet plan execution.

Dry runs remain render-only. Real execution requires loopback-local endpoints,
exact model identity, a complete reliability report, and bounded benchmark
orchestration before a candidate may reach loadout selection.
"""
from __future__ import annotations

import csv
import ipaddress
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from lms_agent_bench import fleet_bench_plan as _base

_ORIGINAL_EXECUTE_CANDIDATE = _base.execute_candidate
_ORIGINAL_PARSE_ENDPOINT_MAP = _base.parse_endpoint_map
_EXPECTED_MODEL_ID: Optional[str] = None

_RELIABILITY_FIELDS = [
    "reliability_pass",
    "reliability_score",
    "reliability_fingerprint",
    "valid_trials",
    "required_trials",
    "trial_attempts",
    "trial_retry_rate",
    "sample_completeness",
    "success_wilson_lower_95",
    "trial_tps_cv",
    "trial_ttft_cv",
    "tps_relative_mad",
    "ttft_relative_mad",
    "tps_p10",
    "tps_p90",
    "ttft_p90",
    "tps_median_ci95_low",
    "tps_median_ci95_high",
    "ttft_median_ci95_low",
    "ttft_median_ci95_high",
    "warmup_cv",
    "warmup_stable",
    "reliability_failures",
]


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


def _write_identity_failure(
    candidate_dir: Path,
    candidate_id: str,
    expected_model: str,
    exposed_model: str,
) -> None:
    output_dir = candidate_dir / "suite"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "reliable_benchmark.v1",
        "artifact_type": "benchmark_reliability",
        "passed": False,
        "failure": "model_identity_mismatch",
        "candidate_id": candidate_id,
        "expected_model": expected_model,
        "exposed_model": exposed_model,
        "admission": {"admitted": False},
    }
    (output_dir / "reliability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_bounded_suite(command: Sequence[str], log_path: Path) -> int:
    timeout_s = float(os.environ.get("LMS_BENCH_SUITE_TIMEOUT", "6900"))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return int(process.wait(timeout=timeout_s))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.wait(timeout=5)
            log.write(f"\nreliable benchmark suite timed out after {timeout_s}s\n")
            return 124


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
    expected_model = _EXPECTED_MODEL_ID or model
    if model != expected_model:
        _write_identity_failure(
            candidate_dir,
            candidate_id,
            expected_model,
            model,
        )
        (candidate_dir / "suite.log").write_text(
            (
                "benchmark rejected before measurement: endpoint model identity "
                f"{model!r} does not equal planned model {expected_model!r}\n"
            ),
            encoding="utf-8",
        )
        return 1

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
    return _run_bounded_suite(command, candidate_dir / "suite.log")


def _dry_run_base(
    candidate: Mapping[str, Any], candidate_dir: Path
) -> Dict[str, Any]:
    result = {
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
    result.update({field: "" for field in _RELIABILITY_FIELDS})
    return result


def _summary_row(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(iter(csv.DictReader(handle)), {})


def _reliability_details(candidate_dir: Path) -> Dict[str, Any]:
    report_path = candidate_dir / "suite" / "reliability.json"
    summary = _summary_row(candidate_dir / "suite" / "run_summary.csv")
    details: Dict[str, Any] = {
        field: summary.get(field, "") for field in _RELIABILITY_FIELDS
    }
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = {}
        report_summaries = report.get("summaries")
        if isinstance(report_summaries, list) and report_summaries:
            first_summary = report_summaries[0]
            if isinstance(first_summary, Mapping):
                for field in _RELIABILITY_FIELDS:
                    if field in first_summary:
                        details[field] = first_summary[field]
        details["reliability_fingerprint"] = report.get(
            "reliability_fingerprint", ""
        )
        details["reliability_pass"] = bool(report.get("passed"))
        details["valid_trials"] = report.get(
            "valid_trials", details.get("valid_trials", "")
        )
        details["trial_attempts"] = report.get(
            "trial_attempts", details.get("trial_attempts", "")
        )
        if not details.get("reliability_failures") and report.get("failure"):
            details["reliability_failures"] = [report["failure"]]
    else:
        details["reliability_pass"] = False
        details["reliability_failures"] = ["reliability_report_missing"]
    return details


def execute_candidate(
    candidate: Mapping[str, Any],
    args: Any,
    endpoint_map: Mapping[str, str],
    help_cache: Dict[str, str],
) -> Dict[str, Any]:
    """Render dry runs safely and reliability-gate real execution."""
    if args.dry_run:
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

    global _EXPECTED_MODEL_ID
    _EXPECTED_MODEL_ID = str(candidate.get("model", {}).get("id") or "")
    try:
        result = _ORIGINAL_EXECUTE_CANDIDATE(
            candidate, args, endpoint_map, help_cache
        )
    finally:
        _EXPECTED_MODEL_ID = None

    candidate_dir = Path(
        str(
            result.get("candidate_dir")
            or Path(args.output_dir) / _base.safe_slug(str(candidate["candidate_id"]))
        )
    )
    reliability = _reliability_details(candidate_dir)
    result.update(reliability)
    planned_model = str(candidate.get("model", {}).get("id") or "")
    measured_model = str(result.get("model_id") or "")
    raw_failures = reliability.get("reliability_failures") or []
    failures = (
        [str(item) for item in raw_failures]
        if isinstance(raw_failures, list)
        else [str(raw_failures)]
    )
    if measured_model != planned_model:
        failures.append("measured_model_does_not_match_planned_model")
    if reliability.get("reliability_pass") is not True:
        failures.append("benchmark_reliability_gate_failed")
    if failures:
        result["benchmark_exit_code"] = 1
        result["reliability_pass"] = False
        result["reliability_failures"] = sorted(set(str(item) for item in failures))
        existing = str(result.get("error") or "")
        result["error"] = "; ".join(
            item
            for item in (
                existing,
                ", ".join(result["reliability_failures"]),
            )
            if item
        )
    _base.write_json(str(candidate_dir / "result.json"), result)
    return result


def write_results_csv(
    path: Path, results: Sequence[Mapping[str, Any]]
) -> None:
    fields = [
        "candidate_id",
        "engine",
        "backend",
        "model_id",
        "base_url",
        "ok_rate",
        "eval_ok_rate",
        "eval_score_avg",
        "tps_med",
        "ttft_med",
        "memory_peak_bytes",
        "memory_headroom_ratio",
        "concurrency_ok",
        "streaming_ok",
        "cancellation_ok",
        "crash_count",
        "benchmark_exit_code",
        *_RELIABILITY_FIELDS,
        "error",
        "candidate_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def main(argv: Optional[List[str]] = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    _base.run_lms_suite = run_lms_suite
    _base.execute_candidate = execute_candidate
    _base.parse_endpoint_map = parse_endpoint_map
    _base.write_results_csv = write_results_csv
    return _base.main(inject_default_suite(actual_argv))


if __name__ == "__main__":
    raise SystemExit(main())
