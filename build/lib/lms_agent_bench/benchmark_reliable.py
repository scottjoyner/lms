#!/usr/bin/env python3
"""Reliability-first orchestration for local OpenAI-compatible benchmarks.

The legacy benchmark runner remains the single-trial measurement engine. This
module adds deterministic multi-trial execution, strict model identity,
whole-trial retries, resumability, artifact completeness checks, robust
statistics, confidence intervals, and explicit reliability gates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

from lms_agent_bench import __version__
from lms_agent_bench import lms_eval as _lms_eval

sys.modules.setdefault("lms_eval", _lms_eval)
from lms_agent_bench import benchmark_lmstudio_cross_machine_models as _legacy
from lms_agent_bench import benchmark_protocol as _protocol

SCHEMA_VERSION = "reliable_benchmark.v1"
DEFAULT_SEED = 20260802
_TRANSIENT_TOKENS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "connection aborted",
    "temporarily unavailable",
    "remote disconnected",
    "http 408",
    "http 425",
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
)


def utc_now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def float_value(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: Sequence[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def percentile(values: Sequence[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    avg = statistics.mean(values)
    if avg == 0:
        return None
    return statistics.stdev(values) / abs(avg)


def relative_mad(values: Sequence[float]) -> Optional[float]:
    center = median(values)
    if center is None or center == 0:
        return None
    deviation = statistics.median(abs(value - center) for value in values)
    return deviation / abs(center)


def bootstrap_median_ci(
    values: Sequence[float], seed: int, samples: int
) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.median(draw))
    return percentile(estimates, 2.5), percentile(estimates, 97.5)


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return max(0.0, (center - margin) / denominator)


def format_float(value: Optional[float], digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def environment_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "captured_at_utc": utc_now_iso(),
        "platform": platform.platform(),
        "load_average": None,
        "memory": {},
        "thermal_celsius": [],
    }
    try:
        snapshot["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    if platform.system() == "Linux":
        values: Dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                token = rest.strip().split()[0]
                values[key] = int(token) * 1024
        except (OSError, ValueError, IndexError):
            values = {}
        snapshot["memory"] = {
            "total_bytes": values.get("MemTotal"),
            "available_bytes": values.get("MemAvailable"),
            "swap_total_bytes": values.get("SwapTotal"),
            "swap_free_bytes": values.get("SwapFree"),
        }
        temperatures: List[float] = []
        for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
            try:
                raw = float(path.read_text(encoding="utf-8").strip())
                temperatures.append(raw / 1000.0 if raw > 1000 else raw)
            except (OSError, ValueError):
                continue
        snapshot["thermal_celsius"] = temperatures
    return snapshot


def model_ids(base_url: str, timeout_s: float, api_key: Optional[str]) -> List[str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.get(
        base_url.rstrip("/") + "/models",
        headers=headers,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    return [
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, Mapping) and item.get("id")
    ]


def canary(
    base_url: str,
    model: str,
    timeout_s: float,
    api_key: Optional[str],
) -> Dict[str, Any]:
    metrics = _protocol.call_chat_completions_once(
        base_url,
        model,
        [
            {"role": "system", "content": "You are a deterministic health probe."},
            {"role": "user", "content": "Reply with READY and nothing else."},
        ],
        8,
        0.0,
        timeout_s,
        api_key,
    )
    output = metrics.output_text.strip()
    return {
        "ok": bool(metrics.ok and output == "READY"),
        "transport_ok": bool(metrics.ok),
        "output": output[:100],
        "http_status": metrics.http_status,
        "error": metrics.error,
        "wall_s": metrics.wall_s,
    }


def endpoint_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        str(row.get(field, ""))
        for field in ("endpoint_id", "base_url", "model_key")
    )


def preflight_endpoint(
    row: Mapping[str, Any], args: argparse.Namespace, api_key: Optional[str]
) -> Dict[str, Any]:
    base_url = str(row.get("base_url") or "").rstrip("/")
    expected_model = str(row.get("model_key") or "")
    report: Dict[str, Any] = {
        "endpoint_key": endpoint_key(row),
        "base_url": base_url,
        "expected_model": expected_model,
        "started_at_utc": utc_now_iso(),
        "environment_before": environment_snapshot(),
        "models": [],
        "cold_load": None,
        "warmups": [],
        "warmup_cv": None,
        "stable": False,
        "ok": False,
        "errors": [],
    }
    try:
        exposed = model_ids(base_url, args.preflight_timeout, api_key)
        report["models"] = exposed
        if expected_model not in exposed:
            report["errors"].append(
                f"expected model {expected_model!r} is not exposed exactly"
            )
            return report
        cold = canary(base_url, expected_model, args.preflight_timeout, api_key)
        report["cold_load"] = cold
        if not cold["ok"]:
            report["errors"].append("cold-load canary failed")
            return report
        for _ in range(args.warmup_runs):
            probe = canary(base_url, expected_model, args.preflight_timeout, api_key)
            report["warmups"].append(probe)
            if not probe["ok"]:
                report["errors"].append("warmup canary failed")
                return report
        walls = [float(item["wall_s"]) for item in report["warmups"]]
        cv = coefficient_of_variation(walls)
        report["warmup_cv"] = cv
        report["stable"] = cv is None or cv <= args.max_warmup_cv
        if not report["stable"]:
            report["errors"].append(
                f"warmup latency CV {cv:.4f} exceeds {args.max_warmup_cv:.4f}"
            )
            return report
        report["ok"] = True
        return report
    except Exception as exc:
        report["errors"].append(repr(exc))
        return report
    finally:
        report["finished_at_utc"] = utc_now_iso()
        report["environment_after"] = environment_snapshot()


def transient_text(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in _TRANSIENT_TOKENS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reliability-first multi-trial benchmark runner"
    )
    parser.add_argument("--inventory-csv", required=True)
    parser.add_argument("--cases-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--api-key-env", default="LMSTUDIO_API_KEY")
    parser.add_argument("--only-reachable", action="store_true", default=True)
    parser.add_argument("--include-endpoints", default=None)
    parser.add_argument("--exclude-endpoints", default=None)
    parser.add_argument("--include-models", default=None)
    parser.add_argument("--exclude-models", default=None)
    parser.add_argument("--max-context-tokens", type=int, default=8192)

    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--min-valid-trials", type=int, default=3)
    parser.add_argument("--max-trial-attempts", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--preflight-timeout", type=float, default=120)
    parser.add_argument("--trial-timeout", type=float, default=0)
    parser.add_argument("--retry-backoff", type=float, default=2)
    parser.add_argument("--cooldown-between-trials", type=float, default=2)
    parser.add_argument("--max-warmup-cv", type=float, default=0.50)
    parser.add_argument("--min-sample-completeness", type=float, default=1.0)
    parser.add_argument("--min-success-rate", type=float, default=0.98)
    parser.add_argument("--min-eval-success-rate", type=float, default=0.90)
    parser.add_argument("--min-wilson-lower", type=float, default=0.80)
    parser.add_argument("--max-trial-tps-cv", type=float, default=0.20)
    parser.add_argument("--max-trial-ttft-cv", type=float, default=0.35)
    parser.add_argument("--max-relative-mad", type=float, default=0.25)
    parser.add_argument("--max-retry-rate", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Write all reports but do not fail the process on reliability gates",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "repeats",
        "trials",
        "min_valid_trials",
        "max_trial_attempts",
        "warmup_runs",
        "bootstrap_samples",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_valid_trials > args.trials:
        raise ValueError("--min-valid-trials cannot exceed --trials")
    if args.max_trial_attempts < args.trials:
        raise ValueError("--max-trial-attempts cannot be less than --trials")
    for name in (
        "min_sample_completeness",
        "min_success_rate",
        "min_eval_success_rate",
        "min_wilson_lower",
        "max_retry_rate",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")


def load_filtered_inventory(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = _legacy.load_inventory_rows(Path(args.inventory_csv))
    filtered = _legacy.filter_inventory_rows(
        rows,
        include_endpoints=_legacy.parse_csv_set(args.include_endpoints),
        exclude_endpoints=_legacy.parse_csv_set(args.exclude_endpoints),
        include_models=_legacy.parse_csv_set(args.include_models),
        exclude_models=_legacy.parse_csv_set(args.exclude_models),
        only_reachable=args.only_reachable,
    )
    seen: set[str] = set()
    for row in filtered:
        key = endpoint_key(row)
        if key in seen:
            raise ValueError(f"duplicate inventory endpoint/model row: {key}")
        seen.add(key)
    if not filtered:
        raise ValueError("no inventory rows matched filters")
    return filtered


def input_fingerprint(
    args: argparse.Namespace,
    inventory_path: Path,
    suite_path: Optional[Path],
    case_keys: Sequence[str],
) -> Tuple[str, Dict[str, Any]]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "runner_version": __version__,
        "inventory_sha256": file_sha256(inventory_path),
        "suite_sha256": file_sha256(suite_path) if suite_path else None,
        "case_keys": list(case_keys),
        "timeout": args.timeout,
        "repeats": args.repeats,
        "max_context_tokens": args.max_context_tokens,
        "trials": args.trials,
        "min_valid_trials": args.min_valid_trials,
        "max_trial_attempts": args.max_trial_attempts,
        "warmup_runs": args.warmup_runs,
        "preflight_timeout": args.preflight_timeout,
        "trial_timeout": args.trial_timeout,
        "retry_backoff": args.retry_backoff,
        "cooldown_between_trials": args.cooldown_between_trials,
        "seed": args.seed,
        "thresholds": {
            "max_warmup_cv": args.max_warmup_cv,
            "min_sample_completeness": args.min_sample_completeness,
            "min_success_rate": args.min_success_rate,
            "min_eval_success_rate": args.min_eval_success_rate,
            "min_wilson_lower": args.min_wilson_lower,
            "max_trial_tps_cv": args.max_trial_tps_cv,
            "max_trial_ttft_cv": args.max_trial_ttft_cv,
            "max_relative_mad": args.max_relative_mad,
            "max_retry_rate": args.max_retry_rate,
        },
    }
    return canonical_hash(core), core


def write_trial_inventory(path: Path, rows: Sequence[Mapping[str, Any]], seed: int) -> None:
    ordered = [dict(row) for row in rows]
    random.Random(seed).shuffle(ordered)
    fields = [
        "host_name",
        "host_ip",
        "endpoint_id",
        "base_url",
        "reachable",
        "model_id",
        "model_key",
    ]
    write_csv(path, ordered, fields)


def write_trial_suite(
    source: Optional[Path], destination: Path, seed: int
) -> Optional[Path]:
    if source is None:
        return None
    suite = read_json(source)
    cases = list(suite.get("cases", []))
    random.Random(seed).shuffle(cases)
    suite["cases"] = cases
    write_json(destination, suite)
    return destination


def raw_command(
    args: argparse.Namespace,
    inventory: Path,
    suite: Optional[Path],
    output_dir: Path,
    sidecar_dir: Path,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "lms_agent_bench.benchmark_raw_entrypoint",
        "--inventory-csv",
        str(inventory),
        "--output-dir",
        str(output_dir),
        "--sidecar-dir",
        str(sidecar_dir),
        "--timeout",
        str(args.timeout),
        "--repeats",
        str(args.repeats),
        "--max-context-tokens",
        str(args.max_context_tokens),
        "--api-key-env",
        args.api_key_env,
    ]
    if suite is not None:
        command.extend(["--cases-file", str(suite)])
    for flag, value in (
        ("--include-endpoints", args.include_endpoints),
        ("--exclude-endpoints", args.exclude_endpoints),
        ("--include-models", args.include_models),
        ("--exclude-models", args.exclude_models),
    ):
        if value:
            command.extend([flag, value])
    return command


def auto_trial_timeout(
    args: argparse.Namespace, inventory_count: int, case_count: int
) -> float:
    if args.trial_timeout > 0:
        return args.trial_timeout
    request_budget = args.timeout * inventory_count * (1 + case_count * args.repeats)
    return max(300.0, request_budget + 120.0)


def expected_sample_keys(
    inventory_rows: Sequence[Mapping[str, Any]],
    case_keys: Sequence[str],
    repeats: int,
) -> set[Tuple[str, str, str, int]]:
    return {
        (
            str(row.get("endpoint_id", "")),
            str(row.get("model_key", "")),
            case_key,
            repeat,
        )
        for row in inventory_rows
        for case_key in case_keys
        for repeat in range(1, repeats + 1)
    }


def validate_trial_artifacts(
    output_dir: Path,
    expected_keys: set[Tuple[str, str, str, int]],
) -> Tuple[bool, List[str], List[Dict[str, str]]]:
    errors: List[str] = []
    required = [
        output_dir / "config.json",
        output_dir / "run_results.csv",
        output_dir / "run_summary.csv",
        output_dir / "task_summary.csv",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing artifact: {path.name}")
    if errors:
        return False, errors, []
    try:
        rows = read_csv(output_dir / "run_results.csv")
    except (OSError, csv.Error) as exc:
        return False, [f"invalid run_results.csv: {exc}"], []
    run_rows = [row for row in rows if row.get("phase") == "run"]
    observed: List[Tuple[str, str, str, int]] = []
    for row in run_rows:
        try:
            repeat = int(row.get("repeat_index") or 0)
        except ValueError:
            repeat = 0
        observed.append(
            (
                str(row.get("endpoint_id", "")),
                str(row.get("model_key", "")),
                str(row.get("case_key", "")),
                repeat,
            )
        )
    observed_set = set(observed)
    if len(observed) != len(observed_set):
        errors.append("duplicate run sample keys")
    missing = sorted(expected_keys - observed_set)
    unexpected = sorted(observed_set - expected_keys)
    if missing:
        errors.append(f"missing {len(missing)} expected run samples")
    if unexpected:
        errors.append(f"found {len(unexpected)} unexpected run samples")
    for row in run_rows:
        if bool_value(row.get("ok")):
            if not str(row.get("output_file") or ""):
                errors.append("successful run sample has no output artifact")
                break
            if float_value(row.get("tokens_per_sec")) is None:
                errors.append("successful run sample has no throughput measurement")
                break
    return not errors, errors, rows


def postflight_endpoints(
    inventory_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    api_key: Optional[str],
) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    for row in inventory_rows:
        key = endpoint_key(row)
        base_url = str(row.get("base_url") or "").rstrip("/")
        model = str(row.get("model_key") or "")
        report: Dict[str, Any] = {"ok": False, "errors": []}
        try:
            exposed = model_ids(base_url, args.preflight_timeout, api_key)
            report["models"] = exposed
            if model not in exposed:
                report["errors"].append("model identity changed after trial")
            health = canary(base_url, model, args.preflight_timeout, api_key)
            report["canary"] = health
            if not health["ok"]:
                report["errors"].append("post-trial canary failed")
            report["ok"] = not report["errors"]
        except Exception as exc:
            report["errors"].append(repr(exc))
        reports[key] = report
    return reports


def existing_valid_trial(
    trial_root: Path, input_fp: str
) -> Optional[Dict[str, Any]]:
    for manifest_path in sorted(trial_root.glob("attempt_*/trial_manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("input_fingerprint") == input_fp
            and manifest.get("valid") is True
        ):
            output_dir = manifest_path.parent / "output"
            rows = read_csv(output_dir / "run_results.csv")
            return {
                "trial_index": int(manifest.get("trial_index")),
                "attempt_index": int(manifest.get("attempt_index")),
                "manifest": manifest,
                "rows": rows,
                "output_dir": str(output_dir),
                "resumed": True,
            }
    return None


def artifact_hashes(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    artifacts: Dict[str, Dict[str, Any]] = {}
    for name in ("config.json", "run_results.csv", "run_summary.csv", "task_summary.csv"):
        path = output_dir / name
        if path.is_file():
            artifacts[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return artifacts


def execute_trial_attempt(
    args: argparse.Namespace,
    trial_index: int,
    attempt_index: int,
    root: Path,
    inventory_rows: Sequence[Mapping[str, Any]],
    suite_source: Optional[Path],
    expected_keys: set[Tuple[str, str, str, int]],
    input_fp: str,
    api_key: Optional[str],
    timeout_s: float,
) -> Dict[str, Any]:
    attempt_dir = root / f"trial_{trial_index:03d}" / f"attempt_{attempt_index:03d}"
    output_dir = attempt_dir / "output"
    sidecar_dir = attempt_dir / "sidecars"
    ensure_dir(output_dir)
    ensure_dir(sidecar_dir)
    seed = args.seed + trial_index * 1009 + attempt_index
    trial_inventory = attempt_dir / "inventory.csv"
    trial_suite = write_trial_suite(
        suite_source, attempt_dir / "suite.json", seed
    )
    write_trial_inventory(trial_inventory, inventory_rows, seed)
    command = raw_command(args, trial_inventory, trial_suite, output_dir, sidecar_dir)
    write_json(attempt_dir / "command.json", command)
    manifest: Dict[str, Any] = {
        "schema_version": "reliable_benchmark_trial.v1",
        "trial_index": trial_index,
        "attempt_index": attempt_index,
        "input_fingerprint": input_fp,
        "seed": seed,
        "started_at_utc": utc_now_iso(),
        "environment_before": environment_snapshot(),
        "command": command,
        "timeout_s": timeout_s,
        "returncode": None,
        "timed_out": False,
        "valid": False,
        "errors": [],
    }
    log_path = attempt_dir / "runner.log"
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
                env=dict(os.environ),
            )
        manifest["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        manifest["returncode"] = 124
        manifest["timed_out"] = True
        manifest["errors"].append(f"trial timeout after {timeout_s:.1f}s")
    manifest["wall_s"] = time.monotonic() - started

    valid, artifact_errors, rows = validate_trial_artifacts(output_dir, expected_keys)
    manifest["errors"].extend(artifact_errors)
    if manifest["returncode"] not in {0, None}:
        manifest["errors"].append(
            f"raw benchmark exited {manifest['returncode']}"
        )
    postflight = postflight_endpoints(inventory_rows, args, api_key)
    manifest["postflight"] = postflight
    if any(not item.get("ok") for item in postflight.values()):
        manifest["errors"].append("post-trial endpoint health failed")
    manifest["valid"] = bool(
        valid
        and manifest["returncode"] == 0
        and not manifest["timed_out"]
        and all(item.get("ok") for item in postflight.values())
    )
    manifest["finished_at_utc"] = utc_now_iso()
    manifest["environment_after"] = environment_snapshot()
    manifest["artifacts"] = artifact_hashes(output_dir)
    manifest["failure_class"] = (
        None
        if manifest["valid"]
        else (
            "transient"
            if transient_text(" ".join(str(item) for item in manifest["errors"]))
            else "incomplete_or_deterministic"
        )
    )
    if log_path.is_file():
        manifest["runner_log_sha256"] = file_sha256(log_path)
    write_json(attempt_dir / "trial_manifest.json", manifest)
    return {
        "trial_index": trial_index,
        "attempt_index": attempt_index,
        "manifest": manifest,
        "rows": rows,
        "output_dir": str(output_dir),
        "resumed": False,
    }


def sample_group_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("host_name", "")),
        str(row.get("host_ip", "")),
        str(row.get("base_url", "")),
        str(row.get("model_key", "")),
    )


def aggregate_group(
    rows: Sequence[Mapping[str, Any]],
    valid_trials: int,
    required_trials: int,
    trial_attempts: int,
    expected_samples: int,
    preflight: Mapping[str, Any],
    args: argparse.Namespace,
    seed_offset: int,
) -> Dict[str, Any]:
    run_rows = [row for row in rows if row.get("phase") == "run"]
    load_rows = [row for row in rows if row.get("phase") == "load"]
    successful = [row for row in run_rows if bool_value(row.get("ok"))]
    evaluated = [row for row in run_rows if row.get("eval_ok") not in {None, ""}]
    eval_successes = sum(1 for row in evaluated if bool_value(row.get("eval_ok")))
    tps_values = [
        value
        for value in (float_value(row.get("tokens_per_sec")) for row in successful)
        if value is not None and value >= 0
    ]
    ttft_values = [
        value
        for value in (float_value(row.get("ttft_s")) for row in successful)
        if value is not None and value >= 0
    ]
    load_values = [
        value
        for value in (float_value(row.get("load_s")) for row in load_rows)
        if value is not None and value >= 0
    ]
    eval_scores = [
        value
        for value in (float_value(row.get("eval_score")) for row in evaluated)
        if value is not None
    ]

    by_trial: Dict[int, List[Mapping[str, Any]]] = {}
    for row in successful:
        try:
            trial_index = int(row.get("trial_index") or 0)
        except (TypeError, ValueError):
            trial_index = 0
        by_trial.setdefault(trial_index, []).append(row)
    trial_tps = []
    trial_ttft = []
    for trial_rows in by_trial.values():
        values = [
            value
            for value in (
                float_value(row.get("tokens_per_sec")) for row in trial_rows
            )
            if value is not None
        ]
        if values:
            trial_tps.append(statistics.median(values))
        values = [
            value
            for value in (float_value(row.get("ttft_s")) for row in trial_rows)
            if value is not None
        ]
        if values:
            trial_ttft.append(statistics.median(values))

    observed_samples = len(run_rows)
    success_count = len(successful)
    completeness = observed_samples / expected_samples if expected_samples else 0.0
    success_rate = success_count / observed_samples if observed_samples else 0.0
    eval_rate = eval_successes / len(evaluated) if evaluated else 0.0
    retry_rate = (
        max(0, trial_attempts - valid_trials) / trial_attempts
        if trial_attempts
        else 1.0
    )
    tps_cv = coefficient_of_variation(trial_tps)
    ttft_cv = coefficient_of_variation(trial_ttft)
    tps_mad = relative_mad(tps_values)
    ttft_mad = relative_mad(ttft_values)
    tps_low, tps_high = bootstrap_median_ci(
        tps_values, args.seed + seed_offset, args.bootstrap_samples
    )
    ttft_low, ttft_high = bootstrap_median_ci(
        ttft_values, args.seed + seed_offset + 1, args.bootstrap_samples
    )
    warmup_ok = bool(preflight.get("ok"))
    failures: List[str] = []
    if not warmup_ok:
        failures.append("preflight_or_warmup_failed")
    if valid_trials < required_trials:
        failures.append("insufficient_valid_trials")
    if completeness < args.min_sample_completeness:
        failures.append("sample_completeness_below_threshold")
    if success_rate < args.min_success_rate:
        failures.append("request_success_rate_below_threshold")
    if eval_rate < args.min_eval_success_rate:
        failures.append("evaluation_success_rate_below_threshold")
    wilson = wilson_lower_bound(success_count, observed_samples)
    if wilson < args.min_wilson_lower:
        failures.append("success_confidence_lower_bound_below_threshold")
    if tps_cv is None or tps_cv > args.max_trial_tps_cv:
        failures.append("throughput_trial_variation_above_threshold")
    if ttft_cv is None or ttft_cv > args.max_trial_ttft_cv:
        failures.append("ttft_trial_variation_above_threshold")
    if tps_mad is None or tps_mad > args.max_relative_mad:
        failures.append("throughput_relative_mad_above_threshold")
    if ttft_mad is None or ttft_mad > args.max_relative_mad:
        failures.append("ttft_relative_mad_above_threshold")
    if retry_rate > args.max_retry_rate:
        failures.append("trial_retry_rate_above_threshold")

    stability_components = [
        min(1.0, success_rate / max(args.min_success_rate, 1e-9)),
        min(1.0, eval_rate / max(args.min_eval_success_rate, 1e-9)),
        min(1.0, completeness / max(args.min_sample_completeness, 1e-9)),
        max(0.0, 1.0 - (tps_cv or 1.0)),
        max(0.0, 1.0 - (ttft_cv or 1.0)),
        max(0.0, 1.0 - retry_rate),
    ]
    reliability_score = statistics.mean(stability_components)
    return {
        "load_s": format_float(mean(load_values), 3),
        "ttft_med": format_float(median(ttft_values), 3),
        "tps_med": format_float(median(tps_values), 3),
        "ok_rate": format_float(success_rate, 6),
        "eval_ok_rate": format_float(eval_rate, 6),
        "eval_score_avg": format_float(mean(eval_scores), 6),
        "cases": observed_samples,
        "expected_samples": expected_samples,
        "observed_samples": observed_samples,
        "successful_samples": success_count,
        "sample_completeness": format_float(completeness),
        "success_wilson_lower_95": format_float(wilson),
        "valid_trials": valid_trials,
        "required_trials": required_trials,
        "trial_attempts": trial_attempts,
        "trial_retry_rate": format_float(retry_rate),
        "trial_tps_cv": format_float(tps_cv),
        "trial_ttft_cv": format_float(ttft_cv),
        "tps_relative_mad": format_float(tps_mad),
        "ttft_relative_mad": format_float(ttft_mad),
        "tps_p10": format_float(percentile(tps_values, 10), 3),
        "tps_p90": format_float(percentile(tps_values, 90), 3),
        "ttft_p90": format_float(percentile(ttft_values, 90), 3),
        "tps_median_ci95_low": format_float(tps_low, 3),
        "tps_median_ci95_high": format_float(tps_high, 3),
        "ttft_median_ci95_low": format_float(ttft_low, 3),
        "ttft_median_ci95_high": format_float(ttft_high, 3),
        "warmup_cv": format_float(float_value(preflight.get("warmup_cv"))),
        "warmup_stable": bool(preflight.get("stable")),
        "reliability_score": format_float(reliability_score),
        "reliability_pass": not failures,
        "reliability_failures": failures,
    }


def aggregate_results(
    trials: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    preflight: Mapping[str, Any],
    total_attempts: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    combined: List[Dict[str, Any]] = []
    for trial in trials:
        for row in trial["rows"]:
            enriched = dict(row)
            enriched["trial_index"] = trial["trial_index"]
            enriched["trial_attempt_index"] = trial["attempt_index"]
            enriched["resumed"] = trial.get("resumed", False)
            combined.append(enriched)

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in combined:
        grouped.setdefault(sample_group_key(row), []).append(row)
    summaries: List[Dict[str, Any]] = []
    case_count = len(cases)
    valid_count = len(trials)
    expected_per_endpoint = valid_count * case_count * args.repeats
    for index, (key, group_rows) in enumerate(sorted(grouped.items())):
        host_name, host_ip, base_url, model_key = key
        inv = next(
            row
            for row in inventory_rows
            if str(row.get("base_url", "")).rstrip("/") == base_url.rstrip("/")
            and str(row.get("model_key", "")) == model_key
        )
        reliability = aggregate_group(
            group_rows,
            valid_count,
            args.min_valid_trials,
            total_attempts,
            expected_per_endpoint,
            preflight.get(endpoint_key(inv), {}),
            args,
            seed_offset=index * 17,
        )
        summaries.append(
            {
                "run_id": "reliable",
                "host_name": host_name,
                "host_ip": host_ip,
                "base_url": base_url,
                "model_key": model_key,
                **reliability,
            }
        )

    task_grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for row in combined:
        if row.get("phase") != "run":
            continue
        key = (*sample_group_key(row), str(row.get("task_family", "")))
        task_grouped.setdefault(key, []).append(row)
    task_rows: List[Dict[str, Any]] = []
    for key, values in sorted(task_grouped.items()):
        host_name, host_ip, base_url, model_key, task_family = key
        successful = [row for row in values if bool_value(row.get("ok"))]
        tps = [
            value
            for value in (float_value(row.get("tokens_per_sec")) for row in successful)
            if value is not None
        ]
        ttft = [
            value
            for value in (float_value(row.get("ttft_s")) for row in successful)
            if value is not None
        ]
        eval_items = [row for row in values if row.get("eval_ok") not in {None, ""}]
        task_rows.append(
            {
                "run_id": "reliable",
                "host_name": host_name,
                "host_ip": host_ip,
                "base_url": base_url,
                "model_key": model_key,
                "task_family": task_family,
                "load_s": "",
                "ttft_med": format_float(median(ttft), 3),
                "tps_med": format_float(median(tps), 3),
                "ok_rate": format_float(
                    len(successful) / len(values) if values else 0.0, 6
                ),
                "eval_ok_rate": format_float(
                    sum(1 for row in eval_items if bool_value(row.get("eval_ok")))
                    / len(eval_items)
                    if eval_items
                    else 0.0,
                    6,
                ),
                "eval_score_avg": format_float(
                    mean(
                        [
                            score
                            for score in (
                                float_value(row.get("eval_score"))
                                for row in eval_items
                            )
                            if score is not None
                        ]
                    ),
                    6,
                ),
                "cases": len(values),
            }
        )
    return combined, summaries, task_rows


def write_reliability_markdown(
    path: Path, summaries: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> None:
    lines = [
        "# Reliable Benchmark Report",
        "",
        f"- Generated UTC: `{report['created_at_utc']}`",
        f"- Input fingerprint: `{report['input_fingerprint']}`",
        f"- Valid trials: `{report['valid_trials']}` / `{report['requested_trials']}`",
        f"- Trial attempts: `{report['trial_attempts']}`",
        f"- Overall pass: `{'yes' if report['passed'] else 'no'}`",
        "",
        "| Host | Model | TPS median | TPS CV | TTFT median | TTFT CV | Completeness | OK rate | Reliability |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in summaries:
        lines.append(
            f"| `{row.get('host_name','')}` | `{row.get('model_key','')}` | "
            f"{row.get('tps_med','')} | {row.get('trial_tps_cv','')} | "
            f"{row.get('ttft_med','')} | {row.get('trial_ttft_cv','')} | "
            f"{row.get('sample_completeness','')} | {row.get('ok_rate','')} | "
            f"{'✅' if row.get('reliability_pass') else '❌'} |"
        )
        failures = row.get("reliability_failures") or []
        if failures:
            lines.append(
                "|  | Failures | "
                + ", ".join(f"`{item}`" for item in failures)
                + " |  |  |  |  |  |  |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        inventory_path = Path(args.inventory_csv).resolve()
        suite_path = Path(args.cases_file).resolve() if args.cases_file else None
        output_dir = Path(args.output_dir).resolve()
        sidecar_dir = Path(args.sidecar_dir).resolve()
        ensure_dir(output_dir)
        ensure_dir(sidecar_dir)
        trials_root = output_dir / "trials"
        ensure_dir(trials_root)

        inventory_rows = load_filtered_inventory(args)
        cases, suite = _legacy.load_cases(args)
        case_keys = [case.case_key for case in cases]
        if len(case_keys) != len(set(case_keys)):
            raise ValueError("benchmark suite expands to duplicate case keys")
        input_fp, input_core = input_fingerprint(
            args, inventory_path, suite_path, case_keys
        )
        api_key = os.getenv(args.api_key_env)
        preflight = {
            endpoint_key(row): preflight_endpoint(row, args, api_key)
            for row in inventory_rows
        }
        write_json(output_dir / "preflight.json", preflight)
        if any(not item.get("ok") for item in preflight.values()):
            report = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "benchmark_reliability",
                "created_at_utc": utc_now_iso(),
                "input_fingerprint": input_fp,
                "passed": False,
                "failure": "preflight_failed",
                "preflight": preflight,
                "admission": {"admitted": False},
            }
            report["reliability_fingerprint"] = canonical_hash(
                {key: value for key, value in report.items() if key != "created_at_utc"}
            )
            write_json(output_dir / "reliability.json", report)
            return 0 if args.diagnostic_only else 1

        expected_keys = expected_sample_keys(
            inventory_rows, case_keys, args.repeats
        )
        timeout_s = auto_trial_timeout(
            args, len(inventory_rows), len(cases)
        )
        valid_trials: List[Dict[str, Any]] = []
        total_attempts = 0
        attempt_records: List[Dict[str, Any]] = []
        next_attempt = 1
        for trial_index in range(1, args.trials + 1):
            if args.resume:
                resumed = existing_valid_trial(
                    trials_root / f"trial_{trial_index:03d}", input_fp
                )
                if resumed:
                    valid_trials.append(resumed)
                    attempt_records.append(resumed["manifest"])
                    continue
            success: Optional[Dict[str, Any]] = None
            while total_attempts < args.max_trial_attempts:
                attempt = execute_trial_attempt(
                    args,
                    trial_index,
                    next_attempt,
                    trials_root,
                    inventory_rows,
                    suite_path,
                    expected_keys,
                    input_fp,
                    api_key,
                    timeout_s,
                )
                next_attempt += 1
                total_attempts += 1
                attempt_records.append(attempt["manifest"])
                if attempt["manifest"]["valid"]:
                    success = attempt
                    break
                if total_attempts < args.max_trial_attempts:
                    time.sleep(args.retry_backoff)
            if success is not None:
                valid_trials.append(success)
            if total_attempts >= args.max_trial_attempts and len(valid_trials) < trial_index:
                break
            if args.cooldown_between_trials > 0 and trial_index < args.trials:
                time.sleep(args.cooldown_between_trials)

        total_attempt_count = max(len(valid_trials), len(attempt_records))
        combined, summaries, task_rows = aggregate_results(
            valid_trials,
            inventory_rows,
            cases,
            preflight,
            total_attempt_count,
            args,
        )
        result_fields = [
            "run_id",
            "created_at_utc",
            "trial_index",
            "trial_attempt_index",
            "resumed",
            "phase",
            "host_name",
            "host_ip",
            "endpoint_id",
            "base_url",
            "model_id",
            "model_key",
            "case_key",
            "task_family",
            "priority",
            "context_tokens",
            "recommendation_signal",
            "repeat_index",
            "ok",
            "http_status",
            "error",
            "wall_s",
            "ttft_s",
            "load_s",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "tokens_per_sec",
            "finish_reason",
            "eval_ok",
            "eval_score",
            "eval_failed_json",
            "eval_result_json",
            "output_file",
        ]
        write_csv(output_dir / "run_results.csv", combined, result_fields)
        summary_fields = [
            "run_id",
            "host_name",
            "host_ip",
            "base_url",
            "model_key",
            "load_s",
            "ttft_med",
            "tps_med",
            "ok_rate",
            "eval_ok_rate",
            "eval_score_avg",
            "cases",
            "expected_samples",
            "observed_samples",
            "successful_samples",
            "sample_completeness",
            "success_wilson_lower_95",
            "valid_trials",
            "required_trials",
            "trial_attempts",
            "trial_retry_rate",
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
            "reliability_score",
            "reliability_pass",
            "reliability_failures",
        ]
        write_csv(output_dir / "run_summary.csv", summaries, summary_fields)
        task_fields = [
            "run_id",
            "host_name",
            "host_ip",
            "base_url",
            "model_key",
            "task_family",
            "load_s",
            "ttft_med",
            "tps_med",
            "ok_rate",
            "eval_ok_rate",
            "eval_score_avg",
            "cases",
        ]
        write_csv(output_dir / "task_summary.csv", task_rows, task_fields)
        passed = bool(
            len(valid_trials) >= args.min_valid_trials
            and summaries
            and all(row.get("reliability_pass") for row in summaries)
        )
        report_core = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "benchmark_reliability",
            "input_fingerprint": input_fp,
            "input": input_core,
            "suite_id": suite.get("suite_id", "legacy_default_cases"),
            "suite_version": suite.get("version"),
            "requested_trials": args.trials,
            "valid_trials": len(valid_trials),
            "trial_attempts": total_attempt_count,
            "passed": passed,
            "preflight": preflight,
            "summaries": summaries,
            "trial_manifests": attempt_records,
            "admission": {"admitted": False},
        }
        report = {
            **report_core,
            "created_at_utc": utc_now_iso(),
            "reliability_fingerprint": canonical_hash(report_core),
        }
        write_json(output_dir / "reliability.json", report)
        write_json(
            output_dir / "config.json",
            {
                **input_core,
                "input_fingerprint": input_fp,
                "created_at_utc": utc_now_iso(),
                "run_uuid": str(uuid.uuid4()),
                "trial_timeout_s": timeout_s,
            },
        )
        write_reliability_markdown(
            sidecar_dir / "RELIABILITY.md", summaries, report
        )
        print(f"Wrote reliable results to {output_dir}")
        return 0 if passed or args.diagnostic_only else 1
    except (OSError, ValueError, json.JSONDecodeError, requests.RequestException) as exc:
        print(f"reliable benchmark rejected: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
