"""Transactional runtime canary, soak, rollback, and evidence verification."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_operator as _operator

PLAN_SCHEMA_VERSION = "runtime_canary_plan.v1"
RUN_SCHEMA_VERSION = "runtime_canary_run.v1"
MANIFEST_SCHEMA_VERSION = "runtime_canary_manifest.v1"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUIRED_COMMANDS = (
    "snapshot",
    "start_candidate",
    "candidate_health",
    "qualification",
    "soak_probe",
    "stop_candidate",
    "rollback",
    "rollback_health",
)
_SECRET_FLAGS = ("--api-key", "--token", "--password", "--secret")
_SAFE_BASE_ENV = ("HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TMPDIR")
_MAX_CAPTURE_BYTES = 1024 * 1024


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    resolved = value.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _safe_directory(path: Path, label: str, *, create: bool = False) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    if create:
        value.mkdir(parents=True, exist_ok=True)
    resolved = value.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def validate_run_id(value: str) -> str:
    if not _RUN_ID_RE.fullmatch(str(value or "")):
        raise ValueError("run ID must use only letters, numbers, dot, dash, and underscore")
    return str(value)


def _default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return parsed


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return parsed


def _command(value: Any, label: str) -> Dict[str, Any]:
    command = _mapping(value, label)
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError(f"{label}.argv must be a nonempty string array")
    if not Path(argv[0]).is_absolute():
        raise ValueError(f"{label}.argv[0] must be an absolute executable path")
    for item in argv:
        if not item or "\x00" in item or "\n" in item or "\r" in item:
            raise ValueError(f"{label}.argv contains an unsafe argument")
        lowered = item.lower()
        if any(lowered == flag or lowered.startswith(flag + "=") for flag in _SECRET_FLAGS):
            raise ValueError(f"{label}.argv may not contain secret-bearing flags; use environment_names")
    executable = Path(argv[0]).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"{label} executable is not available: {executable}")
    timeout = _number(command.get("timeout_seconds", 300), f"{label}.timeout_seconds", minimum=0.1)
    return {
        "argv": list(argv),
        "timeout_seconds": timeout,
        "executable_resolved": str(executable),
        "executable_sha256": _operator.file_sha256(executable),
    }


def load_plan(path: Path) -> Dict[str, Any]:
    source = _regular_file(path, "canary plan")
    if source.stat().st_mode & 0o022:
        raise ValueError("canary plan may not be group/world writable")
    payload = json.loads(source.read_text(encoding="utf-8"))
    plan = _mapping(payload, "canary plan")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported runtime canary plan schema")
    canary_id = validate_run_id(str(plan.get("canary_id") or ""))
    loadout_fingerprint = str(plan.get("loadout_fingerprint") or "")
    if not _SHA_RE.fullmatch(loadout_fingerprint):
        raise ValueError("loadout_fingerprint must be a lowercase sha256 fingerprint")
    working_directory = _safe_directory(Path(str(plan.get("working_directory") or "")), "working directory")
    names = plan.get("environment_names") or []
    if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
        raise ValueError("environment_names must be a string array")
    if len(names) != len(set(names)) or not all(_ENV_RE.fullmatch(item) for item in names):
        raise ValueError("environment_names must contain unique valid environment variable names")
    missing = [name for name in names if name not in os.environ]
    if missing:
        raise ValueError("required environment variables are missing: " + ", ".join(sorted(missing)))
    commands_raw = _mapping(plan.get("commands"), "commands")
    commands = {name: _command(commands_raw.get(name), f"commands.{name}") for name in _REQUIRED_COMMANDS}
    soak_raw = _mapping(plan.get("soak"), "soak")
    soak = {
        "duration_seconds": _number(soak_raw.get("duration_seconds", 1800), "soak.duration_seconds", minimum=0.1),
        "interval_seconds": _number(soak_raw.get("interval_seconds", 10), "soak.interval_seconds", minimum=0.0),
        "minimum_samples": _integer(soak_raw.get("minimum_samples", 30), "soak.minimum_samples", minimum=1),
        "minimum_success_rate": _number(soak_raw.get("minimum_success_rate", 0.99), "soak.minimum_success_rate", minimum=0.0),
        "max_consecutive_failures": _integer(soak_raw.get("max_consecutive_failures", 1), "soak.max_consecutive_failures", minimum=0),
        "max_p95_latency_seconds": _number(soak_raw.get("max_p95_latency_seconds", 30), "soak.max_p95_latency_seconds", minimum=0.0),
        "max_rss_growth_bytes": _integer(soak_raw.get("max_rss_growth_bytes", 1024**3), "soak.max_rss_growth_bytes", minimum=0),
        "minimum_terminal_tps_ratio": _number(soak_raw.get("minimum_terminal_tps_ratio", 0.80), "soak.minimum_terminal_tps_ratio", minimum=0.0),
        "max_temperature_c": None,
    }
    if soak["minimum_success_rate"] > 1 or soak["minimum_terminal_tps_ratio"] > 1:
        raise ValueError("soak rates must not exceed 1")
    if soak_raw.get("max_temperature_c") is not None:
        soak["max_temperature_c"] = _number(soak_raw["max_temperature_c"], "soak.max_temperature_c", minimum=0.0)
    normalized = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "canary_id": canary_id,
        "loadout_fingerprint": loadout_fingerprint,
        "working_directory": str(working_directory),
        "environment_names": sorted(names),
        "commands": commands,
        "soak": soak,
        "admission": {"admitted": False},
    }
    normalized["plan_fingerprint"] = _operator.canonical_hash(normalized)
    normalized["source_path"] = str(source)
    return normalized


def _boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _lock_state(lock_dir: Path) -> str:
    owner_path = lock_dir / "owner.json"
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    if owner.get("hostname") != socket.gethostname():
        return "foreign"
    if owner.get("boot_id") and _boot_id() and owner.get("boot_id") != _boot_id():
        return "stale"
    try:
        os.kill(int(owner.get("pid")), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return "stale"
    except PermissionError:
        return "active"
    return "active"


def acquire_lock(workspace: Path, run_id: str, recover_stale: bool) -> Path:
    lock_dir = workspace / ".runtime-canary.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError:
        state = _lock_state(lock_dir)
        if state != "stale" or not recover_stale:
            raise ValueError(f"runtime canary lock is {state}; explicit stale recovery is required")
        archived = workspace / f".runtime-canary.lock.stale.{int(time.time())}.{os.getpid()}"
        os.replace(lock_dir, archived)
        lock_dir.mkdir()
    _operator.write_json(
        lock_dir / "owner.json",
        {
            "schema_version": "runtime_canary_lock.v1",
            "run_id": run_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "boot_id": _boot_id(),
            "started_at_utc": utc_now(),
        },
    )
    return lock_dir


def release_lock(lock_dir: Optional[Path]) -> None:
    if not lock_dir:
        return
    try:
        (lock_dir / "owner.json").unlink(missing_ok=True)
        lock_dir.rmdir()
    except OSError:
        pass


def _environment(names: Sequence[str]) -> Dict[str, str]:
    result = {name: os.environ[name] for name in names}
    for name in _SAFE_BASE_ENV:
        if name in os.environ:
            result[name] = os.environ[name]
    result["NO_COLOR"] = "1"
    result["TERM"] = "dumb"
    return result


def _terminate(process: subprocess.Popen[bytes]) -> None:
    # A probe's child may ignore SIGTERM or retain its stdout pipe even after
    # the leader exits. Terminate the whole session, then escalate for remaining
    # group members regardless of the leader's individual exit status.
    # All callers start a new session; independently daemonized children remain
    # outside this process-group cleanup guarantee.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _verified_command_argv(spec: Mapping[str, Any]) -> List[str]:
    """Refuse a changed lifecycle executable before each dispatch.

    This limits post-validation drift (including symlink retargeting), but is
    not an atomic guarantee against privileged concurrent modification.
    Operator-approved, owner-protected lifecycle wrappers remain required.
    """
    argv = list(spec["argv"])
    try:
        resolved = Path(argv[0]).resolve(strict=True)
        if (
            str(resolved) != spec["executable_resolved"]
            or not resolved.is_file()
            or not os.access(resolved, os.X_OK)
            or _operator.file_sha256(resolved) != spec["executable_sha256"]
        ):
            raise ValueError("lifecycle executable identity changed after plan validation")
    except (OSError, RuntimeError) as exc:
        raise ValueError("lifecycle executable changed or is unavailable") from exc
    # Retain the requested launcher path: virtualenv Python binaries may be
    # symlinks, and executing their resolved targets can lose venv semantics.
    return argv


def run_command(
    name: str,
    spec: Mapping[str, Any],
    *,
    cwd: Path,
    env: Mapping[str, str],
    logs: Path,
) -> Dict[str, Any]:
    argv = _verified_command_argv(spec)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=float(spec["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            returncode = process.returncode if process.returncode is not None else -signal.SIGKILL
    report = {
        "name": name,
        "argv": list(spec["argv"]),
        "executable_resolved": spec["executable_resolved"],
        "executable_sha256": spec["executable_sha256"],
        "timeout_seconds": spec["timeout_seconds"],
        "timed_out": timed_out,
        "returncode": returncode,
        "duration_seconds": time.monotonic() - started,
        "stdout": stdout_path.name,
        "stdout_sha256": _operator.file_sha256(stdout_path),
        "stderr": stderr_path.name,
        "stderr_sha256": _operator.file_sha256(stderr_path),
    }
    if timed_out:
        raise RuntimeError(f"{name} timed out")
    if returncode != 0:
        raise RuntimeError(f"{name} exited with status {returncode}")
    return report


def _captured_probe(spec: Mapping[str, Any], cwd: Path, env: Mapping[str, str]) -> Dict[str, Any]:
    argv = _verified_command_argv(spec)
    started = time.monotonic()
    # A timed-out probe can spawn descendants. Keep its entire process group
    # isolated so timeout cleanup reaches children, not only the probe parent.
    with subprocess.Popen(
        argv,
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=float(spec["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            stdout, stderr = process.communicate(timeout=2)
    if len(stdout) > _MAX_CAPTURE_BYTES or len(stderr) > _MAX_CAPTURE_BYTES:
        raise RuntimeError("soak probe output exceeded one MiB")
    payload: Dict[str, Any] = {}
    parse_error: Optional[str] = None
    lines = [line for line in stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
    if lines:
        try:
            raw = json.loads(lines[-1])
            if isinstance(raw, Mapping):
                payload = dict(raw)
            else:
                parse_error = "last stdout line was not a JSON object"
        except json.JSONDecodeError as exc:
            parse_error = str(exc)
    else:
        parse_error = "probe produced no JSON output"
    returncode = process.returncode if process.returncode is not None else -signal.SIGKILL
    ok = returncode == 0 and not timed_out and parse_error is None and payload.get("ok") is True
    sample = {
        "captured_at_utc": utc_now(),
        "ok": ok,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "parse_error": parse_error,
        "stderr_tail": stderr.decode("utf-8", errors="replace")[-1000:],
    }
    for key in ("latency_seconds", "rss_bytes", "temperature_c", "tps", "ttft_seconds"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            sample[key] = float(value)
    return sample


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _median_window(values: Sequence[float], *, first: bool) -> Optional[float]:
    if not values:
        return None
    width = min(max(3, len(values) // 4), len(values))
    window = values[:width] if first else values[-width:]
    return statistics.median(window)


def summarize_soak(samples: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> Dict[str, Any]:
    successes = [sample for sample in samples if sample.get("ok") is True]
    success_rate = len(successes) / len(samples) if samples else 0.0
    max_consecutive = 0
    current = 0
    for sample in samples:
        if sample.get("ok") is True:
            current = 0
        else:
            current += 1
            max_consecutive = max(max_consecutive, current)
    # Missing or invalid measurements may not silently disable RSS growth or
    # throughput retention checks. All successful probes must report usable
    # latency, RSS, and positive TPS; temperature is required when gated.
    def usable(sample: Mapping[str, Any], key: str, *, positive: bool = False) -> bool:
        value = sample.get(key)
        return type(value) in (int, float) and math.isfinite(value) and (
            value > 0 if positive else value >= 0
        )

    latencies = [
        float(s["latency_seconds"]) for s in successes if usable(s, "latency_seconds")
    ]
    rss = [float(s["rss_bytes"]) for s in successes if usable(s, "rss_bytes", positive=True)]
    tps = [float(s["tps"]) for s in successes if usable(s, "tps", positive=True)]
    temperatures = [
        float(s["temperature_c"]) for s in successes if usable(s, "temperature_c")
    ]
    p95 = _percentile(latencies, 0.95)
    rss_growth = max(0.0, rss[-1] - rss[0]) if len(rss) >= 2 else 0.0
    baseline_tps = _median_window(tps, first=True)
    terminal_tps = _median_window(tps, first=False)
    terminal_ratio = terminal_tps / baseline_tps if baseline_tps and terminal_tps is not None else None
    failures: List[str] = []
    for key in ("latency_seconds", "rss_bytes", "tps"):
        positive = key in {"rss_bytes", "tps"}
        if any(not usable(s, key, positive=positive) for s in successes):
            failures.append(f"invalid_{key}")
    if any(
        "temperature_c" in s and not usable(s, "temperature_c")
        for s in successes
    ) or (
        policy.get("max_temperature_c") is not None
        and any(not usable(s, "temperature_c") for s in successes)
    ):
        failures.append("invalid_temperature_c")
    if len(rss) < 2:
        failures.append("insufficient_rss_samples")
    if len(tps) < 2:
        failures.append("insufficient_tps_samples")
    if len(samples) < int(policy["minimum_samples"]):
        failures.append("insufficient_samples")
    if success_rate < float(policy["minimum_success_rate"]):
        failures.append("success_rate")
    if max_consecutive > int(policy["max_consecutive_failures"]):
        failures.append("consecutive_failures")
    if p95 is None or p95 > float(policy["max_p95_latency_seconds"]):
        failures.append("p95_latency")
    if rss_growth > int(policy["max_rss_growth_bytes"]):
        failures.append("rss_growth")
    if terminal_ratio is not None and terminal_ratio < float(policy["minimum_terminal_tps_ratio"]):
        failures.append("terminal_tps_ratio")
    max_temperature = max(temperatures) if temperatures else None
    if policy.get("max_temperature_c") is not None and (
        max_temperature is None or max_temperature > float(policy["max_temperature_c"])
    ):
        failures.append("temperature")
    return {
        "sample_count": len(samples),
        "successful_samples": len(successes),
        "success_rate": success_rate,
        "max_consecutive_failures": max_consecutive,
        "p95_latency_seconds": p95,
        "rss_growth_bytes": rss_growth,
        "baseline_tps": baseline_tps,
        "terminal_tps": terminal_tps,
        "terminal_tps_ratio": terminal_ratio,
        "max_temperature_c": max_temperature,
        "failures": failures,
        "passed": not failures,
    }


def run_soak(
    spec: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    cwd: Path,
    env: Mapping[str, str],
    output: Path,
) -> Dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(policy["duration_seconds"])
    samples: List[Dict[str, Any]] = []
    consecutive = 0
    while len(samples) < int(policy["minimum_samples"]) or time.monotonic() < deadline:
        sample = _captured_probe(spec, cwd, env)
        samples.append(sample)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        consecutive = 0 if sample["ok"] else consecutive + 1
        if consecutive > int(policy["max_consecutive_failures"]):
            break
        if len(samples) < int(policy["minimum_samples"]) or time.monotonic() < deadline:
            time.sleep(float(policy["interval_seconds"]))
    summary = summarize_soak(samples, policy)
    summary["duration_seconds"] = time.monotonic() - started
    summary["samples_sha256"] = _operator.file_sha256(output)
    if not summary["passed"]:
        raise RuntimeError("soak gate failed: " + ", ".join(summary["failures"]))
    return summary


def _state_payload(run_id: str, plan: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "canary_id": plan["canary_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "loadout_fingerprint": plan["loadout_fingerprint"],
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "stage": "validated",
        "success": False,
        "candidate_started": False,
        "candidate_start_attempted": False,
        "rollback_attempted": False,
        "rollback_succeeded": False,
        "commands": [],
        "soak": None,
        "failure": None,
        "admission": {"admitted": False},
    }


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    _operator.write_json(path, dict(state))


def _artifact_entries(root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    excluded = {
        "runtime-canary-manifest.json",
        "runtime-canary-manifest.json.sig",
        "runtime-canary-attestation.json",
    }
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"run artifact may not be a symlink: {path}")
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _operator.file_sha256(path),
            }
        )
    return entries


def build_manifest(root: Path, state: Mapping[str, Any]) -> Dict[str, Any]:
    core = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": state["run_id"],
        "canary_id": state["canary_id"],
        "success": state["success"] is True,
        "rollback_succeeded": state["rollback_succeeded"] is True,
        "plan_fingerprint": state["plan_fingerprint"],
        "loadout_fingerprint": state["loadout_fingerprint"],
        "artifacts": _artifact_entries(root),
        "admission": {"admitted": False},
    }
    manifest = {
        **core,
        "created_at_utc": utc_now(),
        "manifest_fingerprint": _operator.canonical_hash(core),
    }
    _operator.write_json(root / "runtime-canary-manifest.json", manifest)
    return manifest


def _verify_success_evidence(root: Path, state: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Require complete, internally consistent evidence before trusting success.

    The manifest is not an independent trust root. Cryptographic signer policy
    and off-host anti-rollback protection remain separate required approvals.
    """
    if state.get("stage") != "complete" or state.get("failure") is not None:
        raise ValueError("successful canary evidence is not at the complete stage")
    if any(state.get(key) is not True for key in (
        "success", "candidate_start_attempted", "candidate_started",
        "rollback_attempted", "rollback_succeeded",
    )):
        raise ValueError("successful canary evidence is missing lifecycle confirmations")
    if state.get("recovery_failures") or state.get("admission") != {"admitted": False}:
        raise ValueError("successful canary evidence contains recovery failures or admission")
    if manifest.get("admission") != {"admitted": False}:
        raise ValueError("canary manifest may not claim admission")

    plan = _mapping(json.loads(_regular_file(root / "plan.normalized.json", "normalized plan").read_text(encoding="utf-8")), "normalized plan")
    fingerprint = plan.get("plan_fingerprint")
    if fingerprint != _operator.canonical_hash({key: value for key, value in plan.items() if key != "plan_fingerprint"}):
        raise ValueError("normalized plan fingerprint mismatch")
    for key in ("canary_id", "loadout_fingerprint", "plan_fingerprint"):
        if plan.get(key) != state.get(key) or plan.get(key) != manifest.get(key):
            raise ValueError(f"successful canary evidence {key} mismatch")
    soak_policy = _mapping(plan.get("soak"), "normalized soak policy")

    expected_stages = (
        "snapshot", "start_candidate", "candidate_health", "qualification",
        "stop_candidate", "rollback", "rollback_health",
    )
    history = state.get("commands")
    if not isinstance(history, list) or [item.get("name") if isinstance(item, Mapping) else None for item in history] != list(expected_stages):
        raise ValueError("successful canary evidence has missing or out-of-order commands")
    normalized_commands = _mapping(plan.get("commands"), "normalized lifecycle commands")
    for name, result in zip(expected_stages, history):
        if result.get("returncode") != 0 or result.get("timed_out") is not False:
            raise ValueError(f"successful canary evidence includes failed command: {name}")
        approved = _mapping(normalized_commands.get(name), f"normalized command {name}")
        # Recorded success and log hashes alone do not establish the approved invocation.
        for field, expected_type in (
            ("argv", list),
            ("executable_resolved", str),
            ("executable_sha256", str),
            ("timeout_seconds", float),
        ):
            pinned = approved.get(field)
            recorded = result.get(field)
            if (
                type(pinned) is not expected_type
                or type(recorded) is not expected_type
                or pinned != recorded
            ):
                raise ValueError(f"successful canary command {name} {field} does not match normalized plan")
        for stream in ("stdout", "stderr"):
            expected_log = f"{name}.{stream}.log"
            if result.get(stream) != expected_log:
                raise ValueError(f"successful canary evidence has missing {stream} log: {name}")
            actual_hash = _operator.file_sha256(_regular_file(root / "logs" / expected_log, "canary log"))
            if result.get(f"{stream}_sha256") != actual_hash:
                raise ValueError(f"successful canary evidence {stream} hash mismatch: {name}")

    soak = _mapping(state.get("soak"), "canary soak result")
    samples_path = _regular_file(root / "soak-samples.jsonl", "raw soak samples")
    if samples_path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("canary soak sample evidence exceeds size limit")
    if soak.get("samples_sha256") != _operator.file_sha256(samples_path):
        raise ValueError("canary soak sample SHA-256 mismatch")
    samples: List[Dict[str, Any]] = []
    try:
        with samples_path.open(encoding="utf-8") as source:
            for line in source:
                if len(samples) >= 100000:
                    raise ValueError("canary soak contains too many samples")
                sample = _mapping(json.loads(line), "canary soak sample")
                # Success claims must agree with the actual probe execution
                # receipt; a rehashed summary must not conceal failed probes.
                required = (
                    "ok", "returncode", "timed_out", "parse_error",
                    "duration_seconds", "captured_at_utc",
                )
                if any(key not in sample for key in required):
                    raise ValueError("canary soak sample is missing execution status")
                if (
                    type(sample["ok"]) is not bool
                    or type(sample["returncode"]) is not int
                    or type(sample["timed_out"]) is not bool
                    or not (
                        sample["parse_error"] is None
                        or isinstance(sample["parse_error"], str)
                    )
                ):
                    raise ValueError("canary soak sample has invalid execution status")
                duration = sample["duration_seconds"]
                if (
                    type(duration) not in (int, float)
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    raise ValueError("canary soak sample has invalid duration")
                timestamp = sample["captured_at_utc"]
                if not isinstance(timestamp, str):
                    raise ValueError("canary soak sample has invalid timestamp")
                try:
                    captured = dt.datetime.fromisoformat(timestamp)
                except ValueError as exc:
                    raise ValueError("canary soak sample has invalid timestamp") from exc
                if captured.tzinfo is None or captured.utcoffset() != dt.timedelta(0):
                    raise ValueError("canary soak sample timestamp must be UTC")
                if sample["ok"] and (
                    sample["returncode"] != 0
                    or sample["timed_out"]
                    or sample["parse_error"] is not None
                ):
                    raise ValueError("canary soak sample success contradicts execution")
                samples.append(sample)
        recomputed = summarize_soak(samples, soak_policy)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid raw canary soak samples") from exc
    if soak.get("passed") is not True or not recomputed["passed"]:
        raise ValueError("successful canary soak is not qualified")
    if any(soak.get(key) != value for key, value in recomputed.items()):
        raise ValueError("canary soak result is inconsistent with raw samples")
    if _number(soak.get("duration_seconds"), "soak duration") < _number(soak_policy.get("duration_seconds"), "required soak duration"):
        raise ValueError("successful canary soak duration is insufficient")


def verify_manifest(run_dir: Path, *, require_success: bool = False) -> Dict[str, Any]:
    root = _safe_directory(run_dir, "canary run directory")
    manifest_path = _regular_file(root / "runtime-canary-manifest.json", "canary manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported runtime canary manifest schema")
    core = {key: value for key, value in manifest.items() if key not in {"created_at_utc", "manifest_fingerprint"}}
    if manifest.get("manifest_fingerprint") != _operator.canonical_hash(core):
        raise ValueError("runtime canary manifest fingerprint mismatch")
    seen = set()
    for item in manifest.get("artifacts") or []:
        entry = _mapping(item, "manifest artifact")
        relative = str(entry.get("path") or "")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or relative in seen:
            raise ValueError("runtime canary manifest contains an unsafe or duplicate path")
        seen.add(relative)
        path = _regular_file(root / candidate, "runtime canary artifact")
        if root not in path.parents:
            raise ValueError("runtime canary artifact escapes run directory")
        if path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"runtime canary artifact size mismatch: {relative}")
        if _operator.file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"runtime canary artifact SHA-256 mismatch: {relative}")
    state = json.loads((root / "runtime-canary-state.json").read_text(encoding="utf-8"))
    if state.get("run_id") != manifest.get("run_id") or state.get("plan_fingerprint") != manifest.get("plan_fingerprint"):
        raise ValueError("runtime canary state does not match manifest")
    if bool(state.get("success")) != bool(manifest.get("success")):
        raise ValueError("runtime canary success does not match manifest")
    if bool(state.get("rollback_succeeded")) != bool(manifest.get("rollback_succeeded")):
        raise ValueError("runtime canary rollback status does not match manifest")
    if state.get("recovery_failures") and state.get("rollback_succeeded") is True:
        raise ValueError("runtime canary cannot claim successful rollback with recovery failures")
    # A recomputed, locally writable manifest hash must not allow an attacker to
    # silently omit a run artifact or add undisclosed evidence after sealing.
    if manifest.get("artifacts") != _artifact_entries(root):
        raise ValueError("runtime canary manifest artifact inventory is incomplete or changed")
    if manifest.get("success") is True:
        _verify_success_evidence(root, state, manifest)
    if require_success and (manifest.get("success") is not True or manifest.get("rollback_succeeded") is not True):
        raise ValueError("runtime canary did not complete successfully with verified rollback")
    return {
        "valid": True,
        "run_id": manifest.get("run_id"),
        "canary_id": manifest.get("canary_id"),
        "success": manifest.get("success") is True,
        "rollback_succeeded": manifest.get("rollback_succeeded") is True,
        "plan_fingerprint": manifest.get("plan_fingerprint"),
        "loadout_fingerprint": manifest.get("loadout_fingerprint"),
        "manifest_fingerprint": manifest.get("manifest_fingerprint"),
        "artifact_count": len(seen),
        "admission": {"admitted": False},
    }


def execute(plan_path: Path, workspace_path: Path, run_id_value: Optional[str], recover_stale_lock: bool) -> Dict[str, Any]:
    plan = load_plan(plan_path)
    workspace = _safe_directory(workspace_path, "canary workspace", create=True)
    run_id = validate_run_id(run_id_value or _default_run_id())
    run_dir = workspace / run_id
    if run_dir.exists():
        raise ValueError(f"runtime canary run already exists: {run_dir}")
    lock: Optional[Path] = None
    state = _state_payload(run_id, plan)
    candidate_start_attempted = False
    env = _environment(plan["environment_names"])
    try:
        lock = acquire_lock(workspace, run_id, recover_stale_lock)
        run_dir.mkdir(mode=0o700)
        logs = run_dir / "logs"
        logs.mkdir(mode=0o700)
        _operator.write_json(run_dir / "plan.normalized.json", {key: value for key, value in plan.items() if key != "source_path"})
        state_path = run_dir / "runtime-canary-state.json"
        _write_state(state_path, state)
        cwd = Path(plan["working_directory"])

        for name in ("snapshot", "start_candidate", "candidate_health", "qualification"):
            state["stage"] = name
            if name == "start_candidate":
                # A failing start wrapper may have already spawned the model.
                # Arm cleanup BEFORE launching it, not only on exit zero.
                candidate_start_attempted = True
                state["candidate_start_attempted"] = True
            _write_state(state_path, state)
            result = run_command(name, plan["commands"][name], cwd=cwd, env=env, logs=logs)
            state["commands"].append(result)
            if name == "start_candidate":
                state["candidate_started"] = True
            _write_state(state_path, state)

        state["stage"] = "soak"
        _write_state(state_path, state)
        state["soak"] = run_soak(
            plan["commands"]["soak_probe"],
            plan["soak"],
            cwd=cwd,
            env=env,
            output=run_dir / "soak-samples.jsonl",
        )
        _write_state(state_path, state)

        for name in ("stop_candidate", "rollback", "rollback_health"):
            state["stage"] = name
            if name == "rollback":
                state["rollback_attempted"] = True
            _write_state(state_path, state)
            result = run_command(name, plan["commands"][name], cwd=cwd, env=env, logs=logs)
            state["commands"].append(result)
            if name == "rollback_health":
                state["rollback_succeeded"] = True
            _write_state(state_path, state)
        state["success"] = True
        state["stage"] = "complete"
    except BaseException as exc:
        state["failure"] = {"type": type(exc).__name__, "message": str(exc), "stage": state.get("stage")}
        if candidate_start_attempted:
            # Discard any prior success flag when entering a fresh recovery.
            # Only the complete recovery sequence may re-establish it.
            state["rollback_succeeded"] = False
            for name in ("stop_candidate", "rollback", "rollback_health"):
                if name == "rollback":
                    state["rollback_attempted"] = True
                try:
                    result = run_command(f"recovery-{name}", plan["commands"][name], cwd=Path(plan["working_directory"]), env=env, logs=run_dir / "logs")
                    state["commands"].append(result)
                    if name == "rollback_health" and not state.get("recovery_failures"):
                        # A green health probe cannot establish rollback if any
                        # required recovery command failed or timed out.
                        state["rollback_succeeded"] = True
                except BaseException as recovery_exc:
                    state.setdefault("recovery_failures", []).append(
                        {"command": name, "type": type(recovery_exc).__name__, "message": str(recovery_exc)}
                    )
        state["stage"] = "failed" if state.get("rollback_succeeded") else "rollback_failed"
    finally:
        if run_dir.exists():
            state["completed_at_utc"] = utc_now()
            _write_state(run_dir / "runtime-canary-state.json", state)
            build_manifest(run_dir, state)
        release_lock(lock)
    if not state["success"]:
        failure = state.get("failure") or {}
        raise RuntimeError(
            f"runtime canary failed at {failure.get('stage')}: {failure.get('message')}; "
            f"rollback_succeeded={state.get('rollback_succeeded')}"
        )
    return verify_manifest(run_dir, require_success=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms-runtime-canary",
        description="Run a transactional candidate lifecycle, sustained soak, and verified rollback",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--recover-stale-lock", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--require-success", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            plan = load_plan(args.plan)
            report = {
                "valid": True,
                "canary_id": plan["canary_id"],
                "plan_fingerprint": plan["plan_fingerprint"],
                "loadout_fingerprint": plan["loadout_fingerprint"],
                "environment_names": plan["environment_names"],
                "admission": {"admitted": False},
            }
        elif args.command == "verify":
            report = verify_manifest(args.run_dir, require_success=args.require_success)
        else:
            report = execute(args.plan, args.workspace, args.run_id, args.recover_stale_lock)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"runtime canary failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
