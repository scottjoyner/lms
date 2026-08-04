"""Deterministic, fail-closed operator workflow for physical fleet evidence.

The operator validates controller and remote readiness, uses explicit SSH trust,
prevents overlapping runs, renders immutable scripts, executes observation,
collects and verifies artifacts, performs a postflight state check, and writes a
cryptographically verifiable local run manifest. It never invents candidates,
admits runtimes, restores KV state, or modifies routing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from lms_agent_bench import fleet_rollout_entrypoint
from lms_agent_bench.fleet_coverage import validate_rollout_coverage
from lms_agent_bench.fleet_operational_hardening import SAFE_SSH_OPTIONS

SCHEMA_VERSION = "fleet_operator_run.v2"
MANIFEST_SCHEMA_VERSION = "fleet_operator_manifest.v1"
DEFAULT_CONTROLLER_FREE_BYTES = 1024**3
DEFAULT_REMOTE_FREE_BYTES = 1024**3


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON so interrupted writes are never mistaken as valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def load_operator_config(config_path: str, env_file: str) -> Dict[str, Any]:
    config = fleet_rollout_entrypoint.load_rollout_config(config_path, env_file)
    coverage = validate_rollout_coverage(config, config_path)
    if not coverage.get("ready"):
        raise ValueError(
            "; ".join(coverage.get("errors") or ["fleet coverage is not ready"])
        )
    return {"config": config, "coverage": coverage}


def resolved_ssh_options(accept_new_host_keys: bool = False) -> Tuple[str, ...]:
    if not accept_new_host_keys:
        return SAFE_SSH_OPTIONS
    return tuple(
        "StrictHostKeyChecking=accept-new"
        if value.startswith("StrictHostKeyChecking=")
        else value
        for value in SAFE_SSH_OPTIONS
    )


def ssh_command(target: str, accept_new_host_keys: bool = False) -> List[str]:
    command = ["ssh"]
    for option in resolved_ssh_options(accept_new_host_keys):
        command.extend(["-o", option])
    command.extend([target, "bash", "-s"])
    return command


def _safe_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    ).strip("-") or "node"


def preflight_script(
    node: Mapping[str, Any],
    update_code: bool,
    *,
    run_id_value: Optional[str] = None,
    phase: str = "preflight",
) -> str:
    if phase not in {"preflight", "postflight"}:
        raise ValueError("preflight phase must be preflight or postflight")
    values = {
        "LMS_PREFLIGHT_NODE": str(node["node_id"]),
        "LMS_PREFLIGHT_REPO": str(node["repo_dir"]),
        "LMS_PREFLIGHT_BRANCH": str(node["branch"]),
        "LMS_PREFLIGHT_COMMIT": str(node.get("expected_commit") or ""),
        "LMS_PREFLIGHT_PYTHON": str(node.get("python") or "python3"),
        "LMS_PREFLIGHT_ROOTS": json.dumps(node.get("model_roots") or []),
        "LMS_PREFLIGHT_ALLOW_UPDATE": "1" if update_code else "0",
        "LMS_PREFLIGHT_ARTIFACT_ROOT": str(
            node.get("artifact_root") or "~/.local/state/lms-fleet"
        ),
        "LMS_PREFLIGHT_LOCK_ROOT": str(
            node.get("lock_root") or "~/.local/state/lms-fleet/locks"
        ),
        "LMS_PREFLIGHT_MIN_FREE": str(
            int(node.get("min_artifact_free_bytes") or DEFAULT_REMOTE_FREE_BYTES)
        ),
        "LMS_PREFLIGHT_MIN_NOFILE": str(
            int(node.get("min_open_file_limit") or 1024)
        ),
        "LMS_PREFLIGHT_RUN_ID": str(run_id_value or ""),
        "LMS_PREFLIGHT_PHASE": phase,
    }
    exports = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in values.items()
    )
    return exports + r'''
set -euo pipefail
PYTHON_BIN="$LMS_PREFLIGHT_PYTHON"
if [[ "$PYTHON_BIN" == */* ]]; then
  test -x "$PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null
fi
"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import pathlib
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time

node = os.environ["LMS_PREFLIGHT_NODE"]
repo = pathlib.Path(os.path.expanduser(os.environ["LMS_PREFLIGHT_REPO"]))
expected_branch = os.environ["LMS_PREFLIGHT_BRANCH"]
expected_commit = os.environ["LMS_PREFLIGHT_COMMIT"].lower()
allow_update = os.environ["LMS_PREFLIGHT_ALLOW_UPDATE"] == "1"
roots = [
    pathlib.Path(os.path.expanduser(item))
    for item in json.loads(os.environ["LMS_PREFLIGHT_ROOTS"])
]
artifact_root = pathlib.Path(
    os.path.expanduser(os.environ["LMS_PREFLIGHT_ARTIFACT_ROOT"])
)
lock_root = pathlib.Path(os.path.expanduser(os.environ["LMS_PREFLIGHT_LOCK_ROOT"]))
minimum_free = int(os.environ["LMS_PREFLIGHT_MIN_FREE"])
minimum_nofile = int(os.environ["LMS_PREFLIGHT_MIN_NOFILE"])
run_id = os.environ["LMS_PREFLIGHT_RUN_ID"]
phase = os.environ["LMS_PREFLIGHT_PHASE"]

def git(*args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

def safe_slug(value):
    return ''.join(
        ch if ch.isalnum() or ch in '-_.' else '-' for ch in value
    ).strip('-') or 'node'

def boot_id():
    try:
        return pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return None

def lock_state(path):
    if not path.is_dir():
        return 'absent', None
    owner_path = path / 'owner.json'
    try:
        owner = json.loads(owner_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return 'unknown', None
    if owner.get('hostname') != socket.gethostname():
        return 'foreign', owner
    current_boot = boot_id()
    if current_boot and owner.get('boot_id') and owner.get('boot_id') != current_boot:
        return 'stale', owner
    try:
        pid = int(owner.get('pid'))
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return 'stale', owner
    except PermissionError:
        return 'active', owner
    return 'active', owner

for command in ('git', 'tar', 'gzip'):
    if shutil.which(command) is None:
        raise SystemExit(f"required remote command is missing: {command}")
if not (repo / ".git").is_dir():
    raise SystemExit(f"not a git checkout: {repo}")
if importlib.util.find_spec("requests") is None:
    raise SystemExit("remote Python cannot import requests")
missing_roots = [str(path) for path in roots if not path.is_dir()]
if missing_roots:
    raise SystemExit("missing model roots: " + ", ".join(missing_roots))
unreadable_roots = [
    str(path) for path in roots if not os.access(path, os.R_OK | os.X_OK)
]
if unreadable_roots:
    raise SystemExit("unreadable model roots: " + ", ".join(unreadable_roots))
status = git("status", "--porcelain", "--untracked-files=all")
if status:
    raise SystemExit("remote checkout is not completely clean")
branch = git("branch", "--show-current")
commit = git("rev-parse", "HEAD").lower()
if not allow_update:
    if branch != expected_branch:
        raise SystemExit(f"expected branch {expected_branch}, found {branch or 'detached'}")
    if commit != expected_commit:
        raise SystemExit(f"expected commit {expected_commit}, found {commit}")

artifact_root.mkdir(parents=True, exist_ok=True)
lock_root.mkdir(parents=True, exist_ok=True)
probe = artifact_root / f'.lms-write-probe-{os.getpid()}'
try:
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b'probe')
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
finally:
    probe.unlink(missing_ok=True)
free_bytes = shutil.disk_usage(artifact_root).free
if free_bytes < minimum_free:
    raise SystemExit(
        f"artifact filesystem free bytes {free_bytes} below required {minimum_free}"
    )
soft_nofile, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
if soft_nofile < minimum_nofile:
    raise SystemExit(
        f"open-file soft limit {soft_nofile} below required {minimum_nofile}"
    )
lock_path = lock_root / f'{safe_slug(node)}.lock'
remote_lock_state, remote_lock_owner = lock_state(lock_path)
if remote_lock_state in {'active', 'unknown', 'foreign'}:
    raise SystemExit(
        f"remote rollout lock is {remote_lock_state}: {lock_path}"
    )

artifact_dir = None
artifact_tar = None
if run_id:
    artifact_dir = artifact_root / run_id / safe_slug(node)
    artifact_tar = pathlib.Path(str(artifact_dir) + '.tar.gz')
    if phase == 'preflight' and (artifact_dir.exists() or artifact_tar.exists()):
        raise SystemExit("remote run ID already has artifacts")
    if phase == 'postflight':
        if not artifact_dir.is_dir():
            raise SystemExit("remote artifact directory is missing after rollout")
        if not artifact_tar.is_file() or artifact_tar.stat().st_size <= 0:
            raise SystemExit("remote artifact archive is missing or empty after rollout")

print(json.dumps({
    "node_id": node,
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python": sys.executable,
    "repo": str(repo),
    "branch": branch,
    "commit": commit,
    "origin": git("remote", "get-url", "origin"),
    "model_roots": [str(path) for path in roots],
    "artifact_root": str(artifact_root),
    "artifact_free_bytes": free_bytes,
    "open_file_soft_limit": soft_nofile,
    "open_file_hard_limit": hard_nofile,
    "remote_lock_state": remote_lock_state,
    "remote_lock_owner": remote_lock_owner,
    "artifact_dir": str(artifact_dir) if artifact_dir else None,
    "artifact_tar": str(artifact_tar) if artifact_tar else None,
    "artifact_tar_size_bytes": (
        artifact_tar.stat().st_size if artifact_tar and artifact_tar.is_file() else None
    ),
    "epoch_seconds": time.time(),
    "boot_id": boot_id(),
    "clean": True,
    "update_code_requested": allow_update,
    "phase": phase,
}, sort_keys=True))
PY
'''


def preflight_node(
    node: Mapping[str, Any],
    update_code: bool,
    *,
    accept_new_host_keys: bool = False,
    run_id_value: Optional[str] = None,
    phase: str = "preflight",
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    started = utc_now()
    monotonic_started = time.monotonic()
    wall_started = time.time()
    try:
        proc = subprocess.run(
            ssh_command(str(node["ssh_target"]), accept_new_host_keys),
            input=preflight_script(
                node,
                update_code,
                run_id_value=run_id_value,
                phase=phase,
            ),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        return {
            "node_id": node["node_id"],
            "ssh_target": node["ssh_target"],
            "phase": phase,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "duration_seconds": time.monotonic() - monotonic_started,
            "returncode": 124,
            "timed_out": True,
            "ok": False,
            "errors": [f"SSH {phase} timed out after {timeout_seconds} seconds"],
            "stdout": str(exc.stdout or "").strip(),
            "stderr": str(exc.stderr or "").strip(),
        }

    detail: Optional[Dict[str, Any]] = None
    if proc.returncode == 0:
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
                if isinstance(parsed, dict):
                    detail = parsed
            except json.JSONDecodeError:
                detail = None
    errors: List[str] = []
    if proc.returncode != 0:
        errors.append(f"remote {phase} returned {proc.returncode}")
    if detail is None:
        errors.append("remote readiness did not return structured detail")
    else:
        if detail.get("node_id") != node["node_id"]:
            errors.append("remote readiness returned the wrong node identity")
        expected_hostnames = [str(value) for value in node.get("expected_hostnames", [])]
        if expected_hostnames and str(detail.get("hostname")) not in expected_hostnames:
            errors.append("remote hostname does not match expected_hostnames")
        try:
            remote_epoch = float(detail["epoch_seconds"])
            controller_midpoint = (wall_started + time.time()) / 2.0
            clock_skew = abs(remote_epoch - controller_midpoint)
        except (KeyError, TypeError, ValueError):
            clock_skew = None
            errors.append("remote readiness did not provide a valid clock")
        else:
            maximum = float(node.get("max_clock_skew_seconds") or 300)
            if clock_skew > maximum:
                errors.append(
                    f"remote clock skew {clock_skew:.3f}s exceeds {maximum:.3f}s"
                )
    return {
        "node_id": node["node_id"],
        "ssh_target": node["ssh_target"],
        "phase": phase,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_seconds": time.monotonic() - monotonic_started,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "ok": not errors,
        "errors": errors,
        "clock_skew_seconds": clock_skew if detail is not None else None,
        "detail": detail,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def run_preflight(
    config: Mapping[str, Any],
    update_code: bool,
    *,
    accept_new_host_keys: bool = False,
    run_id_value: Optional[str] = None,
    phase: str = "preflight",
    timeout_seconds: int = 60,
) -> List[Dict[str, Any]]:
    return [
        preflight_node(
            node,
            update_code,
            accept_new_host_keys=accept_new_host_keys,
            run_id_value=run_id_value,
            phase=phase,
            timeout_seconds=timeout_seconds,
        )
        for node in config["nodes"]
    ]


def controller_readiness(
    config_path: str,
    env_file: str,
    workspace: Path,
    *,
    minimum_free_bytes: int,
    allow_insecure_env_file: bool,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    config = Path(config_path).expanduser().resolve()
    environment = Path(env_file).expanduser().resolve()
    for name, path in (("config", config), ("env_file", environment)):
        if not path.is_file():
            errors.append(f"{name} is not a regular file: {path}")
    for command in ("ssh", "scp"):
        if shutil.which(command) is None:
            errors.append(f"controller command is missing: {command}")
    if environment.is_file() and os.name == "posix":
        mode = environment.stat().st_mode & 0o777
        if mode & 0o077:
            message = (
                f"environment file permissions are {mode:03o}; expected no group/world access"
            )
            if allow_insecure_env_file:
                warnings.append(message)
            else:
                errors.append(message)
    workspace.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(workspace).free
    if free_bytes < minimum_free_bytes:
        errors.append(
            f"controller workspace free bytes {free_bytes} below required {minimum_free_bytes}"
        )
    return {
        "schema_version": "fleet_controller_readiness.v1",
        "created_at_utc": utc_now(),
        "controller_hostname": socket.gethostname(),
        "controller_pid": os.getpid(),
        "config": str(config),
        "config_sha256": file_sha256(config) if config.is_file() else None,
        "env_file": str(environment),
        "env_file_sha256": file_sha256(environment) if environment.is_file() else None,
        "workspace": str(workspace),
        "workspace_free_bytes": free_bytes,
        "minimum_workspace_free_bytes": minimum_free_bytes,
        "ssh_path": shutil.which("ssh"),
        "scp_path": shutil.which("scp"),
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
        "admission": {"admitted": False},
    }


def _boot_id() -> Optional[str]:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def _process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(
    workspace: Path,
    *,
    current_run: str,
    config_sha256: Optional[str],
    recover_stale: bool,
) -> Tuple[Path, Optional[Path]]:
    lock = workspace / ".fleet-operator.lock"
    recovered: Optional[Path] = None
    try:
        lock.mkdir(parents=False)
    except FileExistsError as exc:
        try:
            owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner = None
        same_host = isinstance(owner, Mapping) and owner.get("hostname") == socket.gethostname()
        same_boot = (
            isinstance(owner, Mapping)
            and (not owner.get("boot_id") or not _boot_id() or owner.get("boot_id") == _boot_id())
        )
        active = same_host and same_boot and _process_alive(owner.get("pid"))
        if active:
            raise RuntimeError(f"fleet operator lock is active: {lock}") from exc
        safely_stale = same_host and (not same_boot or not _process_alive(owner.get("pid")))
        if not recover_stale or not safely_stale:
            raise RuntimeError(
                f"fleet operator lock exists and cannot be safely recovered: {lock}; "
                "use --recover-stale-lock only after verifying the recorded owner"
            ) from exc
        recovered = workspace / (
            f".fleet-operator.lock.stale.{run_id()}.{uuid.uuid4().hex[:8]}"
        )
        os.replace(lock, recovered)
        lock.mkdir(parents=False)
    write_json(
        lock / "owner.json",
        {
            "schema_version": "fleet_operator_lock.v2",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "boot_id": _boot_id(),
            "run_id": current_run,
            "config_sha256": config_sha256,
            "started_at_utc": utc_now(),
        },
    )
    return lock, recovered


def release_lock(lock: Path) -> None:
    try:
        for child in lock.iterdir():
            child.unlink()
        lock.rmdir()
    except OSError:
        pass


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait()


def run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    timeout_seconds: int,
) -> Dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    monotonic_started = time.monotonic()
    returncode = 1
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(item) for item in command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            returncode = 124
            log.write(f"\ncommand timed out after {timeout_seconds} seconds\n")
        log.flush()
        os.fsync(log.fileno())
    return {
        "command": list(command),
        "log": str(log_path),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_seconds": time.monotonic() - monotonic_started,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "log_sha256": file_sha256(log_path),
    }


def module_command(module: str, *args: str) -> List[str]:
    return [sys.executable, "-m", module, *args]


def _control_files(root: Path) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "operator-manifest.json":
            continue
        if path.name.endswith(".tar.gz"):
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return files


def _archive_references(root: Path) -> List[Dict[str, Any]]:
    results_path = root / "observe" / "rollout_results.json"
    if not results_path.is_file():
        return []
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    references: List[Dict[str, Any]] = []
    for item in payload.get("results", []):
        if not isinstance(item, Mapping) or not item.get("collected_artifact"):
            continue
        references.append(
            {
                "node_id": item.get("node_id"),
                "path": str(item.get("collected_artifact")),
                "size_bytes": item.get("collected_artifact_size_bytes"),
                "sha256": item.get("collected_artifact_sha256"),
            }
        )
    return references


def build_run_manifest(root: Path, state: Mapping[str, Any]) -> Dict[str, Any]:
    core = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": state.get("run_id"),
        "success": state.get("success") is True,
        "config_sha256": state.get("config_sha256"),
        "env_file_sha256": state.get("env_file_sha256"),
        "control_files": _control_files(root),
        "archives": _archive_references(root),
        "admission": {"admitted": False},
    }
    manifest = {
        **core,
        "created_at_utc": utc_now(),
        "operator_manifest_fingerprint": canonical_hash(core),
    }
    write_json(root / "operator-manifest.json", manifest)
    return manifest


def verify_run_manifest(root: Path, *, require_success: bool = False) -> Dict[str, Any]:
    path = root / "operator-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported operator manifest schema")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at_utc", "operator_manifest_fingerprint"}
    }
    if manifest.get("operator_manifest_fingerprint") != canonical_hash(core):
        raise ValueError("operator manifest fingerprint mismatch")
    for entry in manifest.get("control_files", []):
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe control-file path in operator manifest")
        file_path = root / relative
        if not file_path.is_file():
            raise ValueError(f"operator control file is missing: {relative}")
        if file_path.stat().st_size != int(entry.get("size_bytes")):
            raise ValueError(f"operator control file size mismatch: {relative}")
        if file_sha256(file_path) != entry.get("sha256"):
            raise ValueError(f"operator control file hash mismatch: {relative}")
    for archive in manifest.get("archives", []):
        archive_path = Path(str(archive.get("path") or ""))
        if not archive_path.is_file():
            raise ValueError(f"collected archive is missing: {archive_path}")
        if archive_path.stat().st_size != int(archive.get("size_bytes")):
            raise ValueError(f"collected archive size mismatch: {archive_path}")
        if file_sha256(archive_path) != archive.get("sha256"):
            raise ValueError(f"collected archive hash mismatch: {archive_path}")
    if require_success and manifest.get("success") is not True:
        raise ValueError("operator run did not complete successfully")
    return {
        "valid": True,
        "run_id": manifest.get("run_id"),
        "success": manifest.get("success") is True,
        "operator_manifest_fingerprint": manifest.get(
            "operator_manifest_fingerprint"
        ),
        "control_file_count": len(manifest.get("control_files", [])),
        "archive_count": len(manifest.get("archives", [])),
        "admission": {"admitted": False},
    }


def _common_rollout_args(
    args: argparse.Namespace,
    *,
    current_run: str,
    output_dir: Path,
    command: str,
) -> List[str]:
    values = [
        command,
        "--config",
        args.config,
        "--env-file",
        args.env_file,
        "--all-nodes",
        "--run-id",
        current_run,
        "--output-dir",
        str(output_dir),
    ]
    if args.update_code:
        values.append("--update-code")
    for option in resolved_ssh_options(args.accept_new_host_keys):
        values.extend(["--ssh-option", option])
    if args.accept_new_host_keys:
        values.append("--allow-accept-new-host-keys")
    return values


def preflight_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    controller = controller_readiness(
        args.config,
        args.env_file,
        workspace,
        minimum_free_bytes=args.min_controller_free_bytes,
        allow_insecure_env_file=args.allow_insecure_env_file,
    )
    if not controller["ok"]:
        write_json(workspace / "preflight.json", controller)
        return 1
    loaded = load_operator_config(args.config, args.env_file)
    results = run_preflight(
        loaded["config"],
        args.update_code,
        accept_new_host_keys=args.accept_new_host_keys,
        timeout_seconds=args.preflight_timeout_seconds,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "phase": "preflight",
        "created_at_utc": utc_now(),
        "controller": controller,
        "coverage": loaded["coverage"],
        "results": results,
        "ssh_trust_mode": (
            "accept_new_explicit"
            if args.accept_new_host_keys
            else "strict_known_hosts"
        ),
        "ok": all(item["ok"] for item in results),
        "admission": {"admitted": False},
    }
    write_json(workspace / "preflight.json", report)
    return 0 if report["ok"] else 1


def observe_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    controller = controller_readiness(
        args.config,
        args.env_file,
        workspace,
        minimum_free_bytes=args.min_controller_free_bytes,
        allow_insecure_env_file=args.allow_insecure_env_file,
    )
    loaded = load_operator_config(args.config, args.env_file)
    current_run = args.run_id or run_id()
    root = workspace / current_run
    root.mkdir(parents=True, exist_ok=False)
    lock, recovered_lock = acquire_lock(
        workspace,
        current_run=current_run,
        config_sha256=controller.get("config_sha256"),
        recover_stale=args.recover_stale_lock,
    )
    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": "observe",
        "run_id": current_run,
        "started_at_utc": utc_now(),
        "config": controller.get("config"),
        "config_sha256": controller.get("config_sha256"),
        "env_file": controller.get("env_file"),
        "env_file_sha256": controller.get("env_file_sha256"),
        "coverage": loaded["coverage"],
        "node_ids": [item["node_id"] for item in loaded["config"]["nodes"]],
        "update_code": args.update_code,
        "ssh_trust_mode": (
            "accept_new_explicit"
            if args.accept_new_host_keys
            else "strict_known_hosts"
        ),
        "controller": controller,
        "recovered_local_lock": str(recovered_lock) if recovered_lock else None,
        "success": False,
        "admission": {"admitted": False},
    }
    state_path = root / "operator-state.json"
    write_json(state_path, state)
    outcome = 1
    try:
        if not controller["ok"]:
            state["failure_stage"] = "controller_readiness"
        else:
            preflight = run_preflight(
                loaded["config"],
                args.update_code,
                accept_new_host_keys=args.accept_new_host_keys,
                run_id_value=current_run,
                phase="preflight",
                timeout_seconds=args.preflight_timeout_seconds,
            )
            state["preflight"] = preflight
            write_json(
                root / "preflight.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": "preflight",
                    "coverage": loaded["coverage"],
                    "results": preflight,
                    "ok": all(item["ok"] for item in preflight),
                    "admission": {"admitted": False},
                },
            )
            if not all(item["ok"] for item in preflight):
                state["failure_stage"] = "preflight"
            else:
                render_args = _common_rollout_args(
                    args,
                    current_run=current_run,
                    output_dir=root / "render",
                    command="render",
                )
                render = run_logged(
                    module_command(
                        "lms_agent_bench.fleet_rollout_complete", *render_args
                    ),
                    root / "logs" / "render.log",
                    timeout_seconds=args.render_timeout_seconds,
                )
                state["render"] = render
                if render["returncode"] != 0:
                    state["failure_stage"] = "render"
                else:
                    rollout_args = _common_rollout_args(
                        args,
                        current_run=current_run,
                        output_dir=root / "observe",
                        command="run",
                    )
                    rollout_args.append("--continue-on-error")
                    auto_rollout_timeout = sum(
                        int(node.get("remote_timeout_seconds") or 7200)
                        + int(node.get("scp_timeout_seconds") or 300)
                        for node in loaded["config"]["nodes"]
                    ) + 600
                    rollout_timeout = (
                        args.rollout_timeout_seconds
                        if args.rollout_timeout_seconds > 0
                        else auto_rollout_timeout
                    )
                    rollout = run_logged(
                        module_command(
                            "lms_agent_bench.fleet_rollout_complete", *rollout_args
                        ),
                        root / "logs" / "observe.log",
                        timeout_seconds=rollout_timeout,
                    )
                    state["rollout"] = rollout

                    postflight = run_preflight(
                        loaded["config"],
                        False,
                        accept_new_host_keys=args.accept_new_host_keys,
                        run_id_value=current_run,
                        phase="postflight",
                        timeout_seconds=args.preflight_timeout_seconds,
                    )
                    state["postflight"] = postflight
                    write_json(
                        root / "postflight.json",
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase": "postflight",
                            "results": postflight,
                            "ok": all(item["ok"] for item in postflight),
                            "admission": {"admitted": False},
                        },
                    )

                    results_path = root / "observe" / "rollout_results.json"
                    gate: Optional[Dict[str, Any]] = None
                    if results_path.is_file():
                        gate_args = [
                            "--mode",
                            "observe",
                            "--rollout-results",
                            str(results_path),
                            "--out",
                            str(root / "observe" / "release-gate.json"),
                        ]
                        for node_id in state["node_ids"]:
                            gate_args.extend(["--required-node", node_id])
                        gate = run_logged(
                            module_command(
                                "lms_agent_bench.fleet_gate_entrypoint", *gate_args
                            ),
                            root / "logs" / "gate.log",
                            timeout_seconds=args.gate_timeout_seconds,
                        )
                        state["gate"] = gate
                    else:
                        state["gate"] = {
                            "returncode": 1,
                            "timed_out": False,
                            "error": "rollout_results.json is missing",
                        }

                    rollout_ok = rollout["returncode"] == 0
                    postflight_ok = all(item["ok"] for item in postflight)
                    gate_ok = bool(gate and gate["returncode"] == 0)
                    state["success"] = rollout_ok and postflight_ok and gate_ok
                    if state["success"]:
                        outcome = 0
                    elif not rollout_ok:
                        state["failure_stage"] = "rollout"
                    elif not postflight_ok:
                        state["failure_stage"] = "postflight"
                    else:
                        state["failure_stage"] = "gate"
    except KeyboardInterrupt:
        state["failure_stage"] = "interrupted"
        state["error"] = "operator interrupted"
        outcome = 130
    except BaseException as exc:
        state["failure_stage"] = state.get("failure_stage") or "operator_exception"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        outcome = 1
    finally:
        state["finished_at_utc"] = utc_now()
        write_json(state_path, state)
        try:
            build_run_manifest(root, state)
        finally:
            release_lock(lock)
    return outcome


def verify_command(args: argparse.Namespace) -> int:
    try:
        report = verify_run_manifest(
            Path(args.run_dir).expanduser().resolve(),
            require_success=args.require_success,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fleet operator verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _add_controller_args(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", required=True)
    command.add_argument("--env-file", required=True)
    command.add_argument("--workspace", required=True)
    command.add_argument("--update-code", action="store_true")
    command.add_argument("--accept-new-host-keys", action="store_true")
    command.add_argument("--allow-insecure-env-file", action="store_true")
    command.add_argument(
        "--min-controller-free-bytes",
        type=int,
        default=DEFAULT_CONTROLLER_FREE_BYTES,
    )
    command.add_argument("--preflight-timeout-seconds", type=int, default=60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic fail-closed physical fleet workflows"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    _add_controller_args(preflight)
    observe = sub.add_parser("observe")
    _add_controller_args(observe)
    observe.add_argument("--run-id", default=None)
    observe.add_argument("--recover-stale-lock", action="store_true")
    observe.add_argument("--render-timeout-seconds", type=int, default=300)
    observe.add_argument("--rollout-timeout-seconds", type=int, default=0)
    observe.add_argument("--gate-timeout-seconds", type=int, default=900)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--require-success", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return preflight_command(args)
        if args.command == "observe":
            return observe_command(args)
        return verify_command(args)
    except KeyboardInterrupt:
        print("fleet operator interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"fleet operator failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
