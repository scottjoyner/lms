"""Installed fleet operator boundary with input, retry, and path hardening."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench import fleet_operator as _base

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TRANSIENT_RETURN_CODES = {124, 255}
_PATCHED = False


def validate_run_id(value: str) -> str:
    raw = str(value or "")
    run = raw.strip()
    if raw != run or not _RUN_ID_RE.fullmatch(run) or ".." in run:
        raise ValueError(
            "run_id must be 1-128 safe characters without path separators, "
            "surrounding whitespace, or '..'"
        )
    return run


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    invalid = parsed < 0 if allow_zero else parsed <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")


def validate_args(args: Any) -> None:
    if args.command in {"preflight", "observe"}:
        _positive(args.min_controller_free_bytes, "min_controller_free_bytes")
        _positive(args.preflight_timeout_seconds, "preflight_timeout_seconds")
    if args.command == "observe":
        if args.run_id:
            validate_run_id(args.run_id)
        _positive(args.render_timeout_seconds, "render_timeout_seconds")
        _positive(
            args.rollout_timeout_seconds,
            "rollout_timeout_seconds",
            allow_zero=True,
        )
        _positive(args.gate_timeout_seconds, "gate_timeout_seconds")


def _hardened_controller_readiness(
    original: Any,
    config_path: str,
    env_file: str,
    workspace: Path,
    *,
    minimum_free_bytes: int,
    allow_insecure_env_file: bool,
) -> Dict[str, Any]:
    report = original(
        config_path,
        env_file,
        workspace,
        minimum_free_bytes=minimum_free_bytes,
        allow_insecure_env_file=allow_insecure_env_file,
    )
    errors = list(report.get("errors") or [])
    warnings = list(report.get("warnings") or [])
    for name, raw in (("config", config_path), ("env_file", env_file)):
        path = Path(raw).expanduser()
        if path.is_symlink():
            errors.append(f"{name} may not be a symbolic link: {path}")
        if path.is_file() and os.name == "posix" and path.stat().st_uid != os.getuid():
            errors.append(f"{name} is not owned by the current controller user: {path}")
    workspace_path = Path(workspace).expanduser()
    if workspace_path.is_symlink():
        errors.append(f"workspace may not be a symbolic link: {workspace_path}")
    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = not errors
    return report


def _sanitize_preflight_script(original: Any, *args: Any, **kwargs: Any) -> str:
    script = original(*args, **kwargs)
    raw = '"origin": git("remote", "get-url", "origin"),'
    safe = (
        '"origin_fingerprint": "sha256:" + '
        '__import__("hashlib").sha256('
        'git("remote", "get-url", "origin").encode("utf-8")'
        ').hexdigest(),'
    )
    if raw not in script:
        raise RuntimeError("preflight origin field changed unexpectedly")
    return script.replace(raw, safe, 1)


def _retry_preflight_node(
    original: Any,
    node: Mapping[str, Any],
    update_code: bool,
    **kwargs: Any,
) -> Dict[str, Any]:
    maximum = int(node.get("preflight_attempts") or 2)
    backoff = float(node.get("preflight_retry_backoff_seconds") or 2.0)
    if maximum <= 0:
        maximum = 1
    if backoff < 0:
        backoff = 0.0
    attempts: List[Dict[str, Any]] = []
    final: Dict[str, Any] = {}
    for attempt in range(1, maximum + 1):
        result = dict(original(node, update_code, **kwargs))
        attempts.append(
            {
                "attempt": attempt,
                "returncode": result.get("returncode"),
                "timed_out": result.get("timed_out") is True,
                "ok": result.get("ok") is True,
                "started_at_utc": result.get("started_at_utc"),
                "finished_at_utc": result.get("finished_at_utc"),
                "duration_seconds": result.get("duration_seconds"),
                "errors": result.get("errors"),
                "stderr": result.get("stderr"),
            }
        )
        final = result
        if result.get("ok") is True:
            break
        try:
            returncode = int(result.get("returncode"))
        except (TypeError, ValueError):
            returncode = -1
        transient = result.get("timed_out") is True or returncode in _TRANSIENT_RETURN_CODES
        if not transient or attempt >= maximum:
            break
        if backoff:
            time.sleep(backoff * attempt)
    final["attempts"] = attempts
    final["attempt_count"] = len(attempts)
    final["maximum_attempts"] = maximum
    return final


def _harden_verify_run_manifest(
    original: Any, root: Path, *, require_success: bool = False
) -> Dict[str, Any]:
    resolved_root = Path(root).expanduser().resolve()
    manifest_path = resolved_root / "operator-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen_control: set[str] = set()
    for entry in manifest.get("control_files", []):
        relative = str(entry.get("path") or "")
        if relative in seen_control:
            raise ValueError(f"duplicate operator control-file entry: {relative}")
        seen_control.add(relative)
    seen_archives: set[str] = set()
    for entry in manifest.get("archives", []):
        raw = str(entry.get("path") or "")
        archive = Path(raw).expanduser().resolve()
        try:
            archive.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"collected archive escapes the operator run directory: {archive}"
            ) from exc
        key = str(archive)
        if key in seen_archives:
            raise ValueError(f"duplicate collected archive entry: {archive}")
        seen_archives.add(key)
    return original(resolved_root, require_success=require_success)


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    original_readiness = _base.controller_readiness
    original_script = _base.preflight_script
    original_node = _base.preflight_node
    original_verify = _base.verify_run_manifest
    _base.controller_readiness = lambda *args, **kwargs: _hardened_controller_readiness(
        original_readiness, *args, **kwargs
    )
    _base.preflight_script = lambda *args, **kwargs: _sanitize_preflight_script(
        original_script, *args, **kwargs
    )
    _base.preflight_node = lambda node, update_code, **kwargs: _retry_preflight_node(
        original_node, node, update_code, **kwargs
    )
    _base.verify_run_manifest = lambda root, require_success=False: (
        _harden_verify_run_manifest(
            original_verify, root, require_success=require_success
        )
    )
    _PATCHED = True


def _observe_with_preacquired_lock(args: Any) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    current_run = validate_run_id(args.run_id or _base.run_id())
    args.run_id = current_run
    config = Path(args.config).expanduser().resolve()
    config_sha = _base.file_sha256(config) if config.is_file() else None
    lock, recovered = _base.acquire_lock(
        workspace,
        current_run=current_run,
        config_sha256=config_sha,
        recover_stale=args.recover_stale_lock,
    )
    original_acquire = _base.acquire_lock

    def already_acquired(*_args: Any, **_kwargs: Any):
        return lock, recovered

    _base.acquire_lock = already_acquired
    try:
        result = _base.observe_command(args)
    finally:
        _base.acquire_lock = original_acquire
        if lock.exists():
            _base.release_lock(lock)
    root = workspace / current_run
    if result != 0 and root.is_dir() and not any(root.iterdir()):
        root.rmdir()
    return result


def build_parser():
    return _base.build_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    apply_patches()
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        validate_args(args)
        if args.command == "preflight":
            return _base.preflight_command(args)
        if args.command == "observe":
            return _observe_with_preacquired_lock(args)
        return _base.verify_command(args)
    except KeyboardInterrupt:
        print("fleet operator interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"fleet operator failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
