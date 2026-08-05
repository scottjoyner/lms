"""Reliable remote execution and atomic artifact collection for fleet rollout."""
from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute_remote_reliably(
    entrypoint: Any,
    node: Mapping[str, Any],
    script: str,
    run_id: str,
    collect_dir: Path,
    ssh_options: Sequence[str],
    collect: bool,
) -> Dict[str, Any]:
    """Run once remotely and retry only the immutable artifact transfer.

    The remote workload is never retried automatically. Artifact collection is
    safe to retry because the remote shell has already completed packaging and
    every attempt writes a distinct partial file. Only a successful complete
    SCP is atomically promoted to the final archive path.
    """
    base = entrypoint._base  # noqa: SLF001
    started_at = base.utc_now_iso()
    remote_timeout = int(node.get("remote_timeout_seconds") or 7200)
    try:
        process = subprocess.run(
            base.ssh_command(str(node["ssh_target"]), ssh_options),
            input=script,
            text=True,
            capture_output=True,
            timeout=remote_timeout,
            check=False,
        )
        returncode = process.returncode
        stdout = process.stdout
        stderr = process.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = _timeout_text(exc.stdout)
        stderr = (
            _timeout_text(exc.stderr)
            + f"\nremote rollout timed out after {remote_timeout}s"
        ).strip()
        timed_out = True

    result: Dict[str, Any] = {
        "node_id": node["node_id"],
        "ssh_target": node["ssh_target"],
        "started_at_utc": started_at,
        "finished_at_utc": base.utc_now_iso(),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "remote_timeout_seconds": remote_timeout,
        "collected_artifact": None,
        "scp_attempts": [],
    }
    if not collect:
        return result

    remote_tar = base.remote_artifact_path(node, run_id) + ".tar.gz"
    collect_dir.mkdir(parents=True, exist_ok=True)
    local_tar = collect_dir / f"{base.safe_slug(str(node['node_id']))}.tar.gz"
    scp_timeout = int(node.get("scp_timeout_seconds") or 300)
    maximum_attempts = int(node.get("scp_attempts") or 3)
    backoff_seconds = float(node.get("scp_retry_backoff_seconds") or 2.0)
    if maximum_attempts <= 0:
        maximum_attempts = 1
    if backoff_seconds < 0:
        backoff_seconds = 0.0

    final_returncode = -1
    final_stdout = ""
    final_stderr = ""
    final_timed_out = False
    for attempt in range(1, maximum_attempts + 1):
        partial = local_tar.with_name(
            f".{local_tar.name}.attempt-{attempt}.partial"
        )
        partial.unlink(missing_ok=True)
        attempt_started = base.utc_now_iso()
        monotonic_started = time.monotonic()
        try:
            transfer = subprocess.run(
                base.scp_command(
                    str(node["ssh_target"]),
                    remote_tar,
                    partial,
                    ssh_options,
                ),
                text=True,
                capture_output=True,
                timeout=scp_timeout,
                check=False,
            )
            scp_returncode = transfer.returncode
            scp_stdout = transfer.stdout
            scp_stderr = transfer.stderr
            scp_timed_out = False
        except subprocess.TimeoutExpired as exc:
            scp_returncode = 124
            scp_stdout = _timeout_text(exc.stdout)
            scp_stderr = (
                _timeout_text(exc.stderr)
                + f"\nartifact collection timed out after {scp_timeout}s"
            ).strip()
            scp_timed_out = True

        attempt_record = {
            "attempt": attempt,
            "started_at_utc": attempt_started,
            "finished_at_utc": base.utc_now_iso(),
            "duration_seconds": time.monotonic() - monotonic_started,
            "returncode": scp_returncode,
            "timed_out": scp_timed_out,
            "stdout": scp_stdout,
            "stderr": scp_stderr,
            "partial_path": str(partial),
            "partial_size_bytes": partial.stat().st_size if partial.is_file() else 0,
        }
        result["scp_attempts"].append(attempt_record)
        final_returncode = scp_returncode
        final_stdout = scp_stdout
        final_stderr = scp_stderr
        final_timed_out = scp_timed_out

        if scp_returncode == 0 and partial.is_file() and partial.stat().st_size > 0:
            os.replace(partial, local_tar)
            _fsync_dir(local_tar.parent)
            result["collected_artifact"] = str(local_tar)
            result["collected_artifact_size_bytes"] = local_tar.stat().st_size
            result["collected_artifact_sha256"] = _hash_file(local_tar)
            break
        partial.unlink(missing_ok=True)
        if attempt < maximum_attempts and backoff_seconds:
            time.sleep(backoff_seconds * attempt)

    result["scp_returncode"] = final_returncode
    result["scp_stdout"] = final_stdout
    result["scp_stderr"] = final_stderr
    result["scp_timed_out"] = final_timed_out
    result["scp_timeout_seconds"] = scp_timeout
    result["scp_attempt_count"] = len(result["scp_attempts"])
    result["scp_max_attempts"] = maximum_attempts
    return result


def apply_transfer_hardening(entrypoint: Any) -> None:
    if getattr(entrypoint, "_lms_transfer_hardened", False):
        return

    def hardened_execute(
        node: Mapping[str, Any],
        script: str,
        run_id: str,
        collect_dir: Path,
        ssh_options: Sequence[str],
        collect: bool,
    ) -> Dict[str, Any]:
        return execute_remote_reliably(
            entrypoint,
            node,
            script,
            run_id,
            collect_dir,
            ssh_options,
            collect,
        )

    entrypoint.execute_remote = hardened_execute
    entrypoint._lms_transfer_hardened = True  # noqa: SLF001
