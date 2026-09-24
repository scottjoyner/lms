"""Signed runtime identity witness binding qualified model bytes to one live process."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse, urlunparse

from lms_agent_bench import fleet_evidence_attestation as _ssh
from lms_agent_bench import fleet_operator as _operator
from lms_agent_bench import runtime_canary_attestation as _canary_attestation
from lms_agent_bench.model_loadout import validate_manifest

SCHEMA_VERSION = "fleet-runtime-identity-witness.v1"
DEFAULT_NAMESPACE = "lms-runtime-identity-witness"


def _normalize_runtime_url(value: str) -> str:
    raw = str(value or "").strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("runtime URL must be a valid http(s) endpoint")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("runtime URL may not contain credentials")
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _boot_id() -> str:
    value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("boot identity is unavailable")
    return value


def _process_start_ticks(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    try:
        tail = stat.rsplit(")", 1)[1].strip().split()
        return int(tail[19])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unable to parse process start time for pid {pid}") from exc


def _process_executable(pid: int) -> Path:
    path = Path(f"/proc/{pid}/exe")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"unable to resolve executable for pid {pid}") from exc
    if not resolved.is_file():
        raise ValueError(f"process executable is not a regular file: {resolved}")
    return resolved


def _process_references_model(pid: int, model_path: Path) -> str:
    target = str(model_path)
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        cmdline = ""
    if target in cmdline:
        return "cmdline"

    try:
        maps = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8", errors="replace")
    except OSError:
        maps = ""
    if target in maps:
        return "proc_maps"
    raise ValueError(
        "model path is not bound to the selected process through cmdline or /proc maps"
    )


def _regular_model(path: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError("model path may not be a symbolic link")
    resolved = value.resolve()
    if not resolved.is_file():
        raise ValueError(f"model path is not a regular file: {resolved}")
    return resolved


def process_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise ValueError("pid must be positive")
    executable = _process_executable(pid)
    return {
        "pid": pid,
        "boot_id": _boot_id(),
        "process_start_ticks": _process_start_ticks(pid),
        "executable_sha256": _ssh.file_sha256(executable),
        "executable_basename": executable.name,
    }


def build_witness(
    *,
    loadout_raw: Mapping[str, Any],
    canary_run_dir: Path,
    allowed_signers: Path,
    canary_identity: str,
    pid: int,
    runtime_url: str,
    runtime_kind: str,
    provider_model: str,
    model_path: Path,
    signing_key: Path,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, Any], bytes]:
    loadout = validate_manifest(loadout_raw, require_fingerprint=True)
    canary = _canary_attestation.verify_attestation(
        canary_run_dir,
        allowed_signers,
        canary_identity,
        require_success=True,
    )
    if canary.get("loadout_fingerprint") != loadout["loadout_fingerprint"]:
        raise ValueError("runtime canary belongs to a different exact loadout")

    model = _regular_model(model_path)
    actual_model_sha = _ssh.file_sha256(model)
    if actual_model_sha != loadout["model"]["content_sha256"]:
        raise ValueError("live model file does not match loadout model.content_sha256")

    process = process_identity(pid)
    binding_method = _process_references_model(pid, model)
    normalized_url = _normalize_runtime_url(runtime_url)
    runtime_kind = str(runtime_kind or "").strip().lower()
    provider_model = str(provider_model or "").strip()
    if not runtime_kind or not provider_model:
        raise ValueError("runtime kind and provider model are required")

    signing_key_path = _ssh._require_regular(signing_key, "witness signing key", private=True)  # noqa: SLF001
    core = {
        "schema_version": SCHEMA_VERSION,
        "node_id": str(loadout["node_id"]),
        "runtime_url": normalized_url,
        "runtime_kind": runtime_kind,
        "provider_model": provider_model,
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "model_id": str(loadout["model"]["id"]),
        "model_content_sha256": loadout["model"]["content_sha256"],
        "model_size_bytes": int(model.stat().st_size),
        "model_path": str(model),
        "model_process_binding": binding_method,
        "process": process,
        "canary": {
            "run_id": canary.get("run_id"),
            "canary_id": canary.get("canary_id"),
            "manifest_fingerprint": canary.get("manifest_fingerprint"),
            "attestation_fingerprint": canary.get("attestation_fingerprint"),
            "signing_key_fingerprint": canary.get("signing_key_fingerprint"),
            "rollback_succeeded": canary.get("rollback_succeeded") is True,
        },
        "witness_signing_key_fingerprint": _ssh._key_fingerprint(signing_key_path),  # noqa: SLF001
        "admission": {"admitted": False},
        "created_at_unix": int(time.time()),
    }
    witness = {
        **core,
        "witness_fingerprint": _operator.canonical_hash(core),
    }
    return witness, _canonical_bytes(witness)


def sign_witness_bytes(
    payload: bytes,
    *,
    signing_key: Path,
    namespace: str = DEFAULT_NAMESPACE,
) -> bytes:
    key = _ssh._require_regular(signing_key, "witness signing key", private=True)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    import tempfile

    with tempfile.TemporaryDirectory(prefix="runtime-witness-") as directory:
        path = Path(directory) / "runtime-identity-witness.json"
        path.write_bytes(payload)
        process = subprocess.run(
            [
                _ssh._ssh_keygen(),  # noqa: SLF001
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                namespace,
                str(path),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if process.returncode != 0:
            raise ValueError("runtime witness signing failed: " + process.stderr.strip())
        signature = Path(str(path) + ".sig")
        if not signature.is_file() or signature.stat().st_size <= 0:
            raise ValueError("ssh-keygen produced no runtime witness signature")
        return signature.read_bytes()


def verify_witness_bytes(
    payload: bytes,
    signature: bytes,
    *,
    allowed_signers: Path,
    identity: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    signers = _ssh._require_regular(allowed_signers, "allowed signers")  # noqa: SLF001
    identity = _ssh._identity(identity)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    try:
        witness = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime witness payload is not valid JSON") from exc
    if not isinstance(witness, dict) or witness.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported runtime identity witness schema")
    if witness.get("admission") != {"admitted": False}:
        raise ValueError("runtime identity witness must remain non-admitted")
    fingerprint = str(witness.get("witness_fingerprint") or "")
    core = {key: value for key, value in witness.items() if key != "witness_fingerprint"}
    if fingerprint != _operator.canonical_hash(core):
        raise ValueError("runtime identity witness fingerprint mismatch")
    if _canonical_bytes(witness) != payload:
        raise ValueError("runtime identity witness must use canonical JSON encoding")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="runtime-witness-verify-") as directory:
        signature_path = Path(directory) / "witness.sig"
        signature_path.write_bytes(signature)
        process = subprocess.run(
            [
                _ssh._ssh_keygen(),  # noqa: SLF001
                "-Y",
                "verify",
                "-f",
                str(signers),
                "-I",
                identity,
                "-n",
                namespace,
                "-s",
                str(signature_path),
            ],
            input=payload,
            capture_output=True,
            timeout=30,
            check=False,
        )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError("runtime witness signature verification failed: " + stderr)
    return witness


def observe_process_continuity(witness: Mapping[str, Any]) -> dict[str, Any]:
    process = witness.get("process")
    if not isinstance(process, Mapping):
        return {"valid": False, "reason": "witness_process_missing", "checked_at": int(time.time())}
    try:
        pid = int(process.get("pid") or 0)
        current = process_identity(pid)
    except (OSError, ValueError):
        return {"valid": False, "reason": "process_not_observable", "checked_at": int(time.time())}
    expected = {
        "pid": pid,
        "boot_id": str(process.get("boot_id") or ""),
        "process_start_ticks": int(process.get("process_start_ticks") or 0),
        "executable_sha256": str(process.get("executable_sha256") or ""),
        "executable_basename": str(process.get("executable_basename") or ""),
    }
    valid = current == expected
    return {
        "valid": valid,
        "reason": "match" if valid else "process_identity_changed",
        "checked_at": int(time.time()),
        "pid": pid,
        "boot_id": current.get("boot_id"),
        "process_start_ticks": current.get("process_start_ticks"),
        "executable_sha256": current.get("executable_sha256"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lms-runtime-witness")
    parser.add_argument("--loadout", type=Path, required=True)
    parser.add_argument("--canary-run-dir", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    parser.add_argument("--canary-identity", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument("--runtime-kind", required=True)
    parser.add_argument("--provider-model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        loadout_raw = json.loads(args.loadout.read_text(encoding="utf-8"))
        if not isinstance(loadout_raw, dict):
            raise ValueError("loadout must be a JSON object")
        witness, payload = build_witness(
            loadout_raw=loadout_raw,
            canary_run_dir=args.canary_run_dir,
            allowed_signers=args.allowed_signers,
            canary_identity=args.canary_identity,
            pid=args.pid,
            runtime_url=args.runtime_url,
            runtime_kind=args.runtime_kind,
            provider_model=args.provider_model,
            model_path=args.model_path,
            signing_key=args.signing_key,
            namespace=args.namespace,
        )
        signature = sign_witness_bytes(
            payload,
            signing_key=args.signing_key,
            namespace=args.namespace,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
        Path(str(args.out) + ".sig").write_bytes(signature)
        print(json.dumps(witness, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"runtime identity witness failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
