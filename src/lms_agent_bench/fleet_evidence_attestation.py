"""OpenSSH detached signing and verification for fleet operator evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from lms_agent_bench import fleet_operator as _operator
from lms_agent_bench import fleet_operator_entrypoint as _entrypoint

SCHEMA_VERSION = "fleet_operator_attestation.v1"
DEFAULT_NAMESPACE = "lms-fleet-operator"
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._-]{0,255}$")
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_regular(path: Path, label: str, *, private: bool = False) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError(f"{label} may not be a symbolic link")
    resolved = value.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    if os.name == "posix" and private:
        mode = resolved.stat().st_mode & 0o777
        if mode & 0o077:
            raise ValueError(
                f"{label} permissions are {mode:03o}; expected no group/world access"
            )
        if resolved.stat().st_uid != os.getuid():
            raise ValueError(f"{label} is not owned by the current user")
    return resolved


def _namespace(value: str) -> str:
    namespace = str(value or "")
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError("signature namespace contains unsafe characters")
    return namespace


def _identity(value: str) -> str:
    identity = str(value or "")
    if not _IDENTITY_RE.fullmatch(identity):
        raise ValueError("signer identity contains unsafe characters")
    return identity


def _ssh_keygen() -> str:
    command = shutil.which("ssh-keygen")
    if not command:
        raise ValueError("ssh-keygen is required for fleet evidence attestation")
    return command


def _key_fingerprint(key: Path) -> str:
    process = subprocess.run(
        [_ssh_keygen(), "-lf", str(key), "-E", "sha256"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        raise ValueError("unable to read signing-key fingerprint: " + process.stderr.strip())
    fields = process.stdout.strip().split()
    fingerprint = next((item for item in fields if item.startswith("SHA256:")), None)
    if not fingerprint:
        raise ValueError("ssh-keygen did not return a SHA-256 key fingerprint")
    return fingerprint


def sign_run(
    run_dir: Path,
    key: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    require_success: bool = False,
) -> Dict[str, Any]:
    _entrypoint.apply_patches()
    root = Path(run_dir).expanduser().resolve()
    verified = _operator.verify_run_manifest(root, require_success=require_success)
    manifest = _require_regular(root / "operator-manifest.json", "operator manifest")
    signing_key = _require_regular(key, "signing key", private=True)
    namespace = _namespace(namespace)
    temporary = root / f".operator-manifest.{uuid.uuid4().hex}.signing"
    temporary_signature = Path(str(temporary) + ".sig")
    signature = root / "operator-manifest.json.sig"
    try:
        temporary.write_bytes(manifest.read_bytes())
        process = subprocess.run(
            [
                _ssh_keygen(),
                "-Y",
                "sign",
                "-f",
                str(signing_key),
                "-n",
                namespace,
                str(temporary),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if process.returncode != 0:
            raise ValueError("operator manifest signing failed: " + process.stderr.strip())
        if not temporary_signature.is_file() or temporary_signature.stat().st_size <= 0:
            raise ValueError("ssh-keygen produced no detached signature")
        os.replace(temporary_signature, signature)
        _operator._fsync_dir(root)  # noqa: SLF001
    finally:
        temporary.unlink(missing_ok=True)
        temporary_signature.unlink(missing_ok=True)

    core = {
        "schema_version": SCHEMA_VERSION,
        "run_id": verified.get("run_id"),
        "run_success": verified.get("success") is True,
        "manifest": "operator-manifest.json",
        "manifest_sha256": file_sha256(manifest),
        "signature": signature.name,
        "signature_sha256": file_sha256(signature),
        "namespace": namespace,
        "signing_key_fingerprint": _key_fingerprint(signing_key),
        "admission": {"admitted": False},
    }
    attestation = {
        **core,
        "created_at_utc": _operator.utc_now(),
        "attestation_fingerprint": _operator.canonical_hash(core),
    }
    _operator.write_json(root / "operator-attestation.json", attestation)
    return attestation


def verify_attestation(
    run_dir: Path,
    allowed_signers: Path,
    identity: str,
    *,
    require_success: bool = False,
) -> Dict[str, Any]:
    _entrypoint.apply_patches()
    root = Path(run_dir).expanduser().resolve()
    manifest_report = _operator.verify_run_manifest(
        root, require_success=require_success
    )
    manifest = _require_regular(root / "operator-manifest.json", "operator manifest")
    attestation_path = _require_regular(
        root / "operator-attestation.json", "operator attestation"
    )
    signers = _require_regular(allowed_signers, "allowed signers")
    identity = _identity(identity)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported operator attestation schema")
    core = {
        key: value
        for key, value in attestation.items()
        if key not in {"created_at_utc", "attestation_fingerprint"}
    }
    if attestation.get("attestation_fingerprint") != _operator.canonical_hash(core):
        raise ValueError("operator attestation fingerprint mismatch")
    namespace = _namespace(str(attestation.get("namespace") or ""))
    if attestation.get("manifest") != manifest.name:
        raise ValueError("attestation references a different manifest")
    signature_name = str(attestation.get("signature") or "")
    if Path(signature_name).name != signature_name:
        raise ValueError("attestation signature path is unsafe")
    signature = _require_regular(root / signature_name, "operator signature")
    if file_sha256(manifest) != attestation.get("manifest_sha256"):
        raise ValueError("attested manifest SHA-256 mismatch")
    if file_sha256(signature) != attestation.get("signature_sha256"):
        raise ValueError("attested signature SHA-256 mismatch")
    if attestation.get("run_id") != manifest_report.get("run_id"):
        raise ValueError("attestation run ID does not match operator manifest")
    if bool(attestation.get("run_success")) != bool(manifest_report.get("success")):
        raise ValueError("attestation success state does not match operator manifest")

    process = subprocess.run(
        [
            _ssh_keygen(),
            "-Y",
            "verify",
            "-f",
            str(signers),
            "-I",
            identity,
            "-n",
            namespace,
            "-s",
            str(signature),
        ],
        input=manifest.read_bytes(),
        capture_output=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError("operator signature verification failed: " + stderr)
    return {
        "valid": True,
        "run_id": manifest_report.get("run_id"),
        "run_success": manifest_report.get("success") is True,
        "identity": identity,
        "namespace": namespace,
        "signing_key_fingerprint": attestation.get("signing_key_fingerprint"),
        "manifest_sha256": attestation.get("manifest_sha256"),
        "signature_sha256": attestation.get("signature_sha256"),
        "attestation_fingerprint": attestation.get("attestation_fingerprint"),
        "admission": {"admitted": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms-fleet-attest",
        description="Sign and verify fleet operator evidence with OpenSSH",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sign = commands.add_parser("sign")
    sign.add_argument("--run-dir", type=Path, required=True)
    sign.add_argument("--key", type=Path, required=True)
    sign.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    sign.add_argument("--require-success", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--allowed-signers", type=Path, required=True)
    verify.add_argument("--identity", required=True)
    verify.add_argument("--require-success", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "sign":
            report: Mapping[str, Any] = sign_run(
                args.run_dir,
                args.key,
                namespace=args.namespace,
                require_success=args.require_success,
            )
        else:
            report = verify_attestation(
                args.run_dir,
                args.allowed_signers,
                args.identity,
                require_success=args.require_success,
            )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"fleet evidence attestation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
