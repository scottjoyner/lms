"""OpenSSH detached signing and verification for fleet operator evidence."""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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


def _verified_sshsig_key_fingerprint(signature: Path) -> str:
    """Identify the public key carried by an already VERIFIED SSHSIG.

    The embedded public-key blob is OpenSSH wire encoding. Its SHA-256 is
    exactly the fingerprint ssh-keygen -lf -E sha256 reports for that key.
    Call only after ssh-keygen -Y verify succeeds under allowed-signers policy.
    """
    armored = signature if isinstance(signature, bytes) else signature.read_bytes()
    if len(armored) > 256 * 1024:
        raise ValueError("OpenSSH signature is too large")
    lines = armored.splitlines()
    if (
        len(lines) < 3
        or lines[0] != b"-----BEGIN SSH SIGNATURE-----"
        or lines[-1] != b"-----END SSH SIGNATURE-----"
    ):
        raise ValueError("invalid OpenSSH signature armor")
    try:
        encoded = base64.b64decode(b"".join(lines[1:-1]), validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid OpenSSH signature encoding") from exc
    if not encoded.startswith(b"SSHSIG") or len(encoded) < 14:
        raise ValueError("invalid OpenSSH signature envelope")
    if int.from_bytes(encoded[6:10], "big") != 1:
        raise ValueError("unsupported OpenSSH signature version")
    size = int.from_bytes(encoded[10:14], "big")
    if not 0 < size <= 65536 or len(encoded) < 14 + size:
        raise ValueError("invalid OpenSSH signature public key")
    pub = encoded[14:14 + size]
    digest = base64.b64encode(hashlib.sha256(pub).digest()).decode("ascii")
    return "SHA256:" + digest.rstrip("=")


def _verify_pinned_attestation(
    manifest: Path,
    signature: Path,
    recorded_manifest_sha256: str,
    recorded_signature_sha256: str,
    allowed_signers: Path,
    identity: str,
    namespace: str,
    *,
    label: str,
) -> tuple[str, bytes]:
    """Verify one pinned manifest/signature byte pair, then identify its signer.

    Hashing the original files, verifying a *different* read and parsing the
    original signature again permits a same-owner file-replacement race.
    The OpenSSH subprocess sees a private snapshot of the captured signature,
    and the public-key fingerprint comes from those same captured bytes.
    """
    if signature.stat().st_size > 256 * 1024:
        raise ValueError(f"{label} signature exceeds size limit")
    manifest_bytes = manifest.read_bytes()
    signature_bytes = signature.read_bytes()
    if len(signature_bytes) > 256 * 1024:
        raise ValueError(f"{label} signature exceeds size limit")
    if "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != recorded_manifest_sha256:
        raise ValueError(f"attested {label} manifest SHA-256 mismatch")
    if "sha256:" + hashlib.sha256(signature_bytes).hexdigest() != recorded_signature_sha256:
        raise ValueError(f"attested {label} signature SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="lms-attest-verify-") as directory:
        pinned_signature = Path(directory) / "signature"
        pinned_signature.write_bytes(signature_bytes)
        process = subprocess.run(
            [
                _ssh_keygen(), "-Y", "verify", "-f", str(allowed_signers),
                "-I", identity, "-n", namespace, "-s", str(pinned_signature),
            ],
            input=manifest_bytes,
            capture_output=True,
            timeout=30,
            check=False,
        )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"{label} signature verification failed: " + stderr)
    return _verified_sshsig_key_fingerprint(signature_bytes), manifest_bytes


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
    if attestation.get("run_id") != manifest_report.get("run_id"):
        raise ValueError("attestation run ID does not match operator manifest")
    if bool(attestation.get("run_success")) != bool(manifest_report.get("success")):
        raise ValueError("attestation success state does not match operator manifest")

    verified_fingerprint, signed_manifest_bytes = _verify_pinned_attestation(
        manifest,
        signature,
        attestation.get("manifest_sha256"),
        attestation.get("signature_sha256"),
        signers,
        identity,
        namespace,
        label="operator",
    )
    # A valid signature does NOT validate references inside its manifest.
    # Re-check the identical bytes OpenSSH authenticated against real files.
    signed_report = _operator.verify_run_manifest(
        root, require_success=require_success, manifest_bytes=signed_manifest_bytes
    )
    if signed_report != manifest_report:
        raise ValueError("operator verified manifest differs from signed manifest")
    if attestation.get("signing_key_fingerprint") != verified_fingerprint:
        raise ValueError("operator signing key fingerprint does not match verified signature")
    return {
        "valid": True,
        "run_id": manifest_report.get("run_id"),
        "run_success": manifest_report.get("success") is True,
        "identity": identity,
        "namespace": namespace,
        "signing_key_fingerprint": verified_fingerprint,
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
