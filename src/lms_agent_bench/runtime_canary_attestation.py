"""OpenSSH signing and verification for runtime canary manifests."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from lms_agent_bench import fleet_evidence_attestation as _ssh
from lms_agent_bench import fleet_operator as _operator
from lms_agent_bench import runtime_canary as _canary

SCHEMA_VERSION = "runtime_canary_attestation.v1"
DEFAULT_NAMESPACE = "lms-runtime-canary"


def sign_run(
    run_dir: Path,
    key: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    require_success: bool = False,
) -> Mapping[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    verified = _canary.verify_manifest(root, require_success=require_success)
    manifest = _ssh._require_regular(root / "runtime-canary-manifest.json", "runtime canary manifest")  # noqa: SLF001
    signing_key = _ssh._require_regular(key, "signing key", private=True)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    temporary = root / f".runtime-canary-manifest.{uuid.uuid4().hex}.signing"
    temporary_signature = Path(str(temporary) + ".sig")
    signature = root / "runtime-canary-manifest.json.sig"
    try:
        temporary.write_bytes(manifest.read_bytes())
        process = subprocess.run(
            [
                _ssh._ssh_keygen(),  # noqa: SLF001
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
            raise ValueError("runtime canary manifest signing failed: " + process.stderr.strip())
        if not temporary_signature.is_file() or temporary_signature.stat().st_size <= 0:
            raise ValueError("ssh-keygen produced no runtime canary signature")
        os.replace(temporary_signature, signature)
        _operator._fsync_dir(root)  # noqa: SLF001
    finally:
        temporary.unlink(missing_ok=True)
        temporary_signature.unlink(missing_ok=True)

    core = {
        "schema_version": SCHEMA_VERSION,
        "run_id": verified["run_id"],
        "canary_id": verified["canary_id"],
        "run_success": verified["success"],
        "rollback_succeeded": verified["rollback_succeeded"],
        "loadout_fingerprint": verified["loadout_fingerprint"],
        "plan_fingerprint": verified["plan_fingerprint"],
        "manifest": manifest.name,
        "manifest_sha256": _ssh.file_sha256(manifest),
        "signature": signature.name,
        "signature_sha256": _ssh.file_sha256(signature),
        "namespace": namespace,
        "signing_key_fingerprint": _ssh._key_fingerprint(signing_key),  # noqa: SLF001
        "admission": {"admitted": False},
    }
    attestation = {
        **core,
        "created_at_utc": _operator.utc_now(),
        "attestation_fingerprint": _operator.canonical_hash(core),
    }
    _operator.write_json(root / "runtime-canary-attestation.json", attestation)
    return attestation


def verify_attestation(
    run_dir: Path,
    allowed_signers: Path,
    identity: str,
    *,
    require_success: bool = False,
) -> Mapping[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    report = _canary.verify_manifest(root, require_success=require_success)
    manifest = _ssh._require_regular(root / "runtime-canary-manifest.json", "runtime canary manifest")  # noqa: SLF001
    attestation_path = _ssh._require_regular(root / "runtime-canary-attestation.json", "runtime canary attestation")  # noqa: SLF001
    signers = _ssh._require_regular(allowed_signers, "allowed signers")  # noqa: SLF001
    identity = _ssh._identity(identity)  # noqa: SLF001
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported runtime canary attestation schema")
    core = {
        key: value
        for key, value in attestation.items()
        if key not in {"created_at_utc", "attestation_fingerprint"}
    }
    if attestation.get("attestation_fingerprint") != _operator.canonical_hash(core):
        raise ValueError("runtime canary attestation fingerprint mismatch")
    namespace = _ssh._namespace(str(attestation.get("namespace") or ""))  # noqa: SLF001
    if attestation.get("manifest") != manifest.name:
        raise ValueError("runtime canary attestation references a different manifest")
    signature_name = str(attestation.get("signature") or "")
    if Path(signature_name).name != signature_name:
        raise ValueError("runtime canary signature path is unsafe")
    signature = _ssh._require_regular(root / signature_name, "runtime canary signature")  # noqa: SLF001
    if _ssh.file_sha256(manifest) != attestation.get("manifest_sha256"):
        raise ValueError("attested runtime canary manifest SHA-256 mismatch")
    if _ssh.file_sha256(signature) != attestation.get("signature_sha256"):
        raise ValueError("attested runtime canary signature SHA-256 mismatch")
    for key, report_key in (
        ("run_id", "run_id"),
        ("canary_id", "canary_id"),
        ("loadout_fingerprint", "loadout_fingerprint"),
        ("plan_fingerprint", "plan_fingerprint"),
    ):
        if attestation.get(key) != report.get(report_key):
            raise ValueError(f"runtime canary attestation {key} mismatch")
    if bool(attestation.get("run_success")) != bool(report.get("success")):
        raise ValueError("runtime canary attestation success mismatch")
    if bool(attestation.get("rollback_succeeded")) != bool(report.get("rollback_succeeded")):
        raise ValueError("runtime canary attestation rollback mismatch")
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
            str(signature),
        ],
        input=manifest.read_bytes(),
        capture_output=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError("runtime canary signature verification failed: " + stderr)
    return {
        "valid": True,
        **report,
        "identity": identity,
        "namespace": namespace,
        "signing_key_fingerprint": attestation.get("signing_key_fingerprint"),
        "attestation_fingerprint": attestation.get("attestation_fingerprint"),
        "admission": {"admitted": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms-runtime-canary-attest",
        description="Sign and verify runtime canary and rollback evidence",
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
            report = sign_run(
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
        print(f"runtime canary attestation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
