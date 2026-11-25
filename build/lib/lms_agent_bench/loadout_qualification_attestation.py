"""OpenSSH signing and verification for qualification-run manifests."""
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
from lms_agent_bench import loadout_qualification_operator as _qualification

SCHEMA_VERSION = "loadout_qualification_run_attestation.v1"
DEFAULT_NAMESPACE = "lms-loadout-qualification-run"


def sign_run(
    run_dir: Path,
    key: Path,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    require_success: bool = False,
) -> Mapping[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    verified = _qualification.verify_manifest(root, require_success=require_success)
    manifest = _ssh._require_regular(  # noqa: SLF001
        root / "qualification-run-manifest.json",
        "qualification-run manifest",
    )
    signing_key = _ssh._require_regular(key, "signing key", private=True)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    temporary = root / f".qualification-run-manifest.{uuid.uuid4().hex}.signing"
    temporary_signature = Path(str(temporary) + ".sig")
    signature = root / "qualification-run-manifest.json.sig"
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
            raise ValueError(
                "qualification manifest signing failed: " + process.stderr.strip()
            )
        if not temporary_signature.is_file() or temporary_signature.stat().st_size <= 0:
            raise ValueError("ssh-keygen produced no qualification signature")
        os.replace(temporary_signature, signature)
        _operator._fsync_dir(root)  # noqa: SLF001
    finally:
        temporary.unlink(missing_ok=True)
        temporary_signature.unlink(missing_ok=True)

    core = {
        "schema_version": SCHEMA_VERSION,
        "run_id": verified.get("run_id"),
        "run_success": verified.get("success") is True,
        "loadout_fingerprint": verified.get("loadout_fingerprint"),
        "qualification_fingerprint": verified.get("qualification_fingerprint"),
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
    _operator.write_json(root / "qualification-run-attestation.json", attestation)
    return attestation


def verify_attestation(
    run_dir: Path,
    allowed_signers: Path,
    identity: str,
    *,
    require_success: bool = False,
) -> Mapping[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    manifest_report = _qualification.verify_manifest(
        root, require_success=require_success
    )
    manifest = _ssh._require_regular(  # noqa: SLF001
        root / "qualification-run-manifest.json",
        "qualification-run manifest",
    )
    attestation_path = _ssh._require_regular(  # noqa: SLF001
        root / "qualification-run-attestation.json",
        "qualification-run attestation",
    )
    signers = _ssh._require_regular(allowed_signers, "allowed signers")  # noqa: SLF001
    identity = _ssh._identity(identity)  # noqa: SLF001
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported qualification attestation schema")
    core = {
        key: value
        for key, value in attestation.items()
        if key not in {"created_at_utc", "attestation_fingerprint"}
    }
    if attestation.get("attestation_fingerprint") != _operator.canonical_hash(core):
        raise ValueError("qualification attestation fingerprint mismatch")
    namespace = _ssh._namespace(str(attestation.get("namespace") or ""))  # noqa: SLF001
    if attestation.get("manifest") != manifest.name:
        raise ValueError("qualification attestation references a different manifest")
    signature_name = str(attestation.get("signature") or "")
    if Path(signature_name).name != signature_name:
        raise ValueError("qualification signature path is unsafe")
    signature = _ssh._require_regular(root / signature_name, "qualification signature")  # noqa: SLF001
    if _ssh.file_sha256(manifest) != attestation.get("manifest_sha256"):
        raise ValueError("attested qualification manifest SHA-256 mismatch")
    if _ssh.file_sha256(signature) != attestation.get("signature_sha256"):
        raise ValueError("attested qualification signature SHA-256 mismatch")
    if attestation.get("run_id") != manifest_report.get("run_id"):
        raise ValueError("qualification attestation run ID mismatch")
    if bool(attestation.get("run_success")) != bool(manifest_report.get("success")):
        raise ValueError("qualification attestation success mismatch")
    if attestation.get("loadout_fingerprint") != manifest_report.get(
        "loadout_fingerprint"
    ):
        raise ValueError("qualification attestation loadout mismatch")
    if attestation.get("qualification_fingerprint") != manifest_report.get(
        "qualification_fingerprint"
    ):
        raise ValueError("qualification attestation evidence mismatch")

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
        raise ValueError("qualification signature verification failed: " + stderr)
    return {
        "valid": True,
        "run_id": manifest_report.get("run_id"),
        "run_success": manifest_report.get("success") is True,
        "loadout_fingerprint": manifest_report.get("loadout_fingerprint"),
        "qualification_fingerprint": manifest_report.get(
            "qualification_fingerprint"
        ),
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
        prog="lms-loadout-qualification-attest",
        description="Sign and verify exact-loadout qualification-run evidence",
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
        print(f"qualification evidence attestation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
