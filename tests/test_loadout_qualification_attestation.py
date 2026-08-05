from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lms_agent_bench import fleet_operator
from lms_agent_bench import loadout_qualification_attestation as attestation
from lms_agent_bench import loadout_qualification_operator as qualification


pytestmark = pytest.mark.skipif(
    shutil.which("ssh-keygen") is None,
    reason="OpenSSH ssh-keygen is unavailable",
)


def make_run(tmp_path: Path, *, success: bool = True) -> Path:
    root = tmp_path / "qualification-run"
    root.mkdir()
    fleet_operator.write_json(
        root / "qualification-state.json",
        {
            "schema_version": qualification.SCHEMA_VERSION,
            "run_id": "qualification-run",
            "success": success,
            "admission": {"admitted": False},
        },
    )
    qualification.build_manifest(
        root,
        {
            "run_id": "qualification-run",
            "success": success,
            "identity": {
                "loadout_fingerprint": "sha256:" + "1" * 64,
            },
            "sources": {},
            "inputs": {},
            "qualification_fingerprint": "sha256:" + "2" * 64,
        },
    )
    return root


def make_key(tmp_path: Path):
    key = tmp_path / "qualification-key"
    process = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    key.chmod(0o600)
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        "qualification-operator "
        + key.with_suffix(".pub").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return key, allowed


def test_sign_and_verify_qualification_manifest(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)
    signed = attestation.sign_run(root, key, require_success=True)
    verified = attestation.verify_attestation(
        root,
        allowed,
        "qualification-operator",
        require_success=True,
    )
    assert signed["run_success"] is True
    assert signed["loadout_fingerprint"] == "sha256:" + "1" * 64
    assert signed["qualification_fingerprint"] == "sha256:" + "2" * 64
    assert signed["admission"]["admitted"] is False
    assert verified["valid"] is True
    assert verified["identity"] == "qualification-operator"
    assert (root / "qualification-run-manifest.json.sig").is_file()
    assert (root / "qualification-run-attestation.json").is_file()


def test_wrong_signer_identity_is_rejected(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)
    attestation.sign_run(root, key)
    with pytest.raises(ValueError, match="verification failed"):
        attestation.verify_attestation(root, allowed, "wrong-operator")


def test_manifest_tampering_is_rejected_before_signature_verification(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)
    attestation.sign_run(root, key)
    manifest = root / "qualification-run-manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"tamper")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        attestation.verify_attestation(root, allowed, "qualification-operator")


def test_signature_tampering_is_rejected(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)
    attestation.sign_run(root, key)
    signature = root / "qualification-run-manifest.json.sig"
    signature.write_bytes(signature.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="signature SHA-256 mismatch"):
        attestation.verify_attestation(root, allowed, "qualification-operator")


def test_require_success_rejects_failed_qualification(tmp_path):
    root = make_run(tmp_path, success=False)
    key, _allowed = make_key(tmp_path)
    with pytest.raises(ValueError, match="did not complete successfully"):
        attestation.sign_run(root, key, require_success=True)
