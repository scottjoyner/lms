from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lms_agent_bench import fleet_evidence_attestation as attestation
from lms_agent_bench import fleet_operator


pytestmark = pytest.mark.skipif(
    shutil.which("ssh-keygen") is None,
    reason="OpenSSH ssh-keygen is unavailable",
)


def make_run(tmp_path: Path, *, success: bool = True) -> Path:
    root = tmp_path / "run-a"
    root.mkdir()
    fleet_operator.write_json(
        root / "operator-state.json",
        {
            "schema_version": "fleet_operator_run.v2",
            "run_id": "run-a",
            "success": success,
            "admission": {"admitted": False},
        },
    )
    fleet_operator.build_run_manifest(
        root,
        {
            "run_id": "run-a",
            "success": success,
            "config_sha256": "sha256:" + "1" * 64,
            "env_file_sha256": "sha256:" + "2" * 64,
        },
    )
    return root


def make_key(tmp_path: Path):
    key = tmp_path / "operator-key"
    process = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    key.chmod(0o600)
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        "operator " + key.with_suffix(".pub").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return key, allowed


def test_sign_and_verify_operator_manifest(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)

    signed = attestation.sign_run(root, key, require_success=True)
    verified = attestation.verify_attestation(
        root,
        allowed,
        "operator",
        require_success=True,
    )

    assert signed["schema_version"] == attestation.SCHEMA_VERSION
    assert signed["run_success"] is True
    assert signed["admission"]["admitted"] is False
    assert verified["valid"] is True
    assert verified["run_id"] == "run-a"
    assert verified["identity"] == "operator"
    assert (root / "operator-manifest.json.sig").is_file()
    assert (root / "operator-attestation.json").is_file()


def test_wrong_identity_is_rejected(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)
    attestation.sign_run(root, key)
    with pytest.raises(ValueError, match="verification failed"):
        attestation.verify_attestation(root, allowed, "different-operator")


def test_signature_tampering_is_rejected_before_openssh_verify(tmp_path):
    root = make_run(tmp_path)
    key, allowed = make_key(tmp_path)
    attestation.sign_run(root, key)
    signature = root / "operator-manifest.json.sig"
    signature.write_bytes(signature.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="signature SHA-256 mismatch"):
        attestation.verify_attestation(root, allowed, "operator")


def test_insecure_private_key_permissions_are_rejected(tmp_path):
    root = make_run(tmp_path)
    key, _allowed = make_key(tmp_path)
    key.chmod(0o644)
    if Path("/").stat().st_mode:  # Keep the POSIX-only assertion explicit.
        with pytest.raises(ValueError, match="permissions"):
            attestation.sign_run(root, key)


def test_require_success_rejects_failed_run(tmp_path):
    root = make_run(tmp_path, success=False)
    key, _allowed = make_key(tmp_path)
    with pytest.raises(ValueError, match="did not complete successfully"):
        attestation.sign_run(root, key, require_success=True)
