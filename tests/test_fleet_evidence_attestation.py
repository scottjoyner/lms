from __future__ import annotations

import json
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
def test_verified_fleet_signer_fingerprint_cannot_be_rewritten_to_other_allowed_key(tmp_path):
    root = make_run(tmp_path)
    primary, allowed = make_key(tmp_path)
    other = tmp_path / "another-operator-key"
    generated = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(other)],
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stderr.decode(errors="replace")
    other.chmod(0o600)
    # Both keys are authorized for the same identity; a valid signature by
    # primary must never be attributed to 'other' via an unsigned JSON edit.
    with allowed.open("a", encoding="utf-8") as stream:
        stream.write("operator " + other.with_suffix(".pub").read_text(encoding="utf-8"))
    attestation.sign_run(root, primary, require_success=True)
    expected = attestation._key_fingerprint(primary)
    unrelated = attestation._key_fingerprint(other)
    assert expected != unrelated
    genuine = attestation.verify_attestation(root, allowed, "operator", require_success=True)
    assert genuine["signing_key_fingerprint"] == expected

    evidence_path = root / "operator-attestation.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["signing_key_fingerprint"] = unrelated
    core = {
        key: value for key, value in evidence.items()
        if key not in {"created_at_utc", "attestation_fingerprint"}
    }
    evidence["attestation_fingerprint"] = fleet_operator.canonical_hash(core)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="signing key fingerprint"):
        attestation.verify_attestation(root, allowed, "operator", require_success=True)

def test_fleet_attestation_uses_same_signature_bytes_for_verify_and_attribution(
    tmp_path, monkeypatch
):
    root = make_run(tmp_path)
    primary, allowed = make_key(tmp_path)
    other = tmp_path / "race-other-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(other)],
        capture_output=True, check=True,
    )
    other.chmod(0o600)
    with allowed.open("a", encoding="utf-8") as output:
        output.write("operator " + other.with_suffix(".pub").read_text(encoding="utf-8"))

    attestation.sign_run(root, primary, require_success=True)
    signed_manifest = root / "operator-manifest.json"
    signature = root / "operator-manifest.json.sig"
    alternate_input = tmp_path / "alternate-input"
    alternate_input.write_bytes(signed_manifest.read_bytes())
    subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-f", str(other),
         "-n", attestation.DEFAULT_NAMESPACE, str(alternate_input)],
        capture_output=True, check=True,
    )
    alternate_signature = Path(str(alternate_input) + ".sig").read_bytes()
    assert alternate_signature != signature.read_bytes()

    evidence_path = root / "operator-attestation.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["signing_key_fingerprint"] = attestation._key_fingerprint(other)
    core = {k: v for k, v in evidence.items()
            if k not in {"created_at_utc", "attestation_fingerprint"}}
    evidence["attestation_fingerprint"] = fleet_operator.canonical_hash(core)
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    real_run = subprocess.run
    swapped = []

    def swap_original_signature_after_verification(args, **kwargs):
        result = real_run(args, **kwargs)
        if "-Y" in args and "verify" in args and result.returncode == 0:
            signature.write_bytes(alternate_signature)
            swapped.append(True)
        return result

    monkeypatch.setattr(attestation.subprocess, "run", swap_original_signature_after_verification)
    with pytest.raises(ValueError, match="signing key fingerprint"):
        attestation.verify_attestation(root, allowed, "operator", require_success=True)
    assert swapped, "OpenSSH verification was not exercised"
