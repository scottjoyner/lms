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
def test_signed_manifest_must_be_the_same_content_that_was_structurally_validated(
    tmp_path, monkeypatch,
):
    root = make_run(tmp_path)
    signing_key, allowed = make_key(tmp_path)
    attestation.sign_run(root, signing_key, require_success=True)
    manifest_path = root / "operator-manifest.json"
    signature_path = root / "operator-manifest.json.sig"
    attestation_path = root / "operator-attestation.json"

    forged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forged_manifest["control_files"].append({
        "path": "missing-proof.json", "size_bytes": 9,
        "sha256": "sha256:" + "0" * 64,
    })
    core = {
        key: value for key, value in forged_manifest.items()
        if key not in {"created_at_utc", "operator_manifest_fingerprint"}
    }
    forged_manifest["operator_manifest_fingerprint"] = fleet_operator.canonical_hash(core)
    candidate = tmp_path / "invalid-but-signed-manifest"
    candidate.write_text(json.dumps(forged_manifest), encoding="utf-8")
    subprocess.run(
        ["ssh-keygen", "-Y", "sign", "-f", str(signing_key),
         "-n", attestation.DEFAULT_NAMESPACE, str(candidate)],
        capture_output=True, check=True,
    )
    forged_manifest_bytes = candidate.read_bytes()
    forged_signature_bytes = Path(str(candidate) + ".sig").read_bytes()
    metadata = json.loads(attestation_path.read_text(encoding="utf-8"))
    metadata["manifest_sha256"] = "sha256:" + __import__("hashlib").sha256(
        forged_manifest_bytes
    ).hexdigest()
    metadata["signature_sha256"] = "sha256:" + __import__("hashlib").sha256(
        forged_signature_bytes
    ).hexdigest()
    attestation_core = {
        key: value for key, value in metadata.items()
        if key not in {"created_at_utc", "attestation_fingerprint"}
    }
    metadata["attestation_fingerprint"] = fleet_operator.canonical_hash(attestation_core)

    actual_validate = fleet_operator.verify_run_manifest
    changes = []

    def replace_validated_manifest(*args, **kwargs):
        report = actual_validate(*args, **kwargs)
        manifest_path.write_bytes(forged_manifest_bytes)
        signature_path.write_bytes(forged_signature_bytes)
        attestation_path.write_text(json.dumps(metadata), encoding="utf-8")
        changes.append(True)
        return report

    monkeypatch.setattr(attestation._operator, "verify_run_manifest", replace_validated_manifest)
    with pytest.raises(ValueError, match="control file is missing"):
        attestation.verify_attestation(root, allowed, "operator", require_success=True)
    assert changes, "the structurally valid manifest was not checked first"
@pytest.mark.parametrize("symlink_kind", ["file", "directory"])
def test_signed_fleet_control_files_must_reside_inside_run(tmp_path, symlink_kind):
    # All test keys and files are disposable; no physical fleet activity.
    root = make_run(tmp_path)
    outside = tmp_path / "outside-evidence"
    outside.mkdir()
    real_proof = outside / "proof.json"
    real_proof.write_text('{"untrusted": true}\n', encoding="utf-8")
    if symlink_kind == "file":
        name = "claimed-proof.json"
        (root / name).symlink_to(real_proof)
    else:
        name = "claimed-evidence/proof.json"
        (root / "claimed-evidence").symlink_to(outside, target_is_directory=True)
    manifest_path = root / "operator-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["control_files"] = [{
        "path": name,
        "size_bytes": real_proof.stat().st_size,
        "sha256": fleet_operator.file_sha256(real_proof),
    }]
    core = {
        key: value for key, value in manifest.items()
        if key not in {"created_at_utc", "operator_manifest_fingerprint"}
    }
    manifest["operator_manifest_fingerprint"] = fleet_operator.canonical_hash(core)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    key, allowed = make_key(tmp_path)
    with pytest.raises(ValueError, match="control file.*(symlink|escapes)"):
        attestation.sign_run(root, key, require_success=True)
    # Even direct manifest verification must reject the external reference.
    with pytest.raises(ValueError, match="control file.*(symlink|escapes)"):
        fleet_operator.verify_run_manifest(root, require_success=True)
