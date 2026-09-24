from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from lms_agent_bench import runtime_identity_witness as witness
from lms_agent_bench.model_loadout import validate_manifest


def _example_loadout(model_path: Path) -> dict:
    payload = json.loads(
        Path("examples/model-loadouts.v1.example.json").read_text(encoding="utf-8")
    )
    raw = payload["base_manifests"][0]
    raw["node_id"] = "destroyer"
    raw["candidate_id"] = "k2-runtime"
    raw["model"]["id"] = "k2-36b"
    raw["model"]["content_sha256"] = witness._ssh.file_sha256(model_path)
    raw["model"]["size_bytes"] = model_path.stat().st_size
    raw.pop("loadout_fingerprint", None)
    raw.pop("derived", None)
    raw.pop("admission", None)
    return validate_manifest(raw)


def test_build_witness_binds_canary_model_hash_and_process(monkeypatch, tmp_path):
    model_path = tmp_path / "k2.gguf"
    model_path.write_bytes(b"qualified-model-bytes")
    loadout = _example_loadout(model_path)
    key = tmp_path / "operator-key"
    key.write_text("placeholder", encoding="utf-8")
    key.chmod(0o600)

    monkeypatch.setattr(
        witness._canary_attestation,
        "verify_attestation",
        lambda *_args, **_kwargs: {
            "valid": True,
            "run_id": "canary-run",
            "canary_id": "destroyer-k2",
            "loadout_fingerprint": loadout["loadout_fingerprint"],
            "manifest_fingerprint": "sha256:" + "1" * 64,
            "attestation_fingerprint": "sha256:" + "2" * 64,
            "signing_key_fingerprint": "SHA256:canary",
            "rollback_succeeded": True,
        },
    )
    monkeypatch.setattr(
        witness,
        "process_identity",
        lambda _pid: {
            "pid": 4242,
            "boot_id": "boot-id",
            "process_start_ticks": 12345,
            "executable_sha256": "sha256:" + "3" * 64,
            "executable_basename": "llama-server",
        },
    )
    monkeypatch.setattr(
        witness,
        "_process_references_model",
        lambda _pid, _path: "cmdline",
    )
    monkeypatch.setattr(
        witness._ssh,
        "_key_fingerprint",
        lambda _path: "SHA256:witness",
    )

    built, payload = witness.build_witness(
        loadout_raw=loadout,
        canary_run_dir=tmp_path / "canary",
        allowed_signers=tmp_path / "allowed_signers",
        canary_identity="runtime-canary-operator",
        pid=4242,
        runtime_url="http://localhost:1235/v1",
        runtime_kind="llama_cpp",
        provider_model="k2-36b",
        model_path=model_path,
        signing_key=key,
    )

    assert built["runtime_url"] == "http://localhost:1235"
    assert built["runtime_kind"] == "llama_cpp"
    assert built["provider_model"] == "k2-36b"
    assert built["model_content_sha256"] == loadout["model"]["content_sha256"]
    assert built["loadout_fingerprint"] == loadout["loadout_fingerprint"]
    assert built["model_process_binding"] == "cmdline"
    assert built["process"]["process_start_ticks"] == 12345
    assert built["canary"]["rollback_succeeded"] is True
    assert built["admission"]["admitted"] is False
    assert payload == witness._canonical_bytes(built)


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="OpenSSH unavailable")
def test_signed_witness_round_trip(tmp_path):
    key = tmp_path / "witness-key"
    generated = subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        capture_output=True,
        check=False,
    )
    assert generated.returncode == 0
    key.chmod(0o600)
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        "runtime-witness-operator "
        + key.with_suffix(".pub").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    core = {
        "schema_version": witness.SCHEMA_VERSION,
        "node_id": "destroyer",
        "runtime_url": "http://localhost:1235",
        "runtime_kind": "llama_cpp",
        "provider_model": "k2-36b",
        "loadout_fingerprint": "sha256:" + "1" * 64,
        "model_id": "k2-36b",
        "model_content_sha256": "sha256:" + "2" * 64,
        "model_size_bytes": 123,
        "model_path": "/models/k2.gguf",
        "model_process_binding": "cmdline",
        "process": {
            "pid": 42,
            "boot_id": "boot",
            "process_start_ticks": 99,
            "executable_sha256": "sha256:" + "3" * 64,
            "executable_basename": "llama-server",
        },
        "canary": {
            "run_id": "run",
            "canary_id": "canary",
            "manifest_fingerprint": "sha256:" + "4" * 64,
            "attestation_fingerprint": "sha256:" + "5" * 64,
            "signing_key_fingerprint": "SHA256:canary",
            "rollback_succeeded": True,
        },
        "witness_signing_key_fingerprint": "SHA256:witness",
        "admission": {"admitted": False},
        "created_at_unix": 100,
    }
    document = {
        **core,
        "witness_fingerprint": witness._operator.canonical_hash(core),
    }
    payload = witness._canonical_bytes(document)
    signature = witness.sign_witness_bytes(payload, signing_key=key)

    verified = witness.verify_witness_bytes(
        payload,
        signature,
        allowed_signers=allowed,
        identity="runtime-witness-operator",
    )

    assert verified == document


def test_live_process_continuity_uses_pid_start_and_boot_identity():
    current = witness.process_identity(os.getpid())
    document = {"process": current}

    observed = witness.observe_process_continuity(document)

    assert observed["valid"] is True
    assert observed["pid"] == os.getpid()
    assert observed["process_start_ticks"] == current["process_start_ticks"]
