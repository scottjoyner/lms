"""Fail-closed offline wheel inventory tests (no network, install, or models)."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from lms_agent_bench import offline_wheelhouse as verifier


def make_wheel(directory: Path, name: str = "lms_agent_bench", version: str = "0.33.0", *, embedded_name: str | None = None) -> Path:
    wheel = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        folder = f"{name}-{version}.dist-info"
        archive.writestr(f"{folder}/METADATA", f"Metadata-Version: 2.1\nName: {embedded_name or name}\nVersion: {version}\n")
        archive.writestr(f"{folder}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
    wheel.chmod(0o600)
    return wheel


def make_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    wheels = tmp_path / "wheels"
    wheels.mkdir(mode=0o700)
    wheels.chmod(0o700)
    wheel = make_wheel(wheels)
    lock = tmp_path / "lock.txt"
    lock.write_text("lms-agent-bench==0.33.0 --hash=sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest() + "\n")
    lock.chmod(0o600)
    return wheels, lock, wheel


def test_valid_pinned_wheel_has_explicit_non_authorization(tmp_path):
    wheels, lock, _ = make_bundle(tmp_path)
    result = verifier.verify(wheels, lock, expected_version="0.33.0")
    assert result["valid"] and result["wheel_count"] == 1
    assert result["production_authorized"] is False
    assert result["independent_signatures_verified"] is False
    assert result["lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()


def test_modified_wheel_content_is_denied(tmp_path):
    wheels, lock, wheel = make_bundle(tmp_path)
    with wheel.open("ab") as handle:
        handle.write(b"post-lock tampering")
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier.verify(wheels, lock)


def test_missing_wheel_is_denied(tmp_path):
    wheels, lock, wheel = make_bundle(tmp_path)
    wheel.unlink()
    with pytest.raises(ValueError, match="missing pinned"):
        verifier.verify(wheels, lock)


def test_unpinned_extra_wheel_is_denied(tmp_path):
    wheels, lock, _ = make_bundle(tmp_path)
    make_wheel(wheels, "unexpected", "1.0")
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        verifier.verify(wheels, lock)


def test_malformed_and_duplicate_lock_entries_are_denied(tmp_path):
    wheels, lock, _ = make_bundle(tmp_path)
    line = lock.read_text()
    lock.write_text(line + "requests>=2.31.0\n")
    with pytest.raises(ValueError, match="unpinned"):
        verifier.verify(wheels, lock)
    lock.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate pinned"):
        verifier.verify(wheels, lock)


def test_symlinked_wheel_is_denied(tmp_path):
    wheels, lock, wheel = make_bundle(tmp_path)
    target = tmp_path / "copied.whl"
    target.write_bytes(wheel.read_bytes())
    wheel.unlink()
    wheel.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify(wheels, lock)


def test_incorrect_embedded_metadata_is_denied(tmp_path):
    wheels, lock, wheel = make_bundle(tmp_path)
    wheel.unlink()
    wheel = make_wheel(wheels, embedded_name="other-distribution")
    lock.write_text("lms-agent-bench==0.33.0 --hash=sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest() + "\n")
    with pytest.raises(ValueError, match="metadata mismatch"):
        verifier.verify(wheels, lock)


def test_unexpected_version_is_denied(tmp_path):
    wheels, lock, _ = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="unexpected LMS version"):
        verifier.verify(wheels, lock, expected_version="0.34.0")


def test_duplicate_distribution_wheels_are_denied(tmp_path):
    wheels, lock, _ = make_bundle(tmp_path)
    original = next(wheels.glob("*.whl"))
    duplicate = wheels / "lms_agent_bench-0.33.0-1-py3-none-any.whl"
    duplicate.write_bytes(original.read_bytes())
    duplicate.chmod(0o600)
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        verifier.verify(wheels, lock)


def test_cli_failures_return_nonzero_and_never_claim_authorization(tmp_path, capsys):
    wheels, lock, wheel = make_bundle(tmp_path)
    assert verifier.main(["--wheel-dir", str(wheels), "--lock", str(lock), "--expected-version", "0.33.0"]) == 0
    assert json.loads(capsys.readouterr().out)["production_authorized"] is False
    wheel.unlink()
    assert verifier.main(["--wheel-dir", str(wheels), "--lock", str(lock), "--expected-version", "0.33.0"]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["valid"] is False and failure["production_authorized"] is False


def test_writable_wheel_directory_is_denied(tmp_path):
    wheels, lock, _ = make_bundle(tmp_path)
    wheels.chmod(0o770)
    with pytest.raises(ValueError, match="group/world writable"):
        verifier.verify(wheels, lock)
