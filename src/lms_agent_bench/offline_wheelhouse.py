"""Read-only hash-locked wheelhouse verification for offline LMS staging.

This is artifact integrity validation, NOT a physical canary approval or an
independent signature/trust-chain verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

_LOCK = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([0-9][A-Za-z0-9.!+_-]*) --hash=sha256:([a-f0-9]{64})$")


def _canon(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _protected(path: Path, *, directory: bool = False) -> None:
    if path.is_symlink():
        raise ValueError(f"symlink not allowed: {path}")
    info = path.stat()
    valid_type = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not valid_type:
        raise ValueError(f"invalid file type: {path}")
    if info.st_mode & 0o022:
        raise ValueError(f"group/world writable input: {path}")


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _wheel_identity(path: Path) -> tuple[str, str]:
    if not path.name.endswith(".whl"):
        raise ValueError(f"not a wheel: {path.name}")
    parts = path.name[:-4].split("-")
    if len(parts) not in (5, 6) or not all(parts):
        raise ValueError(f"invalid wheel filename: {path.name}")
    dist, version = _canon(parts[0]), parts[1]
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_]*", version):
        raise ValueError(f"invalid wheel version: {path.name}")
    return dist, version


def verify(wheel_dir: Path, lock_file: Path, *, expected_name: str = "lms-agent-bench", expected_version: str | None = None) -> dict[str, Any]:
    """Check exact wheel inventory, zip CRC, metadata names, and SHA-256 pins."""
    _protected(wheel_dir, directory=True)
    _protected(lock_file)
    locked: dict[str, tuple[str, str]] = {}
    for line in lock_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK.fullmatch(line)
        if not match:
            raise ValueError("lock contains unpinned or unsupported requirement")
        name, version, digest = match.groups()
        name = _canon(name)
        if name in locked:
            raise ValueError(f"duplicate pinned distribution: {name}")
        locked[name] = version, digest
    if not locked or _canon(expected_name) not in locked:
        raise ValueError("expected LMS distribution is not pinned")
    if expected_version is not None and locked[_canon(expected_name)][0] != expected_version:
        raise ValueError("unexpected LMS version")

    inventory = []
    observed: set[str] = set()
    for path in sorted(wheel_dir.iterdir()):
        if path.suffix != ".whl":
            continue
        _protected(path)
        name, version = _wheel_identity(path)
        if name not in locked or name in observed:
            raise ValueError(f"unexpected or duplicate wheel distribution: {name}")
        if version != locked[name][0]:
            raise ValueError(f"wheel version mismatch: {name}")
        digest = _sha256(path)
        if digest != locked[name][1]:
            raise ValueError(f"wheel hash mismatch: {name}")
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"invalid wheel CRC: {path.name}")
                metadata = [x for x in archive.namelist() if x.endswith(".dist-info/METADATA")]
                if len(metadata) != 1:
                    raise ValueError(f"missing/ambiguous wheel metadata: {path.name}")
                if archive.getinfo(metadata[0]).file_size > 2 * 1024 * 1024:
                    raise ValueError(f"oversized wheel metadata: {path.name}")
                payload = archive.read(metadata[0]).decode("utf-8")
                fields = {}
                for raw in payload.splitlines():
                    if raw.startswith(("Name: ", "Version: ")):
                        key, value = raw.split(": ", 1)
                        if key in fields:
                            raise ValueError(f"duplicate wheel metadata field: {key}")
                        fields[key] = value.strip()
                if _canon(fields.get("Name", "")) != name or fields.get("Version") != version:
                    raise ValueError(f"wheel embedded metadata mismatch: {path.name}")
        except (zipfile.BadZipFile, UnicodeDecodeError, KeyError) as exc:
            raise ValueError(f"unreadable wheel: {path.name}") from exc
        observed.add(name)
        inventory.append({"name": name, "version": version, "filename": path.name, "sha256": digest})
    if observed != set(locked):
        raise ValueError(f"missing pinned wheel distributions: {sorted(set(locked) - observed)}")
    return {"schema": "lms-offline-wheel-verification-v1", "valid": True, "wheel_count": len(inventory), "lock_sha256": _sha256(lock_file), "wheels": inventory, "production_authorized": False, "independent_signatures_verified": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify local LMS wheel hashes; does not install or authorize a runtime")
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.wheel_dir, args.lock, expected_version=args.expected_version), sort_keys=True))
    except (ValueError, OSError) as exc:
        print(json.dumps({"valid": False, "reason": str(exc), "production_authorized": False}, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
