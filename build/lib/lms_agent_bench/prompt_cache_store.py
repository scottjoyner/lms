"""Atomic content-addressed storage for opaque engine-native KV artifacts."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from lms_agent_bench.prompt_cache_identity import require_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ArtifactWrite:
    payload_sha256: str
    payload_uri: str
    path: Path
    size_bytes: int
    created: bool


class ContentAddressedArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser()
        self.blob_root = self.root / "blobs" / "sha256"
        self.staging_root = self.root / "staging"
        self.quarantine_root = self.root / "quarantine"

    def initialize(self) -> None:
        for path in (self.blob_root, self.staging_root, self.quarantine_root):
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.chmod(0o700)
            except OSError:
                pass

    def path_for(self, payload_sha256: str) -> Path:
        digest = require_sha256(payload_sha256, "payload_sha256")[7:]
        return self.blob_root / digest[:2] / digest[2:4] / f"{digest}.blob"

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            descriptor = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def put_file(self, source: Path) -> ArtifactWrite:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        self.initialize()
        payload_sha256 = sha256_file(source)
        target = self.path_for(payload_sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != payload_sha256:
                raise RuntimeError("existing content-addressed payload is corrupt")
            return ArtifactWrite(
                payload_sha256, target.resolve().as_uri(), target,
                target.stat().st_size, False
            )
        staged = self.staging_root / f"{uuid.uuid4().hex}.staging"
        try:
            with source.open("rb") as reader, staged.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            try:
                staged.chmod(0o600)
            except OSError:
                pass
            if sha256_file(staged) != payload_sha256:
                raise RuntimeError("staged payload failed verification")
            try:
                os.link(staged, target)
                created = True
            except FileExistsError:
                created = False
                if sha256_file(target) != payload_sha256:
                    raise RuntimeError("concurrent payload is corrupt")
            self._fsync_dir(target.parent)
        finally:
            staged.unlink(missing_ok=True)
        return ArtifactWrite(
            payload_sha256, target.resolve().as_uri(), target,
            target.stat().st_size, created
        )

    def verify(self, payload_sha256: str) -> bool:
        path = self.path_for(payload_sha256)
        return path.is_file() and sha256_file(path) == require_sha256(
            payload_sha256, "payload_sha256"
        )

    def quarantine(self, payload_sha256: str, reason: str) -> Optional[Path]:
        self.initialize()
        source = self.path_for(payload_sha256)
        if not source.exists():
            return None
        safe_reason = re.sub(r"[^a-zA-Z0-9_.-]+", "-", reason)[:64] or "invalid"
        target = self.quarantine_root / (
            f"{payload_sha256[7:]}.{safe_reason}.{uuid.uuid4().hex}.blob"
        )
        os.replace(source, target)
        self._fsync_dir(self.quarantine_root)
        return target
