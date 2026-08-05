"""Backend-neutral prompt-cache registry with a record-only SQLite MVP."""
from __future__ import annotations

import abc
import dataclasses
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from lms_agent_bench.hermes_agent_common import canonical_hash, utc_now_iso
from lms_agent_bench.prompt_cache_identity import (
    COMPATIBILITY_SCHEMA_VERSION,
    DEFAULT_BLOCK_SIZE,
    PrefixSequence,
    build_prefix_sequence,
    require_sha256,
)
from lms_agent_bench.prompt_cache_store import ContentAddressedArtifactStore

REGISTRY_SCHEMA_VERSION = "prompt_cache_registry.v1"
ARTIFACT_STATES = {"staging", "verified", "ready", "stale", "evicted", "quarantined"}
SENSITIVITIES = {"public", "internal", "private", "restricted"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclasses.dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    block_hash: str
    compatibility_hash: str
    payload_uri: str
    payload_sha256: str
    serialization_format: str
    serialization_version: str
    size_bytes: int
    state: str
    sensitivity: str
    namespace: str
    node_id: str
    created_at_utc: str
    last_verified_at_utc: Optional[str]
    expires_at_utc: Optional[str]


class PromptCacheRegistry(abc.ABC):
    @abc.abstractmethod
    def initialize(self) -> None: ...

    @abc.abstractmethod
    def register_sequence(
        self, manifest: Mapping[str, Any], sequence: PrefixSequence, *, namespace: str
    ) -> None: ...

    @abc.abstractmethod
    def register_artifact(self, artifact: ArtifactRecord) -> None: ...

    @abc.abstractmethod
    def find_longest_ready_prefix(
        self, sequence: PrefixSequence, *, namespace: str, now_utc: Optional[str] = None
    ) -> Optional[ArtifactRecord]: ...

    @abc.abstractmethod
    def mark_artifact_state(self, artifact_id: str, state: str) -> None: ...

    @abc.abstractmethod
    def record_observation(self, observation: Mapping[str, Any]) -> str: ...

    @abc.abstractmethod
    def stats(self, *, namespace: Optional[str] = None) -> Dict[str, Any]: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS compatibility_manifests (
  compatibility_hash TEXT PRIMARY KEY, manifest_json TEXT NOT NULL,
  loadout_fingerprint TEXT NOT NULL, created_at_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prefix_blocks (
  block_hash TEXT PRIMARY KEY, compatibility_hash TEXT NOT NULL,
  parent_block_hash TEXT, block_index INTEGER NOT NULL,
  cumulative_token_count INTEGER NOT NULL, block_token_count INTEGER NOT NULL,
  token_ids_sha256 TEXT NOT NULL, created_at_utc TEXT NOT NULL,
  FOREIGN KEY (compatibility_hash) REFERENCES compatibility_manifests(compatibility_hash),
  FOREIGN KEY (parent_block_hash) REFERENCES prefix_blocks(block_hash));
CREATE TABLE IF NOT EXISTS prompt_sequences (
  sequence_hash TEXT NOT NULL, namespace TEXT NOT NULL,
  compatibility_hash TEXT NOT NULL, final_block_hash TEXT,
  token_count INTEGER NOT NULL, block_size INTEGER NOT NULL,
  created_at_utc TEXT NOT NULL, PRIMARY KEY (sequence_hash, namespace),
  FOREIGN KEY (compatibility_hash) REFERENCES compatibility_manifests(compatibility_hash),
  FOREIGN KEY (final_block_hash) REFERENCES prefix_blocks(block_hash));
CREATE TABLE IF NOT EXISTS kv_artifacts (
  artifact_id TEXT PRIMARY KEY, block_hash TEXT NOT NULL,
  compatibility_hash TEXT NOT NULL, payload_uri TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL, serialization_format TEXT NOT NULL,
  serialization_version TEXT NOT NULL, size_bytes INTEGER NOT NULL,
  state TEXT NOT NULL, sensitivity TEXT NOT NULL, namespace TEXT NOT NULL,
  node_id TEXT NOT NULL, created_at_utc TEXT NOT NULL,
  last_verified_at_utc TEXT, expires_at_utc TEXT,
  FOREIGN KEY (block_hash) REFERENCES prefix_blocks(block_hash),
  FOREIGN KEY (compatibility_hash) REFERENCES compatibility_manifests(compatibility_hash));
CREATE TABLE IF NOT EXISTS cache_observations (
  observation_id TEXT PRIMARY KEY, sequence_hash TEXT NOT NULL,
  compatibility_hash TEXT NOT NULL, artifact_id TEXT, outcome TEXT NOT NULL,
  matched_tokens INTEGER NOT NULL, tokens_skipped INTEGER NOT NULL,
  restore_ms REAL, prefill_ms_saved REAL, ttft_ms REAL,
  node_id TEXT NOT NULL, namespace TEXT NOT NULL, created_at_utc TEXT NOT NULL,
  details_json TEXT NOT NULL,
  FOREIGN KEY (artifact_id) REFERENCES kv_artifacts(artifact_id));
CREATE INDEX IF NOT EXISTS idx_prefix_parent ON prefix_blocks(parent_block_hash);
CREATE INDEX IF NOT EXISTS idx_prefix_compat_tokens
  ON prefix_blocks(compatibility_hash, cumulative_token_count);
CREATE INDEX IF NOT EXISTS idx_artifact_lookup
  ON kv_artifacts(namespace, compatibility_hash, block_hash, state);
CREATE INDEX IF NOT EXISTS idx_observation_namespace_time
  ON cache_observations(namespace, created_at_utc);
"""


class SQLitePromptCacheRegistry(PromptCacheRegistry):
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path).expanduser()

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT INTO registry_meta VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (REGISTRY_SCHEMA_VERSION,),
            )

    @staticmethod
    def _namespace(value: str) -> str:
        value = str(value or "").strip()
        if not value or len(value) > 256:
            raise ValueError("namespace must contain 1-256 characters")
        return value

    @staticmethod
    def _artifact(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            **{field.name: row[field.name] for field in dataclasses.fields(ArtifactRecord)}
        )

    def register_sequence(
        self, manifest: Mapping[str, Any], sequence: PrefixSequence, *, namespace: str
    ) -> None:
        self.initialize()
        namespace = self._namespace(namespace)
        manifest = dict(manifest)
        if manifest.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION:
            raise ValueError(f"compatibility schema must be {COMPATIBILITY_SCHEMA_VERSION}")
        compatibility_hash = require_sha256(
            str(manifest.get("compatibility_hash") or ""), "compatibility_hash"
        )
        unhashed = {key: value for key, value in manifest.items() if key != "compatibility_hash"}
        if canonical_hash(unhashed) != compatibility_hash:
            raise ValueError("compatibility manifest hash does not match contents")
        if sequence.compatibility_hash != compatibility_hash:
            raise ValueError("sequence compatibility does not match manifest")
        manifest_json = _canonical_json(manifest)
        created = utc_now_iso()
        final_block_hash = sequence.blocks[-1].block_hash if sequence.blocks else None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT manifest_json FROM compatibility_manifests WHERE compatibility_hash = ?",
                (compatibility_hash,),
            ).fetchone()
            if row is not None and row["manifest_json"] != manifest_json:
                connection.rollback()
                raise RuntimeError("compatibility hash collision")
            connection.execute(
                "INSERT OR IGNORE INTO compatibility_manifests VALUES (?, ?, ?, ?)",
                (compatibility_hash, manifest_json, manifest["loadout_fingerprint"], created),
            )
            for block in sequence.blocks:
                existing = connection.execute(
                    "SELECT compatibility_hash, parent_block_hash, block_index, "
                    "cumulative_token_count, block_token_count, token_ids_sha256 "
                    "FROM prefix_blocks WHERE block_hash = ?",
                    (block.block_hash,),
                ).fetchone()
                expected = dataclasses.astuple(block)[1:]
                if existing is not None and tuple(existing) != expected:
                    connection.rollback()
                    raise RuntimeError("prefix block hash collision")
                connection.execute(
                    "INSERT OR IGNORE INTO prefix_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (*dataclasses.astuple(block), created),
                )
            connection.execute(
                "INSERT OR IGNORE INTO prompt_sequences VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    sequence.sequence_hash,
                    namespace,
                    compatibility_hash,
                    final_block_hash,
                    sequence.token_count,
                    sequence.block_size,
                    created,
                ),
            )
            connection.commit()

    def register_artifact(self, artifact: ArtifactRecord) -> None:
        self.initialize()
        if artifact.state not in ARTIFACT_STATES or artifact.sensitivity not in SENSITIVITIES:
            raise ValueError("invalid artifact state or sensitivity")
        self._namespace(artifact.namespace)
        for name in ("artifact_id", "block_hash", "compatibility_hash", "payload_sha256"):
            require_sha256(str(getattr(artifact, name)), name)
        with self._connect() as connection:
            block = connection.execute(
                "SELECT compatibility_hash FROM prefix_blocks WHERE block_hash = ?",
                (artifact.block_hash,),
            ).fetchone()
            if block is None or block["compatibility_hash"] != artifact.compatibility_hash:
                raise ValueError("artifact prefix is missing or incompatible")
            values = dataclasses.asdict(artifact)
            try:
                connection.execute(
                    f"INSERT INTO kv_artifacts({', '.join(values)}) "
                    f"VALUES ({', '.join('?' for _ in values)})",
                    tuple(values.values()),
                )
            except sqlite3.IntegrityError:
                row = connection.execute(
                    "SELECT * FROM kv_artifacts WHERE artifact_id = ?",
                    (artifact.artifact_id,),
                ).fetchone()
                if row is None or self._artifact(row) != artifact:
                    raise RuntimeError("artifact ID collision")

    def find_longest_ready_prefix(
        self, sequence: PrefixSequence, *, namespace: str, now_utc: Optional[str] = None
    ) -> Optional[ArtifactRecord]:
        self.initialize()
        hashes = [block.block_hash for block in sequence.blocks]
        if not hashes:
            return None
        placeholders = ",".join("?" for _ in hashes)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT a.* FROM kv_artifacts a JOIN prefix_blocks b "
                f"ON b.block_hash=a.block_hash WHERE a.namespace=? "
                f"AND a.compatibility_hash=? AND a.state='ready' "
                f"AND a.block_hash IN ({placeholders}) "
                f"AND (a.expires_at_utc IS NULL OR a.expires_at_utc>?) "
                f"ORDER BY b.cumulative_token_count DESC, a.created_at_utc DESC LIMIT 1",
                (
                    self._namespace(namespace),
                    sequence.compatibility_hash,
                    *hashes,
                    now_utc or utc_now_iso(),
                ),
            ).fetchone()
        return None if row is None else self._artifact(row)

    def mark_artifact_state(self, artifact_id: str, state: str) -> None:
        if state not in ARTIFACT_STATES:
            raise ValueError(f"invalid artifact state: {state}")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE kv_artifacts SET state=? WHERE artifact_id=?",
                (state, require_sha256(artifact_id, "artifact_id")),
            )
            if cursor.rowcount != 1:
                raise KeyError(artifact_id)

    def record_observation(self, observation: Mapping[str, Any]) -> str:
        self.initialize()
        value = dict(observation)
        observation_id = str(value.get("observation_id") or uuid.uuid4())
        artifact_id = value.get("artifact_id")
        if artifact_id is not None:
            artifact_id = require_sha256(str(artifact_id), "artifact_id")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO cache_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    observation_id,
                    require_sha256(str(value["sequence_hash"]), "sequence_hash"),
                    require_sha256(str(value["compatibility_hash"]), "compatibility_hash"),
                    artifact_id,
                    str(value["outcome"]),
                    max(0, int(value.get("matched_tokens", 0))),
                    max(0, int(value.get("tokens_skipped", 0))),
                    value.get("restore_ms"),
                    value.get("prefill_ms_saved"),
                    value.get("ttft_ms"),
                    str(value["node_id"]),
                    self._namespace(str(value["namespace"])),
                    utc_now_iso(),
                    _canonical_json(dict(value.get("details") or {})),
                ),
            )
        return observation_id

    def stats(self, *, namespace: Optional[str] = None) -> Dict[str, Any]:
        self.initialize()
        clause, parameters = (
            ("", ())
            if namespace is None
            else (" WHERE namespace = ?", (self._namespace(namespace),))
        )
        with self._connect() as connection:
            artifacts = connection.execute(
                f"SELECT state, COUNT(*) count, COALESCE(SUM(size_bytes),0) bytes "
                f"FROM kv_artifacts{clause} GROUP BY state ORDER BY state",
                parameters,
            ).fetchall()
            observations = connection.execute(
                f"SELECT outcome, COUNT(*) count, "
                f"COALESCE(SUM(tokens_skipped),0) tokens_skipped, "
                f"COALESCE(SUM(prefill_ms_saved),0) prefill_ms_saved "
                f"FROM cache_observations{clause} GROUP BY outcome ORDER BY outcome",
                parameters,
            ).fetchall()
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "namespace": namespace,
            "artifacts": [dict(row) for row in artifacts],
            "observations": [dict(row) for row in observations],
            "admission": {"admitted": False},
        }


class PromptCacheRecorder:
    """Records candidates and evidence but never restores KV state."""

    def __init__(self, registry: PromptCacheRegistry, store: ContentAddressedArtifactStore):
        self.registry, self.store = registry, store

    def initialize(self) -> None:
        self.registry.initialize()
        self.store.initialize()

    def register_local_artifact(
        self,
        manifest: Mapping[str, Any],
        token_ids: Sequence[int],
        artifact_path: Path,
        *,
        namespace: str,
        node_id: str,
        serialization_format: str,
        serialization_version: str,
        sensitivity: str = "private",
        expires_at_utc: Optional[str] = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> ArtifactRecord:
        self.initialize()
        sequence = build_prefix_sequence(
            token_ids,
            str(manifest["compatibility_hash"]),
            block_size=block_size,
        )
        if not sequence.blocks:
            raise ValueError("cannot register an artifact for an empty prompt")
        self.registry.register_sequence(manifest, sequence, namespace=namespace)
        write = self.store.put_file(artifact_path)
        if not self.store.verify(write.payload_sha256):
            raise RuntimeError("payload failed post-write verification")
        now = utc_now_iso()
        artifact = ArtifactRecord(
            artifact_id=canonical_hash(
                {
                    "block_hash": sequence.blocks[-1].block_hash,
                    "compatibility_hash": sequence.compatibility_hash,
                    "payload_sha256": write.payload_sha256,
                    "serialization_format": serialization_format,
                    "serialization_version": serialization_version,
                    "namespace": namespace,
                    "node_id": node_id,
                }
            ),
            block_hash=sequence.blocks[-1].block_hash,
            compatibility_hash=sequence.compatibility_hash,
            payload_uri=write.payload_uri,
            payload_sha256=write.payload_sha256,
            serialization_format=serialization_format,
            serialization_version=serialization_version,
            size_bytes=write.size_bytes,
            state="ready",
            sensitivity=sensitivity,
            namespace=namespace,
            node_id=node_id,
            created_at_utc=now,
            last_verified_at_utc=now,
            expires_at_utc=expires_at_utc,
        )
        self.registry.register_artifact(artifact)
        return artifact

    def observe_request(
        self,
        manifest: Mapping[str, Any],
        token_ids: Sequence[int],
        *,
        namespace: str,
        node_id: str,
        block_size: int = DEFAULT_BLOCK_SIZE,
        estimated_prefill_ms_per_token: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.initialize()
        sequence = build_prefix_sequence(
            token_ids,
            str(manifest["compatibility_hash"]),
            block_size=block_size,
        )
        self.registry.register_sequence(manifest, sequence, namespace=namespace)
        candidate = self.registry.find_longest_ready_prefix(
            sequence,
            namespace=namespace,
        )
        outcome, matched_tokens, artifact_id, verified = "miss", 0, None, False
        if candidate is not None:
            artifact_id = candidate.artifact_id
            matched_tokens = next(
                block.cumulative_token_count
                for block in sequence.blocks
                if block.block_hash == candidate.block_hash
            )
            verified = self.store.verify(candidate.payload_sha256)
            if verified:
                outcome = "candidate_hit"
            else:
                outcome, matched_tokens = "verification_failed", 0
                self.registry.mark_artifact_state(
                    candidate.artifact_id,
                    "quarantined",
                )
                self.store.quarantine(
                    candidate.payload_sha256,
                    "verification-failed",
                )
        estimated = None
        if estimated_prefill_ms_per_token is not None:
            rate = float(estimated_prefill_ms_per_token)
            if rate < 0:
                raise ValueError("estimated_prefill_ms_per_token cannot be negative")
            estimated = rate * matched_tokens
        observation_id = self.registry.record_observation(
            {
                "sequence_hash": sequence.sequence_hash,
                "compatibility_hash": sequence.compatibility_hash,
                "artifact_id": artifact_id,
                "outcome": outcome,
                "matched_tokens": matched_tokens,
                "tokens_skipped": 0,
                "prefill_ms_saved": 0.0,
                "node_id": node_id,
                "namespace": namespace,
                "details": {
                    "record_only": True,
                    "restoration_attempted": False,
                    "payload_verified": verified,
                    "estimated_prefill_ms_saved": estimated,
                    "requested_token_count": sequence.token_count,
                    "block_size": block_size,
                },
            }
        )
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "mode": "record_only",
            "observation_id": observation_id,
            "sequence_hash": sequence.sequence_hash,
            "compatibility_hash": sequence.compatibility_hash,
            "requested_token_count": sequence.token_count,
            "matched_prefix_tokens": matched_tokens,
            "outcome": outcome,
            "candidate_artifact_id": artifact_id,
            "payload_verified": verified,
            "estimated_prefill_ms_saved": estimated,
            "restoration_attempted": False,
            "tokens_skipped": 0,
            "admission": {"admitted": False},
        }
