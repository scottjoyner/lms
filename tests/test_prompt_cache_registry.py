from __future__ import annotations

from lms_agent_bench.prompt_cache_identity import (
    build_compatibility_manifest,
    build_prefix_sequence,
)
from lms_agent_bench.prompt_cache_registry import (
    PromptCacheRecorder,
    SQLitePromptCacheRegistry,
)
from lms_agent_bench.prompt_cache_store import ContentAddressedArtifactStore

HASH_A = "a" * 64


def loadout():
    return {
        "schema_version": "model_loadout_manifest.v1",
        "node_id": "node",
        "candidate_id": "demo",
        "model": {
            "id": "demo",
            "content_sha256": "sha256:" + HASH_A,
            "format": "gguf",
        },
        "architecture": {
            "kind": "dense",
            "total_parameter_count": 1,
            "active_parameter_count_per_token": 1,
        },
        "weight_quantization": {"scheme": "q4"},
        "runtime": {
            "engine": "llama.cpp",
            "backend": "cpu",
            "version": "1",
            "build_commit": "abc",
        },
        "context": {
            "configured_tokens": 4096,
            "model_native_tokens": 4096,
        },
        "kv_cache": {
            "key_dtype": "q8_0",
            "value_dtype": "q8_0",
            "location": "cpu",
        },
        "concurrency": {"parallel_slots": 1},
        "speculative_decoding": {"enabled": False},
    }


def compatibility(**changes):
    arguments = {
        "tokenizer_sha256": HASH_A,
        "chat_template_sha256": "b" * 64,
        "system_prompt_sha256": "c" * 64,
        "tool_schema_sha256": "d" * 64,
        "engine_serialization_abi": "llama-slot-v1",
    }
    arguments.update(changes)
    return build_compatibility_manifest(loadout(), **arguments)


def recorder(root):
    return PromptCacheRecorder(
        SQLitePromptCacheRegistry(root / "registry.sqlite3"),
        ContentAddressedArtifactStore(root),
    )


def test_prefix_identity_is_exact():
    first = compatibility()
    second = compatibility(system_prompt_sha256="e" * 64)
    assert first["compatibility_hash"] != second["compatibility_hash"]
    sequence = build_prefix_sequence(range(10), first["compatibility_hash"], block_size=4)
    same = build_prefix_sequence(range(10), first["compatibility_hash"], block_size=4)
    changed = build_prefix_sequence(
        [*range(9), 99], first["compatibility_hash"], block_size=4
    )
    assert sequence == same
    assert len(sequence.blocks) == 3
    assert sequence.blocks[1].cumulative_token_count == 8
    assert sequence.sequence_hash != changed.sequence_hash


def test_content_addressed_store_is_atomic_and_deduplicates(tmp_path):
    source = tmp_path / "slot.bin"
    source.write_bytes(b"opaque-kv")
    store = ContentAddressedArtifactStore(tmp_path / "cache")
    first = store.put_file(source)
    second = store.put_file(source)
    assert first.created is True
    assert second.created is False
    assert first.path == second.path
    assert store.verify(first.payload_sha256)


def test_longest_prefix_lookup_preserves_record_only_boundary(tmp_path):
    service = recorder(tmp_path / "cache")
    manifest = compatibility()
    payload = tmp_path / "slot.bin"
    payload.write_bytes(b"kv-prefix")
    artifact = service.register_local_artifact(
        manifest,
        [1, 2, 3, 4],
        payload,
        namespace="project-a",
        node_id="node-a",
        serialization_format="llama-slot-cache",
        serialization_version="1",
        block_size=2,
    )
    report = service.observe_request(
        manifest,
        [1, 2, 3, 4, 5, 6],
        namespace="project-a",
        node_id="node-b",
        block_size=2,
        estimated_prefill_ms_per_token=2.5,
    )
    assert report["outcome"] == "candidate_hit"
    assert report["candidate_artifact_id"] == artifact.artifact_id
    assert report["matched_prefix_tokens"] == 4
    assert report["estimated_prefill_ms_saved"] == 10.0
    assert report["restoration_attempted"] is False
    assert report["tokens_skipped"] == 0
    assert report["admission"]["admitted"] is False


def test_namespace_and_compatibility_mismatches_always_miss(tmp_path):
    service = recorder(tmp_path / "cache")
    first = compatibility()
    second = compatibility(tool_schema_sha256="f" * 64)
    payload = tmp_path / "slot.bin"
    payload.write_bytes(b"kv")
    service.register_local_artifact(
        first,
        [1, 2, 3, 4],
        payload,
        namespace="private-a",
        node_id="node-a",
        serialization_format="llama-slot-cache",
        serialization_version="1",
        block_size=2,
    )
    cross_namespace = service.observe_request(
        first,
        [1, 2, 3, 4, 5],
        namespace="private-b",
        node_id="node-b",
        block_size=2,
    )
    cross_compatibility = service.observe_request(
        second,
        [1, 2, 3, 4, 5],
        namespace="private-a",
        node_id="node-b",
        block_size=2,
    )
    assert cross_namespace["outcome"] == "miss"
    assert cross_compatibility["outcome"] == "miss"


def test_corruption_is_quarantined_and_never_counted_as_hit(tmp_path):
    service = recorder(tmp_path / "cache")
    manifest = compatibility()
    payload = tmp_path / "slot.bin"
    payload.write_bytes(b"kv")
    artifact = service.register_local_artifact(
        manifest,
        [1, 2, 3, 4],
        payload,
        namespace="project",
        node_id="node-a",
        serialization_format="llama-slot-cache",
        serialization_version="1",
        block_size=2,
    )
    payload_path = service.store.path_for(artifact.payload_sha256)
    payload_path.write_bytes(b"corrupt")
    report = service.observe_request(
        manifest,
        [1, 2, 3, 4, 5],
        namespace="project",
        node_id="node-b",
        block_size=2,
    )
    assert report["outcome"] == "verification_failed"
    assert report["matched_prefix_tokens"] == 0
    assert report["restoration_attempted"] is False
    assert not payload_path.exists()
    assert list(service.store.quarantine_root.iterdir())


def test_candidate_hits_do_not_claim_actual_savings(tmp_path):
    service = recorder(tmp_path / "cache")
    manifest = compatibility()
    payload = tmp_path / "slot.bin"
    payload.write_bytes(b"kv")
    service.register_local_artifact(
        manifest,
        [1, 2],
        payload,
        namespace="project",
        node_id="node-a",
        serialization_format="test",
        serialization_version="1",
        block_size=2,
    )
    service.observe_request(
        manifest,
        [1, 2, 3],
        namespace="project",
        node_id="node-b",
        block_size=2,
    )
    stats = service.registry.stats(namespace="project")
    assert stats["observations"][0]["outcome"] == "candidate_hit"
    assert stats["observations"][0]["tokens_skipped"] == 0
    assert stats["observations"][0]["prefill_ms_saved"] == 0
    assert stats["admission"]["admitted"] is False
