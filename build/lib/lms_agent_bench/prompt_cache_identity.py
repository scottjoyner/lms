"""Exact loadout and token-prefix identities for prompt-cache evidence."""
from __future__ import annotations

import dataclasses
import hashlib
import struct
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from lms_agent_bench.hermes_agent_common import canonical_hash, normalize_sha256
from lms_agent_bench.model_loadout import identity_core, validate_manifest

COMPATIBILITY_SCHEMA_VERSION = "prompt_cache_compatibility.v1"
DEFAULT_BLOCK_SIZE = 256


def require_sha256(value: str, name: str) -> str:
    try:
        return normalize_sha256(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 value") from exc


def optional_sha256(value: Optional[str], name: str) -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    return require_sha256(str(value), name)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def build_compatibility_manifest(
    loadout: Mapping[str, Any],
    *,
    tokenizer_sha256: str,
    chat_template_sha256: str,
    system_prompt_sha256: str,
    tool_schema_sha256: str,
    engine_serialization_abi: str,
    adapter_sha256: Optional[str] = None,
    multimodal_encoder_sha256: Optional[str] = None,
    preprocessing_sha256: Optional[str] = None,
    device_layout: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the exact compatibility identity required for raw KV reuse."""
    validated = validate_manifest(loadout, require_fingerprint=False)
    abi = str(engine_serialization_abi or "").strip()
    if not abi:
        raise ValueError("engine_serialization_abi is required")
    manifest: Dict[str, Any] = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "loadout_fingerprint": validated["loadout_fingerprint"],
        "loadout_identity": identity_core(validated),
        "tokenizer_sha256": require_sha256(tokenizer_sha256, "tokenizer_sha256"),
        "chat_template_sha256": require_sha256(
            chat_template_sha256, "chat_template_sha256"
        ),
        "system_prompt_sha256": require_sha256(
            system_prompt_sha256, "system_prompt_sha256"
        ),
        "tool_schema_sha256": require_sha256(
            tool_schema_sha256, "tool_schema_sha256"
        ),
        "adapter_sha256": optional_sha256(adapter_sha256, "adapter_sha256"),
        "multimodal_encoder_sha256": optional_sha256(
            multimodal_encoder_sha256, "multimodal_encoder_sha256"
        ),
        "preprocessing_sha256": optional_sha256(
            preprocessing_sha256, "preprocessing_sha256"
        ),
        "engine_serialization_abi": abi,
        "device_layout": dict(device_layout or {}),
    }
    manifest["compatibility_hash"] = canonical_hash(manifest)
    return manifest


@dataclasses.dataclass(frozen=True)
class PrefixBlock:
    block_hash: str
    compatibility_hash: str
    parent_block_hash: Optional[str]
    block_index: int
    cumulative_token_count: int
    block_token_count: int
    token_ids_sha256: str


@dataclasses.dataclass(frozen=True)
class PrefixSequence:
    sequence_hash: str
    compatibility_hash: str
    token_count: int
    block_size: int
    blocks: Tuple[PrefixBlock, ...]


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    payload = bytearray()
    for index, raw in enumerate(token_ids):
        if isinstance(raw, bool):
            raise ValueError(f"token_ids[{index}] must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"token_ids[{index}] must be an integer") from exc
        if value < 0 or value > 2**64 - 1:
            raise ValueError(f"token_ids[{index}] is outside uint64 range")
        payload.extend(struct.pack(">Q", value))
    return bytes(payload)


def build_prefix_sequence(
    token_ids: Sequence[int],
    compatibility_hash: str,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> PrefixSequence:
    """Build a Merkle-style prefix chain from exact token IDs."""
    compatibility_hash = require_sha256(
        compatibility_hash, "compatibility_hash"
    )
    if isinstance(block_size, bool) or int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    block_size = int(block_size)
    normalized = [int(item) for item in token_ids]
    _token_bytes(normalized)
    compatibility_digest = bytes.fromhex(compatibility_hash[7:])
    parent_digest = bytes(32)
    parent_hash: Optional[str] = None
    cumulative = 0
    blocks = []
    for block_index, offset in enumerate(range(0, len(normalized), block_size)):
        current = normalized[offset : offset + block_size]
        token_payload = _token_bytes(current)
        cumulative += len(current)
        block_hash = sha256_bytes(
            b"lms-prefix-block-v1\0"
            + compatibility_digest
            + parent_digest
            + struct.pack(">IQQ", block_size, block_index, cumulative)
            + token_payload
        )
        blocks.append(
            PrefixBlock(
                block_hash=block_hash,
                compatibility_hash=compatibility_hash,
                parent_block_hash=parent_hash,
                block_index=block_index,
                cumulative_token_count=cumulative,
                block_token_count=len(current),
                token_ids_sha256=sha256_bytes(token_payload),
            )
        )
        parent_hash = block_hash
        parent_digest = bytes.fromhex(block_hash[7:])
    sequence_hash = sha256_bytes(
        b"lms-prefix-sequence-v1\0"
        + compatibility_digest
        + parent_digest
        + struct.pack(">QQ", block_size, len(normalized))
    )
    return PrefixSequence(
        sequence_hash=sequence_hash,
        compatibility_hash=compatibility_hash,
        token_count=len(normalized),
        block_size=block_size,
        blocks=tuple(blocks),
    )
