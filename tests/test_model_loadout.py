from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lms_agent_bench.model_loadout import (
    build_matrix,
    validate_manifest,
)


def example_bases():
    payload = json.loads(
        Path("examples/model-loadouts.v1.example.json").read_text(encoding="utf-8")
    )
    return payload["base_manifests"]


def test_dense_and_moe_architecture_identity_is_explicit():
    dense, moe = [validate_manifest(item) for item in example_bases()]

    assert dense["architecture"]["kind"] == "dense"
    assert dense["architecture"]["active_parameter_count_per_token"] == 8_000_000_000
    assert dense["derived"]["active_parameter_ratio"] == 1.0

    assert moe["architecture"]["kind"] == "moe"
    assert moe["architecture"]["total_parameter_count"] == 35_000_000_000
    assert moe["architecture"]["active_parameter_count_per_token"] == 3_000_000_000
    assert moe["architecture"]["expert_count_total"] == 128
    assert moe["architecture"]["expert_count_active_per_token"] == 8
    assert moe["derived"]["active_parameter_ratio"] == pytest.approx(3 / 35)


def test_kv_cache_is_separate_from_weight_quantization_and_fingerprinted():
    dense = validate_manifest(example_bases()[0])
    changed = copy.deepcopy(dense)
    changed.pop("loadout_fingerprint")
    changed.pop("derived")
    changed.pop("admission")
    changed["kv_cache"]["key_dtype"] = "q4_0"
    changed["kv_cache"]["value_dtype"] = "q4_0"
    changed = validate_manifest(changed)

    assert dense["weight_quantization"] == changed["weight_quantization"]
    assert dense["loadout_fingerprint"] != changed["loadout_fingerprint"]
    assert dense["derived"]["kv_bytes_per_token"] == 65_536
    assert changed["derived"]["kv_bytes_per_token"] == 32_768
    assert dense["derived"]["estimated_kv_cache_bytes"] == 536_870_912


def test_moe_requires_active_parameter_and_expert_metadata():
    manifest = example_bases()[1]
    manifest = copy.deepcopy(manifest)
    manifest["architecture"].pop("active_parameter_count_per_token")
    with pytest.raises(ValueError, match="active_parameter_count_per_token"):
        validate_manifest(manifest)

    manifest = copy.deepcopy(example_bases()[1])
    manifest["architecture"].pop("expert_count_active_per_token")
    with pytest.raises(ValueError, match="expert_count_active_per_token"):
        validate_manifest(manifest)


def test_context_above_native_requires_explicit_rope_scaling():
    manifest = copy.deepcopy(example_bases()[0])
    manifest["context"]["configured_tokens"] = 65_536
    manifest["kv_cache"]["capacity_tokens"] = 65_536
    with pytest.raises(ValueError, match="rope_scaling_type"):
        validate_manifest(manifest)

    manifest["context"]["rope_scaling_type"] = "yarn"
    assert validate_manifest(manifest)["context"]["configured_tokens"] == 65_536


def test_controlled_matrix_generates_unique_exact_loadouts():
    axes = json.loads(
        Path("examples/model-loadout-matrix.axes.example.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = build_matrix(example_bases(), axes)

    assert matrix["candidate_count"] == 128
    fingerprints = [item["loadout_fingerprint"] for item in matrix["candidates"]]
    assert len(fingerprints) == len(set(fingerprints))
    assert {item["architecture"]["kind"] for item in matrix["candidates"]} == {
        "dense",
        "moe",
    }
    assert {item["context"]["configured_tokens"] for item in matrix["candidates"]} == {
        4096,
        8192,
        16384,
        32768,
    }
    assert all(item["kv_cache"]["capacity_tokens"] == 32768 for item in matrix["candidates"])


def test_weight_quantization_variants_require_separate_hashed_model_bases():
    with pytest.raises(ValueError, match="separate base manifests"):
        build_matrix(
            [example_bases()[0]],
            {"weight_quantization.scheme": ["Q4_K_M", "Q8_0"]},
        )
