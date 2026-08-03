#!/usr/bin/env python3
"""Exact model/runtime loadout identity and controlled benchmark matrices.

A benchmark result belongs to one immutable loadout, not merely a model name.
This module records architecture, weight quantization, context configuration,
KV-cache representation, backend/offload, batching, concurrency, and optional
speculative decoding, then fingerprints the complete identity.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from lms_agent_bench.hermes_agent_common import canonical_hash, load_json, normalize_sha256, write_json

SCHEMA_VERSION = "model_loadout_manifest.v1"
MATRIX_SCHEMA_VERSION = "model_loadout_matrix.v1"
_ARCHITECTURE_KINDS = {"dense", "moe", "hybrid_moe", "recurrent", "other"}
_KV_LOCATIONS = {"cpu", "gpu", "unified", "mixed", "unknown"}


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _positive_int(value: Any, name: str, *, optional: bool = False) -> Optional[int]:
    if value is None and optional:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _positive_float(value: Any, name: str, *, optional: bool = False) -> Optional[float]:
    if value is None and optional:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def dtype_bits(dtype: str, explicit: Any = None) -> Optional[float]:
    if explicit is not None:
        return _positive_float(explicit, "KV dtype bits")
    normalized = str(dtype or "").strip().lower().replace("-", "_")
    aliases = {
        "f64": 64.0,
        "fp64": 64.0,
        "f32": 32.0,
        "fp32": 32.0,
        "f16": 16.0,
        "fp16": 16.0,
        "bf16": 16.0,
        "q8_0": 8.0,
        "int8": 8.0,
        "q6_k": 6.0,
        "q5_k": 5.0,
        "q5_1": 5.0,
        "q5_0": 5.0,
        "q4_0": 4.0,
        "q4_1": 4.0,
        "q4_k": 4.0,
        "int4": 4.0,
        "q3_k": 3.0,
        "q2_k": 2.0,
        "fp8": 8.0,
    }
    return aliases.get(normalized)


def estimate_kv_bytes_per_token(manifest: Mapping[str, Any]) -> Optional[float]:
    kv = manifest["kv_cache"]
    explicit = kv.get("bytes_per_token")
    if explicit is not None:
        return _positive_float(explicit, "kv_cache.bytes_per_token")
    architecture = manifest["architecture"]
    layers = architecture.get("layer_count")
    hidden = architecture.get("hidden_size")
    heads = architecture.get("head_count")
    kv_heads = architecture.get("kv_head_count") or heads
    if not all((layers, hidden, heads, kv_heads)):
        return None
    key_bits = dtype_bits(str(kv.get("key_dtype") or ""), kv.get("key_bits"))
    value_bits = dtype_bits(str(kv.get("value_dtype") or ""), kv.get("value_bits"))
    if key_bits is None or value_bits is None:
        return None
    head_dimension = float(hidden) / float(heads)
    return float(layers) * float(kv_heads) * head_dimension * (key_bits + value_bits) / 8.0


def identity_core(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key not in {"loadout_fingerprint", "derived", "admission"}
    }


def comparison_group(manifest: Mapping[str, Any]) -> str:
    core = {
        "model": manifest["model"],
        "architecture": manifest["architecture"],
        "weight_quantization": manifest["weight_quantization"],
        "runtime": {
            key: value
            for key, value in manifest["runtime"].items()
            if key
            not in {
                "batch_size",
                "ubatch_size",
                "flash_attention",
                "gpu_layers",
                "tensor_split",
                "threads",
                "engine_arguments",
            }
        },
        "speculative_decoding": manifest.get("speculative_decoding", {"enabled": False}),
    }
    return canonical_hash(core)


def validate_manifest(raw: Mapping[str, Any], *, require_fingerprint: bool = False) -> Dict[str, Any]:
    manifest = copy.deepcopy(dict(raw))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"loadout schema must be {SCHEMA_VERSION}")
    for field in ("node_id", "candidate_id"):
        if not str(manifest.get(field) or "").strip():
            raise ValueError(f"loadout requires {field}")

    model = _mapping(manifest.get("model"), "model")
    if not str(model.get("id") or "").strip():
        raise ValueError("model.id is required")
    model["content_sha256"] = normalize_sha256(str(model.get("content_sha256") or ""))
    if not str(model.get("format") or "").strip():
        raise ValueError("model.format is required")
    if model.get("size_bytes") is not None:
        model["size_bytes"] = _positive_int(model["size_bytes"], "model.size_bytes")
    manifest["model"] = model

    architecture = _mapping(manifest.get("architecture"), "architecture")
    kind = str(architecture.get("kind") or "").strip()
    if kind not in _ARCHITECTURE_KINDS:
        raise ValueError("architecture.kind must be dense, moe, hybrid_moe, recurrent, or other")
    total = _positive_int(architecture.get("total_parameter_count"), "architecture.total_parameter_count")
    active = _positive_int(
        architecture.get("active_parameter_count_per_token"),
        "architecture.active_parameter_count_per_token",
        optional=True,
    )
    if kind == "dense":
        active = active or total
        if active != total:
            raise ValueError("dense architecture active parameters must equal total parameters")
    if kind in {"moe", "hybrid_moe"}:
        if active is None:
            raise ValueError("MoE architecture requires active_parameter_count_per_token")
        if active > total:
            raise ValueError("active parameters cannot exceed total parameters")
        expert_total = _positive_int(architecture.get("expert_count_total"), "architecture.expert_count_total")
        expert_active = _positive_int(
            architecture.get("expert_count_active_per_token"),
            "architecture.expert_count_active_per_token",
        )
        if expert_active > expert_total:
            raise ValueError("active expert count cannot exceed total expert count")
        architecture["expert_count_total"] = expert_total
        architecture["expert_count_active_per_token"] = expert_active
    architecture["total_parameter_count"] = total
    architecture["active_parameter_count_per_token"] = active
    for field in ("layer_count", "hidden_size", "head_count", "kv_head_count"):
        if architecture.get(field) is not None:
            architecture[field] = _positive_int(architecture[field], f"architecture.{field}")
    manifest["architecture"] = architecture

    quant = _mapping(manifest.get("weight_quantization"), "weight_quantization")
    if not str(quant.get("scheme") or "").strip():
        raise ValueError("weight_quantization.scheme is required")
    for field in ("nominal_bits", "effective_bits_per_weight"):
        if quant.get(field) is not None:
            quant[field] = _positive_float(quant[field], f"weight_quantization.{field}")
    if quant.get("group_size") is not None:
        quant["group_size"] = _positive_int(quant["group_size"], "weight_quantization.group_size")
    quant["mixed_precision"] = bool(quant.get("mixed_precision", False))
    manifest["weight_quantization"] = quant

    runtime = _mapping(manifest.get("runtime"), "runtime")
    if not str(runtime.get("engine") or "").strip() or not str(runtime.get("backend") or "").strip():
        raise ValueError("runtime.engine and runtime.backend are required")
    for field in ("threads", "batch_size", "ubatch_size"):
        if runtime.get(field) is not None:
            runtime[field] = _positive_int(runtime[field], f"runtime.{field}")
    if runtime.get("gpu_layers") is not None:
        runtime["gpu_layers"] = int(runtime["gpu_layers"])
        if runtime["gpu_layers"] < 0:
            raise ValueError("runtime.gpu_layers cannot be negative")
    manifest["runtime"] = runtime

    context = _mapping(manifest.get("context"), "context")
    configured = _positive_int(context.get("configured_tokens"), "context.configured_tokens")
    native = _positive_int(context.get("model_native_tokens"), "context.model_native_tokens", optional=True)
    if native and configured > native and not context.get("rope_scaling_type"):
        raise ValueError("context exceeds model native context without an explicit rope_scaling_type")
    context["configured_tokens"] = configured
    context["model_native_tokens"] = native
    if context.get("prompt_tokens_target") is not None:
        target = _positive_int(context["prompt_tokens_target"], "context.prompt_tokens_target")
        if target > configured:
            raise ValueError("prompt_tokens_target cannot exceed configured context")
        context["prompt_tokens_target"] = target
    manifest["context"] = context

    kv = _mapping(manifest.get("kv_cache"), "kv_cache")
    for field in ("key_dtype", "value_dtype"):
        if not str(kv.get(field) or "").strip():
            raise ValueError(f"kv_cache.{field} is required")
    if str(kv.get("location") or "") not in _KV_LOCATIONS:
        raise ValueError("kv_cache.location is invalid")
    for field in ("key_bits", "value_bits", "bytes_per_token"):
        if kv.get(field) is not None:
            kv[field] = _positive_float(kv[field], f"kv_cache.{field}")
    if kv.get("capacity_tokens") is not None:
        kv["capacity_tokens"] = _positive_int(kv["capacity_tokens"], "kv_cache.capacity_tokens")
        if kv["capacity_tokens"] < configured:
            raise ValueError("kv_cache.capacity_tokens is smaller than configured context")
    kv["shared_across_requests"] = bool(kv.get("shared_across_requests"))
    kv["prefix_reuse"] = bool(kv.get("prefix_reuse"))
    kv["persistent"] = bool(kv.get("persistent", False))
    manifest["kv_cache"] = kv

    concurrency = _mapping(manifest.get("concurrency"), "concurrency")
    slots = _positive_int(concurrency.get("parallel_slots"), "concurrency.parallel_slots")
    concurrency["parallel_slots"] = slots
    concurrency["continuous_batching"] = bool(concurrency.get("continuous_batching"))
    manifest["concurrency"] = concurrency

    speculative = _mapping(manifest.get("speculative_decoding", {"enabled": False}), "speculative_decoding")
    speculative["enabled"] = bool(speculative.get("enabled", False))
    if speculative["enabled"]:
        if not str(speculative.get("draft_model_id") or "").strip():
            raise ValueError("enabled speculative decoding requires draft_model_id")
        speculative["draft_model_content_sha256"] = normalize_sha256(
            str(speculative.get("draft_model_content_sha256") or "")
        )
    manifest["speculative_decoding"] = speculative

    kv_per_token = estimate_kv_bytes_per_token(manifest)
    total_parameters = architecture["total_parameter_count"]
    active_parameters = architecture.get("active_parameter_count_per_token")
    size_bytes = model.get("size_bytes")
    derived = {
        "active_parameter_ratio": (
            float(active_parameters) / float(total_parameters) if active_parameters else None
        ),
        "weight_bytes_per_total_parameter": (
            float(size_bytes) / float(total_parameters) if size_bytes else None
        ),
        "effective_weight_bits_from_artifact": (
            float(size_bytes) * 8.0 / float(total_parameters) if size_bytes else None
        ),
        "kv_bytes_per_token": kv_per_token,
        "estimated_kv_cache_bytes": (
            int(kv_per_token * configured * slots) if kv_per_token is not None else None
        ),
        "comparison_group": comparison_group(manifest),
    }
    manifest["derived"] = derived
    core = identity_core(manifest)
    fingerprint = canonical_hash(core)
    supplied = str(manifest.get("loadout_fingerprint") or "")
    if require_fingerprint and supplied != fingerprint:
        raise ValueError("loadout fingerprint mismatch")
    manifest["loadout_fingerprint"] = fingerprint
    manifest["admission"] = {"admitted": False}
    return manifest


def _set_path(value: Dict[str, Any], dotted: str, replacement: Any) -> None:
    parts = dotted.split(".")
    cursor: Dict[str, Any] = value
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = replacement


def build_matrix(base_manifests: Iterable[Mapping[str, Any]], axes: Mapping[str, Sequence[Any]], *, max_candidates: int = 256) -> Dict[str, Any]:
    normalized_bases = [validate_manifest(item) for item in base_manifests]
    if not normalized_bases:
        raise ValueError("matrix requires at least one base manifest")
    prohibited_axes = {
        "model.content_sha256",
        "model.id",
        "architecture.kind",
        "architecture.total_parameter_count",
        "architecture.active_parameter_count_per_token",
        "weight_quantization.scheme",
        "weight_quantization.nominal_bits",
    }
    bad = sorted(set(axes) & prohibited_axes)
    if bad:
        raise ValueError(
            "model architecture and weight quantization variants require separate base manifests with exact model hashes: "
            + ", ".join(bad)
        )
    for key, values in axes.items():
        if not key or not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"matrix axis {key!r} requires a non-empty array")
    keys = sorted(axes)
    combinations = list(itertools.product(*(axes[key] for key in keys))) if keys else [()]
    if len(combinations) * len(normalized_bases) > max_candidates:
        raise ValueError("matrix exceeds max_candidates")
    candidates: List[Dict[str, Any]] = []
    for base in normalized_bases:
        for combination in combinations:
            candidate = copy.deepcopy(base)
            for key, replacement in zip(keys, combination):
                _set_path(candidate, key, replacement)
            candidate.pop("loadout_fingerprint", None)
            candidate.pop("derived", None)
            candidate.pop("admission", None)
            candidate["candidate_id"] = "pending"
            normalized = validate_manifest(candidate)
            normalized["candidate_id"] = normalized["loadout_fingerprint"].split(":", 1)[1][:20]
            normalized.pop("loadout_fingerprint", None)
            normalized.pop("derived", None)
            normalized.pop("admission", None)
            candidates.append(validate_manifest(normalized))
    fingerprints = [item["loadout_fingerprint"] for item in candidates]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("matrix generated duplicate loadout identities")
    core = {"axes": dict(axes), "candidate_fingerprints": fingerprints}
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "candidate_count": len(candidates),
        "axes": dict(axes),
        "candidates": candidates,
        "matrix_fingerprint": canonical_hash(core),
        "admission": {"admitted": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and expand exact model loadout identities")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--out", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--bases", required=True, help="JSON array or object containing base_manifests")
    matrix.add_argument("--axes", required=True, help="JSON object keyed by dotted loadout paths")
    matrix.add_argument("--max-candidates", type=int, default=256)
    matrix.add_argument("--out", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            raw = load_json(Path(args.manifest))
            if not isinstance(raw, Mapping):
                raise ValueError("manifest must be a JSON object")
            write_json(Path(args.out), validate_manifest(raw))
            return 0
        raw_bases = load_json(Path(args.bases))
        bases = raw_bases.get("base_manifests") if isinstance(raw_bases, Mapping) else raw_bases
        if not isinstance(bases, list):
            raise ValueError("bases must be an array or contain base_manifests")
        axes = load_json(Path(args.axes))
        if not isinstance(axes, Mapping):
            raise ValueError("axes must be a JSON object")
        write_json(Path(args.out), build_matrix(bases, axes, max_candidates=args.max_candidates))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"model loadout failed: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
