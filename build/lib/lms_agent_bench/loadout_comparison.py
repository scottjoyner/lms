#!/usr/bin/env python3
"""Compare exact-loadout Hermes reports without collapsing quality and speed."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lms_agent_bench.hermes_agent_common import SCHEMA_VERSION as HERMES_SCHEMA
from lms_agent_bench.hermes_agent_common import canonical_hash, load_json, utc_now_iso, write_json
from lms_agent_bench.model_loadout import validate_manifest

SCHEMA_VERSION = "model_loadout_comparison.v1"


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def report_row(report: Mapping[str, Any], source: str) -> Dict[str, Any]:
    if report.get("schema_version") != HERMES_SCHEMA:
        raise ValueError(f"unsupported Hermes report schema in {source}")
    raw_loadout = report.get("loadout")
    if not isinstance(raw_loadout, Mapping):
        raise ValueError(f"Hermes report has no exact loadout in {source}")
    loadout = validate_manifest(raw_loadout, require_fingerprint=True)
    identity = report.get("identity") if isinstance(report.get("identity"), Mapping) else {}
    if identity.get("loadout_fingerprint") != loadout["loadout_fingerprint"]:
        raise ValueError(f"Hermes report/loadout fingerprint mismatch in {source}")
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), Mapping) else {}
    gate = report.get("gate") if isinstance(report.get("gate"), Mapping) else {}
    total = float(loadout["architecture"]["total_parameter_count"])
    active = float(loadout["architecture"].get("active_parameter_count_per_token") or total)
    size_bytes = loadout["model"].get("size_bytes")
    model_gib = float(size_bytes) / (1024.0**3) if size_bytes else None
    tasks_per_hour = float(aggregate.get("successful_tasks_per_hour") or 0.0)
    effect_per_minute = float(aggregate.get("successful_effect_weight_per_minute") or 0.0)
    completion_tps = aggregate.get("completion_tokens_per_second_end_to_end")
    row = {
        "source": source,
        "qualified": gate.get("passed") is True and report.get("dry_run") is False,
        "suite_id": report.get("suite_id"),
        "suite_fingerprint": report.get("suite_fingerprint"),
        "node_id": loadout["node_id"],
        "candidate_id": loadout["candidate_id"],
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "model_id": loadout["model"]["id"],
        "model_content_sha256": loadout["model"]["content_sha256"],
        "architecture_kind": loadout["architecture"]["kind"],
        "total_parameters": int(total),
        "active_parameters_per_token": int(active),
        "active_parameter_ratio": active / total,
        "expert_count_total": loadout["architecture"].get("expert_count_total"),
        "expert_count_active_per_token": loadout["architecture"].get("expert_count_active_per_token"),
        "weight_quantization": loadout["weight_quantization"].get("scheme"),
        "effective_weight_bits_from_artifact": loadout["derived"].get("effective_weight_bits_from_artifact"),
        "configured_context_tokens": loadout["context"]["configured_tokens"],
        "kv_key_dtype": loadout["kv_cache"]["key_dtype"],
        "kv_value_dtype": loadout["kv_cache"]["value_dtype"],
        "kv_location": loadout["kv_cache"]["location"],
        "kv_prefix_reuse": loadout["kv_cache"]["prefix_reuse"],
        "kv_shared_across_requests": loadout["kv_cache"]["shared_across_requests"],
        "parallel_slots": loadout["concurrency"]["parallel_slots"],
        "engine": loadout["runtime"]["engine"],
        "backend": loadout["runtime"]["backend"],
        "flash_attention": loadout["runtime"].get("flash_attention"),
        "batch_size": loadout["runtime"].get("batch_size"),
        "ubatch_size": loadout["runtime"].get("ubatch_size"),
        "gpu_layers": loadout["runtime"].get("gpu_layers"),
        "model_size_gib": model_gib,
        "overall_task_pass_rate": aggregate.get("overall_task_pass_rate"),
        "effect_checkpoint_rate": aggregate.get("effect_checkpoint_rate"),
        "argument_validity_rate": aggregate.get("argument_validity_rate"),
        "successful_tasks_per_hour": tasks_per_hour,
        "successful_effect_weight_per_minute": effect_per_minute,
        "completion_tokens_per_second_end_to_end": completion_tps,
        "tasks_per_hour_per_active_billion": tasks_per_hour / (active / 1e9),
        "tasks_per_hour_per_total_billion": tasks_per_hour / (total / 1e9),
        "tasks_per_hour_per_model_gib": tasks_per_hour / model_gib if model_gib else None,
        "effect_weight_per_minute_per_active_billion": effect_per_minute / (active / 1e9),
    }
    return row


def dimensions(row: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "model_content_sha256",
        "architecture_kind",
        "total_parameters",
        "active_parameters_per_token",
        "weight_quantization",
        "configured_context_tokens",
        "kv_key_dtype",
        "kv_value_dtype",
        "kv_location",
        "kv_prefix_reuse",
        "kv_shared_across_requests",
        "parallel_slots",
        "engine",
        "backend",
        "flash_attention",
        "batch_size",
        "ubatch_size",
        "gpu_layers",
    )
    return {key: row.get(key) for key in keys}


def pairwise(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    for left, right in itertools.combinations(rows, 2):
        left_dimensions = dimensions(left)
        right_dimensions = dimensions(right)
        changed = [key for key in left_dimensions if left_dimensions[key] != right_dimensions[key]]
        same_model = left["model_content_sha256"] == right["model_content_sha256"]
        if same_model and len(changed) == 1:
            classification = "single_axis_same_model"
        elif same_model:
            classification = "multi_axis_same_model"
        elif left["architecture_kind"] != right["architecture_kind"]:
            classification = "cross_model_architecture_observational"
        else:
            classification = "cross_model_variant_observational"
        comparisons.append(
            {
                "left_loadout_fingerprint": left["loadout_fingerprint"],
                "right_loadout_fingerprint": right["loadout_fingerprint"],
                "same_node": left["node_id"] == right["node_id"],
                "same_suite": left["suite_fingerprint"] == right["suite_fingerprint"],
                "changed_dimensions": changed,
                "classification": classification,
                "causal_comparison_allowed": (
                    classification == "single_axis_same_model"
                    and left["node_id"] == right["node_id"]
                    and left["suite_fingerprint"] == right["suite_fingerprint"]
                ),
                "task_pass_rate_delta_right_minus_left": (
                    float(right.get("overall_task_pass_rate") or 0.0)
                    - float(left.get("overall_task_pass_rate") or 0.0)
                ),
                "tasks_per_hour_delta_right_minus_left": (
                    float(right.get("successful_tasks_per_hour") or 0.0)
                    - float(left.get("successful_tasks_per_hour") or 0.0)
                ),
            }
        )
    return comparisons


def compare_reports(paths: Sequence[str]) -> Dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two Hermes reports")
    rows = []
    for value in paths:
        report = load_json(Path(value))
        if not isinstance(report, Mapping):
            raise ValueError(f"report is not an object: {value}")
        rows.append(report_row(report, value))
    core = {
        "rows": rows,
        "comparisons": pairwise(rows),
        "ranking_policy": {
            "quality_is_not_averaged_with_speed": True,
            "qualified_rows_only_for_operational_ranking": True,
            "normalized_architecture_metrics_are_explanatory_only": True,
            "cross_model_architecture_comparisons_are_observational": True,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "model_loadout_comparison",
        "created_at_utc": utc_now_iso(),
        **core,
        "comparison_fingerprint": canonical_hash(core),
        "admission": {"admitted": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare exact-loadout Hermes benchmark reports")
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        write_json(Path(args.out), compare_reports(args.report))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"loadout comparison failed: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
