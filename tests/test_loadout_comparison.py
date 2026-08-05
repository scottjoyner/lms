from __future__ import annotations

import copy
import json
from pathlib import Path

from lms_agent_bench.loadout_comparison import compare_reports
from lms_agent_bench.model_loadout import validate_manifest


def bases():
    return json.loads(
        Path("examples/model-loadouts.v1.example.json").read_text(encoding="utf-8")
    )["base_manifests"]


def report(loadout, *, tasks_per_hour=60.0, pass_rate=1.0):
    exact = validate_manifest(loadout)
    return {
        "schema_version": "hermes_agent_benchmark.v1",
        "suite_id": "hermes_agent_intelligence.v1",
        "suite_fingerprint": "sha256:" + "a" * 64,
        "dry_run": False,
        "loadout": exact,
        "identity": {
            "loadout_fingerprint": exact["loadout_fingerprint"],
        },
        "aggregate": {
            "overall_task_pass_rate": pass_rate,
            "effect_checkpoint_rate": pass_rate,
            "argument_validity_rate": 1.0,
            "successful_tasks_per_hour": tasks_per_hour,
            "successful_effect_weight_per_minute": 5.0,
            "completion_tokens_per_second_end_to_end": 10.0,
        },
        "gate": {"passed": pass_rate == 1.0},
    }


def write_report(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


def test_single_axis_same_model_comparison_is_causal_candidate(tmp_path):
    first = bases()[0]
    second = copy.deepcopy(first)
    second["context"]["configured_tokens"] = 16_384
    second["kv_cache"]["capacity_tokens"] = 16_384

    left = write_report(tmp_path / "left.json", report(first, tasks_per_hour=60))
    right = write_report(tmp_path / "right.json", report(second, tasks_per_hour=50))
    comparison = compare_reports([left, right])

    pair = comparison["comparisons"][0]
    assert pair["classification"] == "single_axis_same_model"
    assert pair["changed_dimensions"] == ["configured_context_tokens"]
    assert pair["causal_comparison_allowed"] is True
    assert pair["tasks_per_hour_delta_right_minus_left"] == -10


def test_dense_vs_moe_is_observational_and_reports_active_parameter_efficiency(tmp_path):
    dense = write_report(tmp_path / "dense.json", report(bases()[0], tasks_per_hour=60))
    moe = write_report(tmp_path / "moe.json", report(bases()[1], tasks_per_hour=90))
    comparison = compare_reports([dense, moe])

    dense_row, moe_row = comparison["rows"]
    pair = comparison["comparisons"][0]
    assert pair["classification"] == "cross_model_architecture_observational"
    assert pair["causal_comparison_allowed"] is False
    assert dense_row["active_parameter_ratio"] == 1.0
    assert moe_row["active_parameter_ratio"] < 0.1
    assert moe_row["tasks_per_hour_per_active_billion"] > dense_row["tasks_per_hour_per_active_billion"]
    assert comparison["ranking_policy"]["quality_is_not_averaged_with_speed"] is True
