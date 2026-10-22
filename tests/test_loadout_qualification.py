from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lms_agent_bench.hermes_agent_common import canonical_hash
from lms_agent_bench.loadout_qualification import (
    BASE_HERMES_SUITE_ID,
    CONTEXT_HERMES_SUITE_ID,
    build_qualification,
    build_throughput_evidence,
    verify_qualification,
)
from lms_agent_bench.model_loadout import validate_manifest


def example_loadout():
    payload = json.loads(
        Path("examples/model-loadouts.v1.example.json").read_text(encoding="utf-8")
    )
    return validate_manifest(payload["base_manifests"][0])


def reliable_hash(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def reliability_report(loadout):
    summary = {
        "host_name": loadout["node_id"],
        "base_url": "http://127.0.0.1:8080/v1",
        "model_key": loadout["model"]["id"],
        "tps_med": "42.500",
        "tps_p10": "40.000",
        "tps_p90": "45.000",
        "ttft_med": "0.125",
        "ttft_p90": "0.200",
        "ok_rate": "1.000000",
        "eval_ok_rate": "1.000000",
        "eval_score_avg": "0.950000",
        "sample_completeness": "1.000000",
        "reliability_score": "0.990000",
        "valid_trials": 3,
        "trial_attempts": 3,
        "reliability_pass": True,
        "reliability_failures": [],
    }
    core = {
        "schema_version": "reliable_benchmark.v1",
        "artifact_type": "benchmark_reliability",
        "input_fingerprint": "sha256:" + "1" * 64,
        "input": {},
        "suite_id": "agent_skill_suite.v1",
        "suite_version": 1,
        "requested_trials": 3,
        "valid_trials": 3,
        "trial_attempts": 3,
        "passed": True,
        "preflight": {},
        "summaries": [summary],
        "trial_manifests": [],
        "admission": {"admitted": False},
    }
    return {
        **core,
        "created_at_utc": "2026-08-04T00:00:00+00:00",
        "reliability_fingerprint": reliable_hash(core),
    }


def hermes_report(loadout, suite_id):
    identity = {
        "node_id": loadout["node_id"],
        "candidate_id": loadout["candidate_id"],
        "model_id": loadout["model"]["id"],
        "model_content_sha256": loadout["model"]["content_sha256"],
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "architecture_kind": loadout["architecture"]["kind"],
        "total_parameter_count": loadout["architecture"]["total_parameter_count"],
        "active_parameter_count_per_token": loadout["architecture"].get(
            "active_parameter_count_per_token"
        ),
        "weight_quantization": loadout["weight_quantization"],
        "configured_context_tokens": loadout["context"]["configured_tokens"],
        "kv_cache": loadout["kv_cache"],
        "parallel_slots": loadout["concurrency"]["parallel_slots"],
        "runtime": loadout["runtime"],
        "endpoint": "http://127.0.0.1:8080/v1",
        "loopback_only": True,
    }
    gate = {
        "passed": True,
        "policy": {},
        "failures": [],
        "intelligence_qualified": True,
        "admission": {"admitted": False},
    }
    aggregate = {
        "case_count": 1,
        "attempted_trial_count": 3,
        "valid_trial_count": 3,
        "passed_trial_count": 3,
        "overall_task_pass_rate": 1.0,
        "effect_checkpoint_rate": 1.0,
        "argument_validity_rate": 1.0,
        "timeout_or_crash_rate": 0.0,
        "successful_tasks_per_hour": 120.0,
        "successful_effect_weight_per_minute": 2.0,
        "tool_calls_per_minute": 4.0,
        "completion_tokens_per_second_end_to_end": 12.5,
        "total_wall_seconds": 90.0,
        "cases": [
            {
                "case_key": "case-a",
                "priority": "P0",
                "task_family": "coding",
                "recovery_case": False,
                "attempted_trials": 3,
                "valid_trials": 3,
                "passed_trials": 3,
                "trial_pass_rate": 1.0,
                "reliability_complete": True,
                "median_wall_seconds": 30.0,
                "p95_wall_seconds": 32.0,
            }
        ],
    }
    core = {
        "identity": identity,
        "suite_id": suite_id,
        "suite_fingerprint": canonical_hash({"suite_id": suite_id}),
        "trials_per_case": 3,
        "trials": [{"valid": True}],
        "aggregate": aggregate,
        "gate": gate,
        "dry_run": False,
        "admission": {"admitted": False},
    }
    return {
        "schema_version": "hermes_agent_benchmark.v1",
        "artifact_type": "hermes_agent_intelligence_benchmark",
        "created_at_utc": "2026-08-04T00:00:00+00:00",
        "run_id": suite_id,
        "loadout": loadout,
        **core,
        "benchmark_fingerprint": canonical_hash(core),
    }


def test_combined_qualification_requires_one_exact_loadout():
    loadout = example_loadout()
    throughput = build_throughput_evidence(loadout, reliability_report(loadout))
    qualification = build_qualification(
        loadout,
        throughput,
        hermes_report(loadout, BASE_HERMES_SUITE_ID),
        hermes_report(loadout, CONTEXT_HERMES_SUITE_ID),
    )
    verified = verify_qualification(qualification, loadout)

    assert qualification["qualified"] is True
    assert qualification["admission"]["admitted"] is False
    assert verified["identity"]["loadout_fingerprint"] == loadout["loadout_fingerprint"]
    assert all(qualification["gates"].values())
    metrics = qualification["decision_metrics"]
    assert metrics["performance"]["observed_tokens_per_second_median"] == 42.5
    assert metrics["performance"]["time_to_first_token_seconds_median"] == 0.125
    assert metrics["performance"]["prompt_processing_tokens_per_second"] is None
    assert metrics["performance"]["peak_device_memory_bytes"] is None
    assert metrics["capability"]["base_hermes"]["overall_task_pass_rate"] == 1.0
    assert metrics["capability"]["base_hermes"]["cases"][0]["task_family"] == "coding"
    assert verified["decision_metrics"] == metrics


def test_throughput_binding_rejects_wrong_node_or_model():
    loadout = example_loadout()
    report = reliability_report(loadout)
    report["summaries"][0]["host_name"] = "wrong-node"
    core = {
        key: value
        for key, value in report.items()
        if key not in {"created_at_utc", "reliability_fingerprint"}
    }
    report["reliability_fingerprint"] = reliable_hash(core)
    with pytest.raises(ValueError, match="node"):
        build_throughput_evidence(loadout, report)


def test_context_report_cannot_be_replaced_by_base_suite():
    loadout = example_loadout()
    throughput = build_throughput_evidence(loadout, reliability_report(loadout))
    base = hermes_report(loadout, BASE_HERMES_SUITE_ID)
    with pytest.raises(ValueError, match="unexpected Hermes suite"):
        build_qualification(loadout, throughput, base, base)


def test_mismatched_loadout_fingerprint_is_rejected():
    loadout = example_loadout()
    throughput = build_throughput_evidence(loadout, reliability_report(loadout))
    changed = copy.deepcopy(loadout)
    changed.pop("loadout_fingerprint")
    changed.pop("derived")
    changed.pop("admission")
    changed["kv_cache"]["key_dtype"] = "q4_0"
    changed = validate_manifest(changed)
    with pytest.raises(ValueError, match="different loadout"):
        build_qualification(
            changed,
            throughput,
            hermes_report(changed, BASE_HERMES_SUITE_ID),
            hermes_report(changed, CONTEXT_HERMES_SUITE_ID),
        )


def test_tampered_qualification_fails_verification():
    loadout = example_loadout()
    throughput = build_throughput_evidence(loadout, reliability_report(loadout))
    qualification = build_qualification(
        loadout,
        throughput,
        hermes_report(loadout, BASE_HERMES_SUITE_ID),
        hermes_report(loadout, CONTEXT_HERMES_SUITE_ID),
    )
    qualification["gates"]["loopback_only"] = False
    with pytest.raises(ValueError, match="gates failed"):
        verify_qualification(qualification)


def test_dry_run_hermes_evidence_is_rejected():
    loadout = example_loadout()
    throughput = build_throughput_evidence(loadout, reliability_report(loadout))
    base = hermes_report(loadout, BASE_HERMES_SUITE_ID)
    base["dry_run"] = True
    core = {
        key: base[key]
        for key in (
            "identity",
            "suite_id",
            "suite_fingerprint",
            "trials_per_case",
            "trials",
            "aggregate",
            "gate",
            "dry_run",
            "admission",
        )
    }
    base["benchmark_fingerprint"] = canonical_hash(core)
    with pytest.raises(ValueError, match="dry-run"):
        build_qualification(
            loadout,
            throughput,
            base,
            hermes_report(loadout, CONTEXT_HERMES_SUITE_ID),
        )


def test_decision_metrics_rejects_fabricated_unmeasured_memory():
    loadout = example_loadout()
    throughput = build_throughput_evidence(loadout, reliability_report(loadout))
    qualification = build_qualification(
        loadout,
        throughput,
        hermes_report(loadout, BASE_HERMES_SUITE_ID),
        hermes_report(loadout, CONTEXT_HERMES_SUITE_ID),
    )
    qualification["decision_metrics"]["performance"]["peak_device_memory_bytes"] = 123
    core = {
        key: value
        for key, value in qualification.items()
        if key not in {"created_at_utc", "qualification_fingerprint"}
    }
    qualification["qualification_fingerprint"] = canonical_hash(core)
    with pytest.raises(ValueError, match="peak_device_memory_bytes"):
        verify_qualification(qualification)
