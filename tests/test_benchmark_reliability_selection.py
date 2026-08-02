from __future__ import annotations

from lms_agent_bench.fleet_loadout_entrypoint import select_loadout


CANDIDATE = {
    "candidate_id": "candidate-1",
    "engine": "llama.cpp",
    "backend": "cpu",
    "model": {"id": "model.gguf", "path": "/models/model.gguf"},
    "context_tokens": 4096,
    "parallel_slots": 1,
}
PLAN = {
    "observation_fingerprint": "sha256:" + "1" * 64,
    "plan_fingerprint": "sha256:" + "2" * 64,
    "candidates": [CANDIDATE],
    "rejected_candidates": [],
}


def base_metrics():
    return {
        "candidate_id": "candidate-1",
        "ok_rate": "1.0",
        "eval_score_avg": "1.0",
        "eval_ok_rate": "1.0",
        "tps_med": "20.0",
        "ttft_med": "0.4",
        "memory_headroom_ratio": "0.25",
        "concurrency_ok": "true",
        "streaming_ok": "true",
        "cancellation_ok": "true",
        "crash_count": "0",
        "benchmark_exit_code": "0",
    }


def reliable_metrics():
    return {
        **base_metrics(),
        "reliability_pass": "true",
        "reliability_score": "0.98",
        "reliability_fingerprint": "sha256:" + "3" * 64,
        "valid_trials": "3",
        "required_trials": "3",
        "trial_attempts": "3",
        "trial_retry_rate": "0.0",
        "sample_completeness": "1.0",
        "success_wilson_lower_95": "0.90",
        "trial_tps_cv": "0.03",
        "trial_ttft_cv": "0.04",
        "tps_relative_mad": "0.02",
        "ttft_relative_mad": "0.02",
    }


def test_selection_rejects_legacy_row_without_reliability_evidence():
    artifact = select_loadout(PLAN, [base_metrics()])
    assert artifact["selected"] is None
    ranked = artifact["ranked_results"][0]
    assert ranked["eligible"] is False
    assert ranked["gates"]["measurement_reliability"] is False
    assert "measurement_reliability_gate_failed" in ranked["hard_failures"]
    assert "reliability_fingerprint_missing_or_invalid" in ranked["hard_failures"]


def test_selection_accepts_stable_complete_three_trial_evidence():
    artifact = select_loadout(PLAN, [reliable_metrics()])
    assert artifact["selected"]["candidate_id"] == "candidate-1"
    assert artifact["selected"]["eligible"] is True
    assert artifact["selected"]["gates"]["measurement_reliability"] is True
    assert artifact["admission"]["admitted"] is False


def test_selection_rejects_high_variance_even_when_flag_claims_pass():
    metrics = reliable_metrics()
    metrics["trial_tps_cv"] = "0.50"
    artifact = select_loadout(PLAN, [metrics])
    assert artifact["selected"] is None
    ranked = artifact["ranked_results"][0]
    assert "throughput_trial_cv_above_0.20" in ranked["hard_failures"]
    assert ranked["gates"]["measurement_reliability"] is False


def test_selection_rejects_incomplete_measurement_matrix():
    metrics = reliable_metrics()
    metrics["sample_completeness"] = "0.99"
    artifact = select_loadout(PLAN, [metrics])
    assert artifact["selected"] is None
    assert "incomplete_measurement_matrix" in artifact["ranked_results"][0][
        "hard_failures"
    ]
