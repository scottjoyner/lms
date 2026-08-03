"""Installed entrypoint for fair, conservative fleet loadout planning.

The historical planner remains the candidate generator. This wrapper improves
its KV-cache estimate, fairly truncates the matrix, and requires cryptographic
multi-trial measurement reliability before a benchmark row can be selected.
"""
from __future__ import annotations

import math
import re
import sys
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from lms_agent_bench import fleet_loadout as _base

_ORIGINAL_BUILD_PLAN = _base.build_plan
_ORIGINAL_SELECT_LOADOUT = _base.select_loadout
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def kv_bytes_per_token(model: Mapping[str, Any]) -> int:
    explicit = _base.int_or_none(model.get("kv_bytes_per_token"))
    if explicit and explicit > 0:
        return explicit

    layers = _base.int_or_none(
        model.get("num_hidden_layers") or model.get("n_layers")
    )
    kv_heads = _base.int_or_none(
        model.get("num_key_value_heads") or model.get("n_kv_heads")
    )
    head_dim = _base.int_or_none(model.get("head_dim"))
    element_bytes = _base.int_or_none(model.get("kv_element_bytes")) or 2
    if layers and kv_heads and head_dim:
        return max(1, 2 * layers * kv_heads * head_dim * element_bytes)

    params = float(model.get("parameter_billions") or 1.0)
    heuristic = int(math.sqrt(max(params, 0.01)) * 64 * 1024)
    return max(64 * 1024, min(1024 * 1024, heuristic))


def _group_key(candidate: Mapping[str, Any]) -> Tuple[str, str]:
    model = candidate.get("model", {})
    return str(model.get("id") or "unknown"), str(candidate.get("backend") or "cpu")


def fair_candidate_sample(
    candidates: Sequence[Mapping[str, Any]], max_candidates: int
) -> List[Dict[str, Any]]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    groups: "OrderedDict[Tuple[str, str], Deque[Dict[str, Any]]]" = OrderedDict()
    for raw in candidates:
        candidate = dict(raw)
        groups.setdefault(_group_key(candidate), deque()).append(candidate)

    selected: List[Dict[str, Any]] = []
    while len(selected) < max_candidates:
        progress = False
        for queue in groups.values():
            if queue and len(selected) < max_candidates:
                selected.append(queue.popleft())
                progress = True
        if not progress:
            break
    return selected


def build_plan(
    observation: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    contexts: Sequence[int] = _base.DEFAULT_CONTEXTS,
    max_candidates: int = 96,
) -> Dict[str, Any]:
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    _base.kv_bytes_per_token = kv_bytes_per_token

    backend_count = max(
        1,
        len(observation.get("hardware", {}).get("supported_backends", []) or ["cpu"]),
    )
    full_limit = max(
        max_candidates,
        len(models) * backend_count * max(1, len(contexts)) * 16 + len(models),
    )
    full_plan = _ORIGINAL_BUILD_PLAN(
        observation,
        models,
        contexts=contexts,
        max_candidates=full_limit,
    )
    sampled = fair_candidate_sample(full_plan.get("candidates", []), max_candidates)
    plan_core = {
        "observation_fingerprint": observation.get("observation_fingerprint"),
        "candidates": sampled,
        "rejected_candidates": full_plan.get("rejected_candidates", []),
    }
    benchmark_contract = dict(full_plan.get("benchmark_contract") or {})
    required_metrics = list(benchmark_contract.get("required_metrics") or [])
    for metric in (
        "reliability_pass",
        "reliability_fingerprint",
        "valid_trials",
        "sample_completeness",
        "success_wilson_lower_95",
        "trial_retry_rate",
        "trial_tps_cv",
        "trial_ttft_cv",
        "tps_relative_mad",
        "ttft_relative_mad",
    ):
        if metric not in required_metrics:
            required_metrics.append(metric)
    required_gates = list(benchmark_contract.get("required_gates") or [])
    if "measurement_reliability" not in required_gates:
        required_gates.append("measurement_reliability")
    benchmark_contract.update(
        {
            "required_metrics": required_metrics,
            "required_gates": required_gates,
            "minimum_valid_trials": 3,
            "minimum_sample_completeness": 1.0,
            "reliability_report_required": True,
        }
    )
    full_plan.update(
        {
            **plan_core,
            "plan_fingerprint": _base.canonical_hash(plan_core),
            "benchmark_contract": benchmark_contract,
            "planning_policy": {
                "candidate_limit": max_candidates,
                "generated_candidate_count": len(full_plan.get("candidates", [])),
                "published_candidate_count": len(sampled),
                "truncation": "round_robin_by_model_and_backend",
                "kv_estimator": "explicit_metadata_or_architecture_or_sqrt_parameter_heuristic",
            },
        }
    )
    return full_plan


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _reliability_failures(metrics: Mapping[str, Any]) -> List[str]:
    failures: List[str] = []
    if not _base.bool_value(metrics.get("reliability_pass")):
        failures.append("measurement_reliability_gate_failed")
    fingerprint = str(metrics.get("reliability_fingerprint") or "").lower()
    if not _SHA256_RE.fullmatch(fingerprint):
        failures.append("reliability_fingerprint_missing_or_invalid")
    if _int_value(metrics.get("valid_trials")) < 3:
        failures.append("fewer_than_three_valid_trials")
    if _base.float_value(metrics.get("sample_completeness")) < 1.0:
        failures.append("incomplete_measurement_matrix")
    if _base.float_value(metrics.get("trial_retry_rate"), 1.0) > 0.25:
        failures.append("trial_retry_rate_above_0.25")
    if _base.float_value(metrics.get("trial_tps_cv"), 1.0) > 0.20:
        failures.append("throughput_trial_cv_above_0.20")
    if _base.float_value(metrics.get("trial_ttft_cv"), 1.0) > 0.35:
        failures.append("ttft_trial_cv_above_0.35")
    if _base.float_value(metrics.get("tps_relative_mad"), 1.0) > 0.25:
        failures.append("throughput_relative_mad_above_0.25")
    if _base.float_value(metrics.get("ttft_relative_mad"), 1.0) > 0.25:
        failures.append("ttft_relative_mad_above_0.25")
    if _int_value(metrics.get("benchmark_exit_code"), default=1) != 0:
        failures.append("benchmark_process_failed")
    return failures


def select_loadout(
    plan: Mapping[str, Any], results: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    artifact = _ORIGINAL_SELECT_LOADOUT(plan, results)
    ranked = list(artifact.get("ranked_results") or [])
    for item in ranked:
        metrics = item.get("metrics")
        if not isinstance(metrics, Mapping):
            reliability_failures = ["benchmark_metrics_missing"]
        else:
            reliability_failures = _reliability_failures(metrics)
        hard_failures = list(item.get("hard_failures") or [])
        for failure in reliability_failures:
            if failure not in hard_failures:
                hard_failures.append(failure)
        item["hard_failures"] = hard_failures
        item["eligible"] = not hard_failures
        gates = dict(item.get("gates") or {})
        gates["measurement_reliability"] = not reliability_failures
        item["gates"] = gates
        reliability_score = (
            _base.float_value(metrics.get("reliability_score"))
            if isinstance(metrics, Mapping)
            else 0.0
        )
        item["score"] = round(
            float(item.get("score") or 0.0) * (0.90 + 0.10 * reliability_score),
            6,
        )

    ranked.sort(key=lambda item: (item["eligible"], item["score"]), reverse=True)
    selected = next((item for item in ranked if item["eligible"]), None)
    fallback = next(
        (
            item
            for item in ranked
            if item["eligible"]
            and selected
            and item["candidate_id"] != selected["candidate_id"]
        ),
        None,
    )
    artifact["ranked_results"] = ranked
    artifact["selected"] = selected
    artifact["fallback"] = fallback
    artifact["selection_fingerprint"] = _base.canonical_hash(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"created_at_utc", "selection_fingerprint"}
        }
    )
    return artifact


def main(argv: Optional[List[str]] = None) -> int:
    _base.kv_bytes_per_token = kv_bytes_per_token
    _base.build_plan = build_plan
    _base.select_loadout = select_loadout
    return _base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
