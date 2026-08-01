"""Installed entrypoint for fair, conservative fleet loadout planning.

The historical planner remains the candidate generator. This wrapper improves
its default KV-cache estimate and fairly truncates the generated matrix so one
model or backend cannot consume the entire candidate limit.
"""
from __future__ import annotations

import math
import sys
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from lms_agent_bench import fleet_loadout as _base

_ORIGINAL_BUILD_PLAN = _base.build_plan


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
        # K and V tensors per layer, using the configured cache element width.
        return max(1, 2 * layers * kv_heads * head_dim * element_bytes)

    params = float(model.get("parameter_billions") or 1.0)
    # Layer count and GQA width grow much more slowly than parameter count.
    # sqrt(params) tracks common 1B-70B architectures conservatively without
    # rejecting viable 32 GB and unified-memory candidates by default.
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
    # The base planner has an early-stop limit. Ask it for the complete practical
    # matrix, then apply a fair bounded sample for the published plan.
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
    full_plan.update(
        {
            **plan_core,
            "plan_fingerprint": _base.canonical_hash(plan_core),
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


def main(argv: Optional[List[str]] = None) -> int:
    _base.kv_bytes_per_token = kv_bytes_per_token
    _base.build_plan = build_plan
    return _base.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
