"""Strict one-variable native Hermes model-request ablation for isolated research.

Never enabled in production or normal benchmarks. Fixed generation settings
are identical across both paired conditions; only the first system message
may change. No prompt content or HTTP headers are retained in receipts.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


GENERATION = {"temperature": 0.0, "max_tokens": 640, "seed": 43}
MODES = frozenset({"control", "minimal"})


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_ablation_request(
    original: dict[str, Any], *, mode: str, canonical_system: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make a local copy, preserving all non-system wire fields verbatim.

    The returned receipt is intentionally metadata-only. The harness must
    compare BOTH first-turn wire captures before interpreting any outcome.
    """
    if mode not in MODES:
        raise ValueError("unsupported native research control")
    if (not isinstance(canonical_system, str)
            or not 40 <= len(canonical_system) <= 250):
        raise ValueError("expected fixed short benchmark case system instruction")
    if not isinstance(original, dict):
        raise ValueError("model request must be object")
    if "model" not in original or not isinstance(original.get("tools"), list):
        raise ValueError("native model and tools required")
    messages = original.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("native system and task message required")
    first = messages[0]
    if (not isinstance(first, dict) or first.get("role") != "system"
            or not isinstance(first.get("content"), str)):
        raise ValueError("unexpected initial system message format")
    if (not any(isinstance(x, dict) and x.get("role") == "user"
                for x in messages[1:])):
        raise ValueError("user task absent")
    original_content = first["content"]
    if len(original_content) < 500:
        raise ValueError("no substantial Hermes system preamble present")
    request = dict(original)
    request["messages"] = list(messages)
    if mode == "minimal":
        request["messages"][0] = {**first, "content": canonical_system}
    first_turn = [m.get("role") for m in messages] == ["system", "user"]
    adapted_fields = []
    for key, value in GENERATION.items():
        if key in original and original[key] not in (None, value):
            if first_turn:
                raise ValueError("inconsistent initial model sampling constraint")
            # Native Hermes may intentionally adapt its sampling after errors.
            # The A/B intervention changes only initial system text; never
            # suppress native decoding policy on later turns.
            adapted_fields.append(key)
            continue
        request[key] = value
    # Never mutate the caller's message history or alter tool schemas.
    assert request["tools"] is original["tools"]
    assert request["messages"][1:] == original["messages"][1:]
    return request, {
        "schema": "lms-hermes-native-preamble-ablation.v1",
        "mode": mode,
        "no_production_effects_authorized": True,
        "original_system_chars": len(original_content),
        "original_system_sha256": _digest(original_content),
        "effective_system_chars": len(request["messages"][0]["content"]),
        "effective_system_sha256": _digest(request["messages"][0]["content"]),
        "model_sha256": _digest(request["model"]),
        "tool_list_sha256": _digest(request["tools"]),
        "user_message_sha256": [_digest(x.get("content")) for x in messages
                                if isinstance(x, dict) and x.get("role") == "user"],
        "other_roles": [x.get("role") for x in messages if isinstance(x, dict)],
        "generation": {key: request[key] for key in GENERATION
                       if key in request},
        "initial_turn": first_turn,
        "later_turn_native_adaptations": sorted(adapted_fields),
        "raw_prompts_saved": False,
        "raw_credentials_saved": False,
        "qualification_authorized": False,
    }
