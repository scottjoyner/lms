"""Process-local native Hermes tool-presentation trial: same frozen bridge preamble.

Research only. The original full Hermes preamble is shared via a transient
private file across the two isolated subprocesses, deleted by the parent after
both arms. No raw system message is kept in the evidence archive.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

MODES = frozenset(("bridge", "direct"))
FIRST_GENERATION = {"temperature": 0.0, "seed": 43, "max_tokens": 640}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def adapt_request(original, *, mode: str, pinned: Path):
    if mode not in MODES:
        raise ValueError("unexpected mode")
    if not isinstance(original, dict) or not isinstance(original.get("tools"), list):
        raise ValueError("invalid native tool envelope")
    messages = original.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("missing native model messages")
    system = messages[0]
    if system.get("role") != "system" or not isinstance(system.get("content"), str):
        raise ValueError("expected native system preamble")
    original_content = system["content"]
    if mode == "bridge" and not pinned.exists():
        if len(original_content) < 500:
            raise ValueError("missing full native system preamble")
        descriptor = os.open(str(pinned), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(original_content)
        pinned.chmod(0o600)
    if not pinned.is_file() or pinned.is_symlink() or pinned.stat().st_mode & 0o077:
        raise ValueError("private pinned native preamble unavailable")
    frozen = pinned.read_text(encoding="utf-8")
    if len(frozen) < 500:
        raise ValueError("invalid frozen native preamble")
    if mode == "bridge" and original_content != frozen:
        raise ValueError("native bridge preamble drifted inside arm")
    request = dict(original)
    request["messages"] = [{**system, "content": frozen}, *messages[1:]]
    initial = [item.get("role") for item in messages] == ["system", "user"]
    for key, value in FIRST_GENERATION.items():
        if initial:
            if key in request and request[key] not in (None, value):
                raise ValueError("noncanonical native first-turn decoding")
            request[key] = value
    receipt = {
        "mode": mode, "first_turn": initial,
        "original_native_system_sha256": _digest(original_content),
        "pinned_effective_system_sha256": _digest(frozen),
        "tool_names": [x["function"]["name"] for x in request["tools"]],
        "tool_contract_sha256": _digest(request["tools"]),
        "user_content_sha256": [_digest(m.get("content")) for m in messages
                                if m.get("role") == "user"],
        "generation": {key: request[key] for key in FIRST_GENERATION if key in request},
        "raw_preamble_retained_in_receipt": False,
        "qualification_authorized": False,
    }
    return request, receipt
