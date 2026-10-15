"""Fail-closed native Hermes nine-direct-tool order ablation, scratch research only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CANONICAL = tuple("mcp__lms_benchmark__" + x for x in (
    "graph_schema", "graph_query", "workspace_read", "workspace_write",
    "workspace_patch", "workspace_run_tests", "weather_lookup",
    "send_email", "graph_delete_all",
))
ALPHABETICAL = tuple(sorted(CANONICAL))
MODES = {"canonical": CANONICAL, "alphabetical": ALPHABETICAL}
FIRST_GENERATION = {"temperature": 0.0, "seed": 43, "max_tokens": 640}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def adapt_order(original: dict, *, mode: str, pinned: Path):
    if mode not in MODES:
        raise ValueError("unsupported native tool order")
    if not isinstance(original, dict) or not isinstance(original.get("tools"), list):
        raise ValueError("missing native direct tool list")
    messages = original.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("missing native messages")
    system = messages[0]
    if not isinstance(system, dict) or system.get("role") != "system":
        raise ValueError("unexpected initial model role")
    native = system.get("content")
    if not isinstance(native, str):
        raise ValueError("native preamble not text")
    toolmap = {}
    for item in original["tools"]:
        name = (item.get("function") or {}).get("name")
        if name in toolmap or name not in CANONICAL:
            raise ValueError("duplicate or out-of-scope direct tool")
        toolmap[name] = item
    if len(toolmap) != len(CANONICAL):
        raise ValueError("nine exact MCP tools required")
    if mode == "canonical" and not pinned.exists():
        if len(native) < 500:
            raise ValueError("native preamble unexpectedly short")
        descriptor = os.open(str(pinned), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
            fh.write(native)
    if not pinned.is_file() or pinned.is_symlink() or pinned.stat().st_mode & 0o077:
        raise ValueError("private frozen preamble unavailable")
    frozen = pinned.read_text(encoding="utf-8")
    if len(frozen) < 500 or (mode == "canonical" and native != frozen):
        raise ValueError("native canonical preamble changed or invalid")
    request = dict(original)
    request["messages"] = [{**system, "content": frozen}, *messages[1:]]
    request["tools"] = [toolmap[name] for name in MODES[mode]]
    first = [x.get("role") for x in messages] == ["system", "user"]
    for key, value in FIRST_GENERATION.items():
        if first:
            if key in original and original[key] not in (None, value):
                raise ValueError("first-turn decoding mismatch")
            request[key] = value
    return request, {
        "mode": mode, "first_turn": first,
        "pinned_system_sha256": digest(frozen),
        "native_system_sha256": digest(native),
        "tool_names": list(MODES[mode]),
        "unordered_contract_hash": digest([toolmap[n] for n in sorted(CANONICAL)]),
        "ordered_contract_hash": digest(request["tools"]),
        "user_sha256": [digest(m.get("content")) for m in messages if m.get("role") == "user"],
        "generation": {k: request[k] for k in FIRST_GENERATION if k in request},
        "raw_prompts_saved": False, "release_authorized": False,
    }
