"""Research-only one-variable graph_query description A/B for native Hermes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from lms_agent_bench.hermes_native_tool_order import CANONICAL, FIRST_GENERATION

MODES = ("baseline", "example")
EXAMPLE = (
    "Read-only owner lookup on isolated research Neo4j; supply read_only=true; "
    "parameters unsupported. For a named service, bind the case-sensitive "
    "Service label and filter by its name. An approved query shape is: "
    "MATCH (s:Service {name: 'router-api'}) RETURN s.owner AS owner. "
    "This is a query example only; execute it against the independent "
    "source graph to determine the actual owner. Never infer owner values."
)
ORIGINAL = ("Read-only owner lookup on isolated research Neo4j; "
            "supply read_only=true; parameters unsupported.")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def adapt_description(original: dict, *, mode: str, pinned: Path):
    if mode not in MODES:
        raise ValueError("research-only description mode invalid")
    messages = original.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("native first system and user required")
    system = messages[0]
    if not isinstance(system, dict) or system.get("role") != "system":
        raise ValueError("first role must be native system")
    native = system.get("content")
    if not isinstance(native, str):
        raise ValueError("expected string system content")
    tools = original.get("tools")
    if not isinstance(tools, list) or len(tools) != 9:
        raise ValueError("exact nine-tool model surface required")
    names = [((item.get("function") or {}).get("name")) for item in tools]
    if len(names) != 9 or set(names) != set(CANONICAL):
        raise ValueError("nine canonical model tools required")
    if mode == "baseline" and not pinned.exists():
        if len(native) < 500:
            raise ValueError("native preamble missing")
        fd = os.open(str(pinned), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(native)
    if not pinned.is_file() or pinned.is_symlink() or pinned.stat().st_mode & 0o077:
        raise ValueError("private pinned system preamble missing")
    frozen = pinned.read_text(encoding="utf-8")
    if len(frozen) < 500 or (mode == "baseline" and native != frozen):
        raise ValueError("native system preamble drift")
    catalog = {}
    for tool in tools:
        name = tool["function"]["name"]
        catalog[name] = {**tool, "function": dict(tool["function"])}
    query_name = "mcp__lms_benchmark__graph_query"
    if catalog[query_name]["function"].get("description") != ORIGINAL:
        raise ValueError("unrecognized original graph-query tool description")
    if mode == "example":
        catalog[query_name]["function"]["description"] = EXAMPLE
    request = dict(original)
    request["messages"] = [{**system, "content": frozen}, *messages[1:]]
    request["tools"] = [catalog[name] for name in CANONICAL]
    first = [m.get("role") for m in messages] == ["system", "user"]
    if first:
        for key, value in FIRST_GENERATION.items():
            if key in original and original[key] not in (None, value):
                raise ValueError("first-turn generation mismatch")
            request[key] = value
    return request, {
        "mode": mode, "first_turn": first,
        "pinned_system_sha256": digest(frozen),
        "description_sha256": digest(catalog[query_name]["function"]["description"]),
        "model_visible_names": list(CANONICAL),
        "user_sha256": [digest(x.get("content")) for x in messages if x.get("role") == "user"],
        "generation": {k: request[k] for k in FIRST_GENERATION if k in request},
        "release_authorized": False, "raw_prompts_saved": False,
    }
