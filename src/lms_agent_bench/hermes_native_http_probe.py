#!/usr/bin/env python3
"""Read-only HTTP request evidence for explicit native Hermes research trials.

The outgoing HTTP request is passed unchanged to the original httpx method.
Only request hashes and tool-result hashes are retained; no prompts or API keys.
A 2xx proves receipt by the local inference HTTP server, not model attention.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _tool_messages(payload):
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    result = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content, tool_id = message.get("content"), message.get("tool_call_id")
        if isinstance(content, str) and isinstance(tool_id, str):
            result.append({
                "tool_call_id": tool_id,
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            })
    return result


def _request_surface(body):
    """Observe the actual model-facing envelope, excluding all message bodies.

    Stable field-level hashes make bridge-vs-direct comparison reproducible
    without storing the user's prompt, assistant responses or API headers.
    """
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise ValueError("invalid outbound model request shape")
    def content_digest(message):
        raw = json.dumps(message.get("content"), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest(), len(raw)
    roles = []
    system, user = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("invalid model message")
        role = message.get("role")
        roles.append(role)
        if role == "system":
            system.append(content_digest(message))
        elif role == "user":
            user.append(content_digest(message))
    tool_contracts = []
    for item in tools:
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            raise ValueError("invalid tool definition")
        function = item["function"]
        contract = function.get("parameters") or {}
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        tool_contracts.append({
            "name": function.get("name"),
            "parameters_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            "description_sha256": hashlib.sha256(str(function.get("description", "")).encode()).hexdigest(),
            "property_keys": sorted((contract.get("properties") or {}).keys()),
        })
    return {
        "advertised_tool_contracts": tool_contracts,
        "advertised_tool_count": len(tool_contracts),
        "model_message_roles": roles,
        "system_content_sha256": [x[0] for x in system],
        "system_content_chars": [x[1] for x in system],
        "user_content_sha256": [x[0] for x in user],
        "generation": {k: body[k] for k in (
            "stream", "temperature", "seed", "max_tokens", "tool_choice",
        ) if k in body},
        "all_other_options_sha256": hashlib.sha256(json.dumps(
            {k: v for k, v in body.items() if k not in ("messages", "tools")},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest(),
    }


@contextmanager
def observe_loopback_http(path: Path, *, endpoint: str) -> Iterator[None]:
    """Record successful HTTP sends containing exact model-facing tool messages.

    The scope must be one single-agent isolated trial. Never install globally
    for any production process. Restores httpx.Client.send even on exception.
    """
    import httpx

    from .hermes_agent_common import require_loopback_endpoint

    base = require_loopback_endpoint(endpoint).rstrip("/")
    original_send = httpx.Client.send
    lock = threading.Lock()

    def observed(client, request, *args, **kwargs):
        url = str(request.url)
        candidate = url.startswith(base + "/") and request.url.path.rstrip("/").endswith(
            "/chat/completions"
        )
        evidence = None
        if candidate:
            try:
                raw = request.content
                body = json.loads(raw)
                if isinstance(body, dict):
                    evidence = {
                        "kind": "loopback_http_response_observed",
                        "request_sha256": hashlib.sha256(raw).hexdigest(),
                        "model": str(body.get("model") or ""),
                        "role_sequence": [
                            m.get("role") for m in body.get("messages", []) if isinstance(m, dict)
                        ],
                        "tool_messages": _tool_messages(body),
                        "wire_surface": _request_surface(body),
                    }
            except (ValueError, TypeError, RuntimeError, AttributeError):
                pass
        response = original_send(client, request, *args, **kwargs)
        if evidence is not None:
            evidence["http_status"] = response.status_code
            evidence["timestamp_unix_ns"] = time.time_ns()
            with lock, path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(evidence, sort_keys=True) + "\n")
        return response

    path.parent.mkdir(parents=True, exist_ok=True)
    httpx.Client.send = observed
    try:
        yield
    finally:
        httpx.Client.send = original_send
