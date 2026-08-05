#!/usr/bin/env python3
"""Shared identity, suite, and trace utilities for Hermes benchmarks."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

SCHEMA_VERSION = "hermes_agent_benchmark.v1"
GATE_SCHEMA_VERSION = "hermes_agent_benchmark_gate.v1"
SUITE_SCHEMA_VERSION = "hermes_agent_suite.v1"
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
WRITE_CYPHER_RE = re.compile(
    r"\b(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_sha256(value: str) -> str:
    text = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(text):
        raise ValueError(
            "model content SHA-256 must contain exactly 64 hexadecimal characters"
        )
    return text if text.startswith("sha256:") else "sha256:" + text


def require_loopback_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must use http or https")
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "Hermes intelligence benchmark endpoint must be loopback-local"
        )
    return value.rstrip("/")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def suite_default_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "benchmarks"
        / "hermes_agent_suite.v1.json"
    )


def validate_suite(suite: Mapping[str, Any]) -> Dict[str, Any]:
    if suite.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise ValueError(f"suite schema must be {SUITE_SCHEMA_VERSION}")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite requires a non-empty cases array")
    seen: set[str] = set()
    normalized_cases: List[Dict[str, Any]] = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, Mapping):
            raise ValueError(f"suite case {index} must be an object")
        case = dict(raw)
        key = str(case.get("case_key") or "").strip()
        if not key:
            raise ValueError(f"suite case {index} has no case_key")
        if key in seen:
            raise ValueError(f"duplicate suite case_key: {key}")
        seen.add(key)
        if str(case.get("priority") or "") not in {"P0", "P1", "P2"}:
            raise ValueError(f"suite case {key} has invalid priority")
        if not str(case.get("prompt") or "").strip():
            raise ValueError(f"suite case {key} has no prompt")
        checkpoints = case.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"suite case {key} requires checkpoints")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, Mapping) or not checkpoint.get("type"):
                raise ValueError(f"suite case {key} has an invalid checkpoint")
            weight = float(checkpoint.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(
                    f"suite case {key} checkpoint weight must be positive"
                )
        workspace = case.get("workspace", {})
        if not isinstance(workspace, Mapping):
            raise ValueError(f"suite case {key} workspace must be an object")
        for relative in workspace:
            path = Path(str(relative))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"suite case {key} has unsafe workspace path: {relative}"
                )
        normalized_cases.append(case)
    minimum = int(suite.get("minimum_valid_trials", 3))
    if minimum < 1:
        raise ValueError("minimum_valid_trials must be positive")
    gate = suite.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("suite requires a gate object")
    return {
        **dict(suite),
        "cases": normalized_cases,
        "minimum_valid_trials": minimum,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            values.append(item)
    return values


def parse_json_text(text: str) -> Optional[Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(
            r"```(?:json)?\s*(.*?)```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
    return None


def nested_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and nested_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(nested_subset(item, wanted) for item in actual)
            for wanted in expected
        )
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.casefold() == expected.casefold()
    return actual == expected


def normalize_tool_name(value: str) -> str:
    name = str(value or "")
    for prefix in (
        "mcp_lms_benchmark_",
        "mcp__lms_benchmark__",
        "lms_benchmark_",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def extract_messages(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    messages = result.get("messages")
    if not isinstance(messages, list):
        return []
    return [dict(item) for item in messages if isinstance(item, Mapping)]


def extract_tool_calls(
    messages: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for message in messages:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            continue
        for raw in raw_calls:
            if not isinstance(raw, Mapping):
                continue
            function = (
                raw.get("function")
                if isinstance(raw.get("function"), Mapping)
                else raw
            )
            name = normalize_tool_name(
                str(function.get("name") or raw.get("name") or "")
            )
            arguments = function.get("arguments", raw.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {
                        "_raw": arguments,
                        "_parse_error": True,
                    }
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "id": raw.get("id"),
                }
            )
    return calls


def collect_usage(value: Any) -> Dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if isinstance(item.get("usage"), Mapping):
                usage = item["usage"]
                for key in totals:
                    raw = usage.get(key)
                    if isinstance(raw, int) and raw >= 0:
                        totals[key] += raw
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if isinstance(value, Mapping) and isinstance(value.get("usage"), Mapping):
        usage = value["usage"]
        return {key: int(usage.get(key) or 0) for key in totals}
    return totals
