"""Fail-closed preflight for *metadata-only* native Hermes tool-presentation A/B pairs.

Never treats metadata agreement as owner-query, model-quality or release evidence.
No production endpoints, credentials or raw prompt contents are accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BRIDGE = {"tool_search", "tool_describe", "tool_call"}
DIRECT = {"mcp__lms_benchmark__" + name for name in (
    "graph_schema", "graph_query", "workspace_read", "workspace_write",
    "workspace_patch", "workspace_run_tests", "weather_lookup",
    "send_email", "graph_delete_all",
)}


class PairRejected(ValueError):
    """Wire-level evidence does not isolate presentation from prompt drift."""


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _fingerprint(value):
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

def _read_arm(path: Path, expected: set[str]) -> dict:
    raw = path.read_bytes()
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise PairRejected("unparseable wire metadata") from exc
    if not records:
        raise PairRejected("missing model requests")
    first = records[0]
    try:
        messages = first["messages"]
        if first["message_roles"][:2] != ["system", "user"]:
            raise PairRejected("first request must begin system/user")
        first_system = messages[0]["content_sha256"]
        first_user = messages[1]["content_sha256"]
        model = first["model_id"]
        settings = first["generation"]
        schema = first["advertised_tools"]
        names = {item["name"] for item in schema}
        if names != expected or len(schema) != len(expected):
            raise PairRejected("unexpected tool-presentation surface")
        if settings.get("temperature") != 0 or settings.get("max_tokens") != 640:
            raise PairRejected("nonmatching bounded decoding parameters")
        if settings.get("stream") is not True:
            raise PairRejected("unexpected model response transport")

        schema_hash = _fingerprint(schema)
        for record in records:
            msgs = record["messages"]
            if not msgs or msgs[0]["content_sha256"] != first_system:
                raise PairRejected("preamble changed inside an arm")
            if len(msgs) < 2 or msgs[1]["content_sha256"] != first_user:
                raise PairRejected("user prompt changed inside an arm")
            if record["model_id"] != model or record["generation"] != settings:
                raise PairRejected("model or decoding settings changed inside an arm")
            if _fingerprint(record["advertised_tools"]) != schema_hash:
                raise PairRejected("tools or tool schemas changed inside an arm")
            if record["raw_bodies_saved"] is not False or record["raw_headers_saved"] is not False:
                raise PairRejected("unsanitized wire capture")
    except (KeyError, IndexError, TypeError) as exc:
        raise PairRejected("incomplete wire receipt") from exc
    return {"count": len(records), "metadata_sha256": _digest(raw),
            "system_sha256": first_system, "user_sha256": first_user,
            "model_id": model, "generation": settings,
            "tool_names": sorted(names), "tool_schema_sha256": schema_hash}


def compare(bridge: Path, direct: Path) -> dict:
    left = _read_arm(bridge, BRIDGE)
    right = _read_arm(direct, DIRECT)
    for field in ("system_sha256", "user_sha256", "model_id", "generation"):
        if left[field] != right[field]:
            raise PairRejected("A/B confound: " + field)
    return {"schema": "hermes-tool-presentation-wire-pair.v1",
            "wire_pair_admissible": True, "native_owner_proven": False,
            "promotion_authorized": False, "bridge": left, "direct": right}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--direct", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.bridge, args.direct)
    except (OSError, PairRejected) as exc:
        print(json.dumps({"wire_pair_admissible": False,
                          "promotion_authorized": False,
                          "reason": str(exc)}, sort_keys=True))
        return 3
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
