"""Native Hermes scratch-only worker-api holdout; no production admission."""
from pathlib import Path

from .hermes_native_generic_query import GENERIC
from .hermes_native_query_description import ORIGINAL, adapt_description, digest


def adapt_worker_request(original: dict, *, pinned: Path):
    """Private preamble pinned on first turn, generic Cypher description only."""
    request, receipt = adapt_description(original, mode="baseline", pinned=pinned)
    tool = next(
        t for t in request["tools"]
        if t["function"]["name"] == "mcp__lms_benchmark__graph_query"
    )
    if tool["function"].get("description") != ORIGINAL:
        raise ValueError("unexpected source tool schema")
    tool["function"]["description"] = GENERIC
    return request, {
        **receipt,
        "mode": "worker-api-holdout",
        "description_sha256": digest(GENERIC),
        "release_authorized": False,
        "qualification_authorized": False,
    }
