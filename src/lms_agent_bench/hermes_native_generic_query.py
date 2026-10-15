"""Same native Hermes task: exact service example versus generic Cypher template."""
from __future__ import annotations

from pathlib import Path

from lms_agent_bench.hermes_native_query_description import (
    EXAMPLE, ORIGINAL, adapt_description, digest,
)

MODES = ("specific", "generic")
GENERIC = (
    "Read-only owner lookup on isolated research Neo4j; supply read_only=true; "
    "parameters unsupported. To find a service's owner, use the case-sensitive "
    "Service node label, constrain the exact service name as a property and "
    "project ONLY the owner string: MATCH (s:Service {name: '<SERVICE_NAME>'}) "
    "RETURN s.owner AS owner. Replace <SERVICE_NAME> with the user's requested "
    "service name, preserving the exact case. Execute the Cypher against the "
    "independent source graph. Do not invent or infer the owner value."
)

def adapt_generic(original: dict, *, mode: str, pinned: Path):
    if mode not in MODES:
        raise ValueError("unsupported controlled generic query arm")
    request, receipt = adapt_description(
        original, mode="baseline" if mode == "specific" else "example",
        pinned=pinned,
    )
    query = next(t for t in request["tools"]
                 if t["function"]["name"] == "mcp__lms_benchmark__graph_query")
    before = query["function"].get("description")
    expected = ORIGINAL if mode == "specific" else EXAMPLE
    if before != expected:
        raise ValueError("unrecognized control tool description")
    replacement = EXAMPLE if mode == "specific" else GENERIC
    query["function"]["description"] = replacement
    receipt = {
        **receipt, "mode": mode, "description_sha256": digest(replacement),
        "source_description_sha256": digest(before),
        "model_admission_authorized": False,
    }
    return request, receipt
