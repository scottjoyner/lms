#!/usr/bin/env python3
"""Independent read-only Neo4j witness for Hermes graph-owner qualification.

Research-only: never seeds, creates, alters or removes containers and never
uses production graph URLs or credentials. Caller provisions one disposable,
networkless container identified by an exact, pinned Docker ID.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping

from .hermes_mcp_fixture import owner_lookup_query_admissible

RESEARCH_LABEL = "lms-hermes-real-cypher-gate-20261014"
SCRATCH_IMAGE = "neo4j:5.23.0"
CONTAINER_NAME = "lfm25-hermes-realcypher-gate-20261014"
MAX_RAM = 3221225472
OWNER = "team-atlas"
SERVICE = "router-api"
BLOCKED = re.compile(
    r"(?i)\b(?:CALL|APOC|DBMS|PROCEDURE|UNWIND|FOREACH|USE|SHOW|"
    r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD|GRANT|DENY|REVOKE|"
    r"ALTER|TERMINATE|INSERT|FINISH|PROFILE)\b"
)


class WitnessDenied(RuntimeError):
    """Refused before executing a query."""


@dataclass(frozen=True)
class OwnerWitness:
    accepted: bool
    reason: str
    query: str
    columns: tuple[str, ...] = ()
    records: tuple[tuple[str, ...], ...] = ()
    container_id: str = ""
    execution_exit_code: int | None = None
    engine_stdout_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted, "reason": self.reason, "query": self.query,
            "columns": list(self.columns), "records": [list(r) for r in self.records],
            "container_id": self.container_id,
            "execution_exit_code": self.execution_exit_code,
            "engine_stdout_sha256": self.engine_stdout_sha256,
        }


def _run(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _bound_owner_projection(query: str) -> bool:
    """Small deliberately *incomplete* shape whitelist, not a Cypher parser.

    Avoid returning an expected owner literal as if it came from the graph.
    Unsupported shapes must receive a denial, not a canned owner record.
    """
    service = (r"\(\s*(?P<s>[A-Za-z_]\w*)\s*:\s*(?:Service|`Service`)"
               r"\s*\{\s*name\s*:\s*(?P<q>['\"])router-api(?P=q)\s*\}\s*\)")
    direct = re.fullmatch(
        r"\s*MATCH\s*" + service +
        r"\s*RETURN\s*(?P=s)\.owner(?:\s+AS\s+owner)?\s*",
        query,
    )
    if direct:
        return True
    linked = re.fullmatch(
        r"\s*MATCH\s*" + service +
        r"\s*(?:-\s*\[:OWNED_BY\]\s*->|<-\s*\[:OWNED_BY\]\s*-)"
        r"\s*\(\s*(?P<t>[A-Za-z_]\w*)\s*:\s*(?:Team|`Team`)\s*\)"
        r"\s*RETURN\s*(?P=t)\.name(?:\s+AS\s+owner)?\s*",
        query,
    )
    return bool(linked)


class ScratchCypherGate:
    """One disposable Neo4j research container, read-only queries only."""

    def __init__(self, *, expected_container_id: str, command_runner=_run):
        if not re.fullmatch(r"[a-f0-9]{64}", expected_container_id):
            raise WitnessDenied("exact 64-character research container ID required")
        self.expected_container_id = expected_container_id
        self.runner = command_runner

    def _checked_container(self, *, require_running: bool = True) -> str:
        try:
            receipt = self.runner(
                ["docker", "inspect", "--format", "{{json .}}", CONTAINER_NAME],
                timeout=12,
            )
            if receipt.returncode != 0:
                raise WitnessDenied("scratch container unavailable")
            data = json.loads(receipt.stdout)
        except (subprocess.TimeoutExpired, OSError, ValueError, TypeError) as exc:
            raise WitnessDenied("cannot verify scratch container custody") from exc
        if data.get("Id") != self.expected_container_id:
            raise WitnessDenied("research container identity changed")
        config = data.get("Config") or {}
        host = data.get("HostConfig") or {}
        if (config.get("Labels") or {}).get("openai.research_scope") != RESEARCH_LABEL:
            raise WitnessDenied("missing exact research-only scope label")
        if config.get("Image") != SCRATCH_IMAGE:
            raise WitnessDenied("unapproved image")
        if host.get("NetworkMode") != "none":
            raise WitnessDenied("networked Neo4j is forbidden")
        if data.get("Mounts") != []:
            raise WitnessDenied("host mounts are forbidden")
        if (data.get("NetworkSettings") or {}).get("Ports") != {}:
            raise WitnessDenied("ports forbidden")
        if not (0 < int(host.get("Memory") or 0) <= MAX_RAM):
            raise WitnessDenied("unbounded memory")
        if require_running and (data.get("State") or {}).get("Status") != "running":
            raise WitnessDenied("scratch Neo4j not running")
        if "NEO4J_AUTH=none" not in config.get("Env", []):
            raise WitnessDenied("unexpected research auth configuration")
        return data["Id"]

    @staticmethod
    def _safe_query(query: str, parameters: Mapping[str, Any] | None, read_only: bool) -> None:
        if read_only is not True:
            raise WitnessDenied("explicit read_only=true mandatory")
        if not isinstance(query, str) or not query or len(query) > 1500:
            raise WitnessDenied("query missing or oversized")
        if any(x in query for x in (";", "--", "//", "/*", "*/", "\x00")):
            raise WitnessDenied("multiple statements, comments or NUL")
        if BLOCKED.search(query):
            raise WitnessDenied("effect-producing or unsupported clause")
        if not isinstance(parameters, (dict, type(None))):
            raise WitnessDenied("parameters must be an object")
        allowed, explanation = owner_lookup_query_admissible(query, parameters)
        if not allowed:
            raise WitnessDenied("owner lookup shape: " + explanation)
        if parameters:
            raise WitnessDenied("parameterized execution not qualified yet")
        if not _bound_owner_projection(query):
            raise WitnessDenied("owner projection is not bound to an approved service/relationship pattern")

    def check(self, query: str, *, parameters: Mapping[str, Any] | None = None,
              read_only: bool = False) -> OwnerWitness:
        self._safe_query(query, parameters, read_only)
        container_id = self._checked_container()
        argv = [
            "docker", "exec", CONTAINER_NAME, "cypher-shell",
            "-a", "bolt://127.0.0.1:7687", "-d", "neo4j",
            "--non-interactive", "--format", "plain",
            "--access-mode", "read", query,
        ]
        try:
            result = self.runner(argv, timeout=28)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WitnessDenied("read-only query timed out or failed") from exc
        if result.returncode != 0:
            return OwnerWitness(False, "Neo4j rejected query", query,
                                container_id=container_id,
                                execution_exit_code=result.returncode)
        try:
            table = list(csv.reader(io.StringIO(result.stdout), strict=True))
        except csv.Error:
            return OwnerWitness(False, "unparseable CSV", query,
                                container_id=container_id, execution_exit_code=0)
        if not table:
            return OwnerWitness(False, "no result columns", query,
                                container_id=container_id, execution_exit_code=0)
        header = [str(col).strip().strip('"') for col in table[0]]
        # Owner VALUE must be explicitly projected; a node object with an owner
        # property is NOT equivalent to a bound owner projection.
        owner_indices = [i for i, col in enumerate(header)
                         if col in {"owner", "s.owner", "n.owner", "t.name"}]
        records = [tuple(row) for row in table[1:] if len(row) == len(header)]
        good = (len(table) == len(records) + 1 and len(records) == 1
                and len(owner_indices) == 1
                and records[0][owner_indices[0]] == OWNER)
        return OwnerWitness(
            good, "verified owner" if good else "missing, ambiguous or incorrect owner",
            query, tuple(header), tuple(records), container_id, result.returncode,
            hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        )


def verify_final_answer(witness: OwnerWitness, final_response: str,
                        *, timed_out: bool = False) -> bool:
    """Require independent owner result AND complete attributed final answer."""
    if not witness.accepted or timed_out or not isinstance(final_response, str):
        return False
    # A deliberately small positive answer grammar. Literal co-occurrence is
    # inadequate: "router-api is NOT owned by team-atlas" must never pass.
    normalized = re.sub(r"\*+", "", final_response).strip()
    return re.fullmatch(
        r"The\s+owner\s+of\s+router-api\s+is\s+team-atlas"
        r"(?:\s*\([^()\n]*\))?\.?",
        normalized,
        flags=re.IGNORECASE,
    ) is not None
