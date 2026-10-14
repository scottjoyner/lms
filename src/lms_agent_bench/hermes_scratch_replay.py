#!/usr/bin/env python3
"""Explicit opt-in disposable Neo4j launcher for archived Hermes qualification.

Research-only. No user-selected Docker image, container, host, network, or
Cypher is accepted. The only write operation seeds a fixed temporary graph.
The graph has no network access, host mounts, published ports or persistence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .hermes_real_cypher_gate import (
    CONTAINER_NAME, RESEARCH_LABEL, SCRATCH_IMAGE,
    ScratchCypherGate, WitnessDenied,
)
from .hermes_real_cypher_audit import score_archived_case

SEED = (
    'CREATE (s:Service {name:"router-api",owner:"team-atlas"}), '
    '(o:Service {name:"worker-api",owner:"team-beta"}), '
    '(t:Team {name:"team-atlas"}), (u:Team {name:"team-beta"}), '
    '(d:Device {id:"device:xwing"}), '
    '(s)-[:OWNED_BY]->(t), (o)-[:OWNED_BY]->(u)'
)
FIXTURE_READBACK = (
    'MATCH (s:Service) RETURN s.name AS service, '
    's.owner AS owner ORDER BY service'
)
RUN_COMMAND = [
    "docker", "run", "-d", "--name", CONTAINER_NAME,
    "--hostname", "lfm25-hermes-witness",
    "--add-host", "lfm25-hermes-witness:127.0.0.1",
    "--label", f"openai.research_scope={RESEARCH_LABEL}",
    "--network", "none", "--security-opt", "no-new-privileges",
    "--pids-limit", "256", "--memory", "3g", "--memory-swap", "3g",
    "--cpus", "2", "--tmpfs", "/data:rw,size=2g",
    "--tmpfs", "/logs:rw,size=64m",
    "-e", "NEO4J_AUTH=none",
    "-e", "NEO4J_server_memory_heap_initial__size=256m",
    "-e", "NEO4J_server_memory_heap_max__size=512m",
    "-e", "NEO4J_server_memory_pagecache_size=256m",
    SCRATCH_IMAGE,
]


def _run(argv: list[str], *, timeout: int):
    return subprocess.run(argv, capture_output=True, text=True,
                          timeout=timeout, check=False)


class ScratchHermesArchive:
    def __init__(self, *, runner=_run):
        self.runner = runner
        self.container_id: str | None = None
        self.destroyed = False
        self.readback: str | None = None

    def _execute(self, args: list[str], *, timeout: int):
        try:
            result = self.runner(args, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WitnessDenied("isolated Docker operation unavailable") from exc
        if result.returncode:
            raise WitnessDenied(
                "isolated Docker operation failed: " + args[1]
            )
        return result

    def _cypher(self, query: str, *, mode: str):
        assert mode in ("read", "write")
        return self._execute([
            "docker", "exec", CONTAINER_NAME,
            "cypher-shell", "-a", "bolt://127.0.0.1:7687",
            "-d", "neo4j", "--non-interactive", "--format", "plain",
            "--access-mode", mode, query,
        ], timeout=27)

    def __enter__(self):
        current = self.runner(
            ["docker", "inspect", "--format", "{{.Id}}", CONTAINER_NAME],
            timeout=12,
        )
        if current.returncode == 0:
            raise WitnessDenied("named container exists: never reuse or replace")
        try:
            created = self._execute(RUN_COMMAND, timeout=35)
            identifier = created.stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{64}", identifier):
                raise WitnessDenied("Docker did not return an exact container ID")
            self.container_id = identifier
            self.gate = ScratchCypherGate(
                expected_container_id=identifier, command_runner=self.runner,
            )
            self.gate._checked_container()
            for attempt in range(15):
                try:
                    self._cypher("RETURN 1 AS ready", mode="read")
                    break
                except WitnessDenied:
                    if attempt == 14:
                        raise WitnessDenied("disposable Neo4j not ready")
                    time.sleep(2)
            empty = self._cypher(
                "MATCH (n) RETURN count(n) AS node_count", mode="read"
            )
            if empty.stdout.splitlines()[-1:] != ["0"]:
                raise WitnessDenied("unexpected pre-seeded research graph")
            # Fixed literals, applied ONLY inside the exact, networkless,
            # mountless newly created container; no arbitrary write API.
            self.gate._checked_container()
            self._cypher(SEED, mode="write")
            readback = self._cypher(FIXTURE_READBACK, mode="read").stdout
            if not all(x in readback for x in (
                "router-api", "team-atlas", "worker-api", "team-beta",
            )):
                raise WitnessDenied("independent fixture readback mismatch")
            self.readback = readback
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.container_id or self.destroyed:
            return
        # NEVER destroy anything without proving immutable custody.
        gate = ScratchCypherGate(
            expected_container_id=self.container_id, command_runner=self.runner,
        )
        gate._checked_container()
        self._execute(
            ["docker", "rm", "-f", CONTAINER_NAME], timeout=25
        )
        self.destroyed = True
        check = self.runner(
            ["docker", "inspect", "--format", "{{.Id}}", CONTAINER_NAME],
            timeout=12,
        )
        if check.returncode == 0:
            self.destroyed = False
            raise WitnessDenied("research teardown not confirmed")

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def run_archived_research(case_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise WitnessDenied("refusing to overwrite an evidence file")
    case_dir = case_dir.resolve(strict=True)
    if not (case_dir / "trial-01" / "trial-report.json").is_file():
        raise WitnessDenied("archive lacks expected Hermes trial receipts")
    with ScratchHermesArchive() as scratch:
        result = score_archived_case(case_dir, scratch.gate)
        result["fixture_readback"] = scratch.readback
        result["exact_container_id"] = scratch.container_id
        result["isolated_docker_image"] = SCRATCH_IMAGE
        result["time_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if not scratch.destroyed:
        raise WitnessDenied("research fixture teardown not established")
    result["research_container_destroyed"] = True
    result["not_production_qualified"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-research-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_research_only:
        parser.error("--confirm-research-only is mandatory")
    def terminate(_signal_number, _frame):
        raise WitnessDenied("research run interrupted by termination signal")

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        report = run_archived_research(args.case_dir, args.output)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    print("REAL_CYPHER_RESEARCH", report["real_end_to_end_pass_count"],
          "/", report["trial_count"], "HOLD", not report["case_gate_passed"])
    # Operator/CI must never equate research completion with model admission.
    return 0 if report["case_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
