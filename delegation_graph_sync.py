#!/usr/bin/env python3
"""delegation_graph_sync.py — agent coordination inbox backed by Neo4j.

Lets agents coordinate current work programmatically:

  (Agent {name}) -[:DELEGATED_TO]-> (delegation {goal, context, status, embedding})
  (delegation) -[:SIMILAR]-> (delegation)        # semantic neighbours (top-k)

An agent that wants help publishes a delegation with status="open". Another
agent (or the orchestrator) claims it (status="claimed", claimed_by=...), does
the work, then marks it "done" with a result. Because delegations carry
embeddings, an agent can find relevant/duplicate/related work via vector search
instead of polling.

This is the coordination layer the user asked for: "coordinate current work
with other agents — health stored in the graph so other agents know
programmatically."

Run:
  python3 delegation_graph_sync.py create \\
      --goal "Benchmark qwen3.5 on deathstar" --context "..." --from-agent build --tags bench
  python3 delegation_graph_sync.py claim --id <delegation_id> --agent orchestrator
  python3 delegation_graph_sync.py complete --id <delegation_id> --result "ok, 38 tps"
  python3 delegation_graph_sync.py list --status open
  python3 delegation_graph_sync.py find "fleet loadout planning" --k 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

from graph_common import Embedder, get_driver, ensure_constraints, vector_query


def _now_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def cmd_create(args: argparse.Namespace) -> int:
    goal = args.goal
    context = args.context or ""
    from_agent = args.from_agent or "unknown"
    tags = [t for t in (args.tags or "").split(",") if t]
    deleg_id = f"deleg:{uuid.uuid4().hex[:12]}"
    emb = Embedder.embed(goal + "\n" + context)
    driver = get_driver()
    try:
        ensure_constraints(driver, [
            "CREATE CONSTRAINT del_id IF NOT EXISTS FOR (n:delegation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT del_agent IF NOT EXISTS FOR (n:Agent) REQUIRE n.name IS UNIQUE",
        ])
        with driver.session() as s:
            s.run(
                """
                MERGE (a:Agent {name: $agent})
                SET a.updated_at = $now
                WITH a
                CREATE (d:delegation {id: $id})
                SET d += $props
                MERGE (a)-[:DELEGATED_TO]->(d)
                """,
                agent=from_agent,
                id=deleg_id,
                now=_now_iso(),
                props={
                    "id": deleg_id,
                    "goal": goal,
                    "context": context,
                    "status": "open",
                    "tags": tags,
                    "from_agent": from_agent,
                    "claimed_by": None,
                    "result": "",
                    "embedding": emb,
                    "created_at_ms": _now_ms(),
                    "updated_at_ms": _now_ms(),
                },
            )
    finally:
        driver.close()
    print(f"Created delegation {deleg_id} from {from_agent} (status=open)")
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    driver = get_driver()
    try:
        with driver.session() as s:
            rec = s.run(
                """
                MATCH (d:delegation {id: $id})
                WHERE d.status = 'open'
                SET d.status = 'claimed', d.claimed_by = $agent, d.updated_at_ms = $now
                RETURN d.id AS id, d.goal AS goal
                """,
                id=args.id, agent=args.agent or "unknown", now=_now_ms(),
            ).single()
    finally:
        driver.close()
    if rec:
        print(f"Claimed {rec['id']}: {rec['goal']}")
        return 0
    print(f"[warn] delegation {args.id} not found or not open", file=sys.stderr)
    return 1


def cmd_complete(args: argparse.Namespace) -> int:
    driver = get_driver()
    try:
        with driver.session() as s:
            rec = s.run(
                """
                MATCH (d:delegation {id: $id})
                SET d.status = 'done', d.result = $result, d.updated_at_ms = $now
                RETURN d.id AS id
                """,
                id=args.id, result=args.result or "", now=_now_ms(),
            ).single()
    finally:
        driver.close()
    print(f"Completed {args.id}" if rec else f"[warn] delegation {args.id} not found")
    return 0 if rec else 1


def cmd_list(args: argparse.Namespace) -> int:
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = s.run(
                "MATCH (d:delegation) WHERE d.status = $status "
                "RETURN d.id AS id, d.goal AS goal, d.from_agent AS from, d.claimed_by AS by, d.tags AS tags "
                "ORDER BY d.created_at_ms DESC LIMIT $limit",
                status=args.status, limit=args.limit,
            ).data()
    finally:
        driver.close()
    if not rows:
        print(f"(no delegations with status={args.status})")
        return 0
    for r in rows:
        print(f"[{r['id']}] {r['goal']}  from={r['from']} claimed_by={r['by']} tags={r['tags']}")
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    driver = get_driver()
    try:
        qvec = Embedder.embed(args.query)
        results = vector_query(driver, "delegation", qvec, k=args.k)
        for node, score in results:
            goal = node.get("goal", "") if hasattr(node, "get") else str(node)
            print(f"[score={score:.3f}] {goal}  (id={node.get('id') if hasattr(node,'get') else '?'})")
    finally:
        driver.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent coordination inbox in Neo4j.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create", help="publish a delegation")
    c.add_argument("--goal", required=True)
    c.add_argument("--context", default="")
    c.add_argument("--from-agent", default="unknown")
    c.add_argument("--tags", default="")
    c.set_defaults(func=cmd_create)

    cl = sub.add_parser("claim", help="claim an open delegation")
    cl.add_argument("--id", required=True)
    cl.add_argument("--agent", default="unknown")
    cl.set_defaults(func=cmd_claim)

    cp = sub.add_parser("complete", help="mark a delegation done")
    cp.add_argument("--id", required=True)
    cp.add_argument("--result", default="")
    cp.set_defaults(func=cmd_complete)

    ls = sub.add_parser("list", help="list delegations by status")
    ls.add_argument("--status", default="open")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_list)

    f = sub.add_parser("find", help="semantic search over delegations")
    f.add_argument("query")
    f.add_argument("--k", type=int, default=5)
    f.set_defaults(func=cmd_find)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
