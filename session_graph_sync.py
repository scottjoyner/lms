#!/usr/bin/env python3
"""session_graph_sync.py — sync opencode (Hermes) sessions into Neo4j.

Hermes' previous sessions are the most valuable memory in the system. This
module mirrors them into the knowledge graph so OTHER agents can:

  * discover what work happened in a session (title, directory, model, tokens)
  * walk a session's message / reasoning / tool-call chain (session -[:HAS]-> ...)
  * ask "what did we do for X?" across all historical sessions

Source of truth: ~/.local/share/opencode/opencode.db (SQLite). The schema is
  session(id, title, directory, project_id, summary_*)
  message(id, session_id, time_created, data{role,time,agent,model,summary})
  part(id, message_id, session_id, time_created, data{type,text,tool,...})

We map:
  session -> :session        (session_id = oc:ses_...)
  message -> :message        (per turn)
  part type=reasoning -> :reasoning
  part type=tool     -> :toolcall   (with name + input)
  part type=text     -> folded into the message body

Idempotent (MERGE on stable ids) and incremental: pass --since-ms to only sync
recent sessions, or run on a cron.

Run:
  python3 session_graph_sync.py sync
  python3 session_graph_sync.py sync --since-days 1
  python3 session_graph_sync.py sync --db /path/to/opencode.db --watch --sleep 300
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_common import get_driver, ensure_constraints

DEFAULT_DB = Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _ms(ts: Any) -> Optional[int]:
    if ts is None:
        return None
    try:
        return int(ts)
    except (TypeError, ValueError):
        return None


def _part_payload(part_data: str) -> Dict[str, Any]:
    try:
        d = json.loads(part_data) if isinstance(part_data, str) else (part_data or {})
    except Exception:
        return {"type": "unknown"}
    return d if isinstance(d, dict) else {"type": "unknown"}


def collect_session(con: sqlite3.Connection, sid: str) -> Dict[str, Any]:
    srow = con.execute(
        "SELECT id, title, directory, project_id, summary_additions FROM session WHERE id=?",
        (sid,),
    ).fetchone()
    if not srow:
        return {}
    # messages
    msgs = con.execute(
        "SELECT id, time_created, data FROM message WHERE session_id=? ORDER BY time_created",
        (sid,),
    ).fetchall()
    messages: List[Dict[str, Any]] = []
    for m in msgs:
        md = {}
        if m["data"]:
            try:
                md = json.loads(m["data"])
            except Exception:
                md = {}
        parts = con.execute(
            "SELECT id, time_created, data FROM part WHERE message_id=? ORDER BY time_created",
            (m["id"],),
        ).fetchall()
        turns: List[Dict[str, Any]] = []
        for p in parts:
            pd = _part_payload(p["data"])
            turns.append({
                "part_id": p["id"],
                "type": pd.get("type"),
                "time_ms": _ms(p["time_created"]),
                "text": pd.get("text"),
                "tool": pd.get("tool"),
                "input": pd.get("input"),
                "output": pd.get("output"),
                "title": pd.get("title"),
            })
        model = md.get("model")
        if isinstance(model, dict):
            model = model.get("modelID") or model.get("id") or json.dumps(model)
        messages.append({
            "message_id": m["id"],
            "time_ms": _ms(m["time_created"]),
            "role": md.get("role"),
            "agent": md.get("agent"),
            "model": model if isinstance(model, str) else (str(model) if model else None),
            "parts": turns,
        })
    return {
        "session_id": srow["id"],
        "title": srow["title"],
        "directory": srow["directory"],
        "project_id": srow["project_id"],
        "messages": messages,
    }


def sync_one(driver, sess: Dict[str, Any]) -> int:
    oc_id = sess["session_id"]
    node_id = f"oc:{oc_id}"
    with driver.session() as s:
        s.run(
            """
            MERGE (se:session {session_id: $node_id})
            SET se += $props
            """,
            node_id=node_id,
            props={
                "session_id": node_id,
                "oc_session_id": oc_id,
                "source": "opencode",
                "kind": "session",
                "title": sess.get("title") or "",
                "directory": sess.get("directory") or "",
                "project_id": sess.get("project_id") or "global",
                "updated_at_ms": _now_ms(),
                "message_count": len(sess["messages"]),
            },
        )
        # messages + parts
        for msg in sess["messages"]:
            mid = msg["message_id"]
            body = " ".join(
                p.get("text") or "" for p in msg["parts"] if p.get("type") == "text" and p.get("text")
            )
            s.run(
                """
                MERGE (m:message {message_id: $mid})
                SET m += $props
                WITH m
                MATCH (se:session {session_id: $sid})
                MERGE (se)-[:HAS]->(m)
                """,
                mid=mid,
                sid=node_id,
                props={
                    "message_id": mid,
                    "role": msg.get("role"),
                    "agent": msg.get("agent"),
                    "model": msg.get("model"),
                    "time_ms": msg.get("time_ms"),
                    "text": body[:8000],
                },
            )
            for p in msg["parts"]:
                if p["type"] == "reasoning" and p.get("text"):
                    pid = f"{mid}:{p['part_id']}"
                    s.run(
                        """
                        MERGE (r:reasoning {part_id: $pid})
                        SET r += $props
                        WITH r
                        MATCH (se:session {session_id: $sid})
                        MERGE (se)-[:HAS]->(r)
                        """,
                        pid=pid,
                        sid=node_id,
                        props={"part_id": pid, "text": p["text"][:8000], "time_ms": p.get("time_ms")},
                    )
                elif p["type"] == "tool":
                    pid = f"{mid}:{p['part_id']}"
                    s.run(
                        """
                        MERGE (t:toolcall {part_id: $pid})
                        SET t += $props
                        WITH t
                        MATCH (se:session {session_id: $sid})
                        MERGE (se)-[:HAS]->(t)
                        """,
                        pid=pid,
                        sid=node_id,
                        props={
                            "part_id": pid,
                            "tool": p.get("tool"),
                            "input": json.dumps(p.get("input"))[:4000] if p.get("input") is not None else None,
                            "time_ms": p.get("time_ms"),
                        },
                    )
    return len(sess["messages"])


def cmd_sync(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[error] opencode db not found: {db_path}", file=sys.stderr)
        return 2
    con = _open_db(db_path)
    try:
        since_ms = None
        if args.since_days:
            since_ms = _now_ms() - int(args.since_days * 86400 * 1000)
        rows = con.execute("SELECT id FROM session ORDER BY id").fetchall()
        sids = [r["id"] for r in rows]
    finally:
        con.close()

    driver = get_driver()
    try:
        ensure_constraints(driver, [
            "CREATE CONSTRAINT sg_session_id IF NOT EXISTS FOR (n:session) REQUIRE n.session_id IS UNIQUE",
            "CREATE CONSTRAINT sg_message_id IF NOT EXISTS FOR (n:message) REQUIRE n.message_id IS UNIQUE",
            "CREATE CONSTRAINT sg_part_id IF NOT EXISTS FOR (n:reasoning) REQUIRE n.part_id IS UNIQUE",
            "CREATE CONSTRAINT sg_tool_id IF NOT EXISTS FOR (n:toolcall) REQUIRE n.part_id IS UNIQUE",
        ])
        synced = 0
        total_msgs = 0
        for sid in sids:
            con = _open_db(db_path)
            try:
                sess = collect_session(con, sid)
            finally:
                con.close()
            if not sess:
                continue
            # incremental: skip sessions whose last message is older than since_ms
            if since_ms is not None:
                last = max((m["time_ms"] or 0) for m in sess["messages"]) if sess["messages"] else 0
                if last and last < since_ms:
                    continue
            total_msgs += sync_one(driver, sess)
            synced += 1
    finally:
        driver.close()

    print(f"Synced {synced} sessions, {total_msgs} messages into Neo4j.")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    while True:
        cmd_sync(args)
        print(f"[watch] next sync in {args.sleep}s", flush=True)
        time.sleep(args.sleep)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync opencode sessions into Neo4j.")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync", help="sync sessions")
    s.add_argument("--since-days", type=float, default=None)
    s.set_defaults(func=cmd_sync)
    w = sub.add_parser("watch", help="sync on interval")
    w.add_argument("--sleep", type=int, default=300)
    w.add_argument("--since-days", type=float, default=1)
    w.set_defaults(func=cmd_watch)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
