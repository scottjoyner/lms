#!/usr/bin/env python3
"""knowledge_graph_sync.py — extract structured knowledge from memories into Neo4j.

Turns raw text (session transcripts, markdown notes, memory exports) into a
queryable knowledge graph:

  (KgNode {kind, source_id, title})          - the source document/memory
    -[:HAS_CHUNK]-> (Chunk {text, chunk_id, embedding})
  (Concept {name, kind})                      - extracted entity/topic
    -[:RELATES_TO]-> (Concept)                - co-occurrence in same chunk
  (Chunk) -[:SIMILAR]-> (Chunk)              - vector neighbours (top-k)

Other agents can then ask questions like "what do we know about X?" via vector
search over Chunk nodes (the existing `chunk_embedding` index) and follow
RELATES_TO/Concept links for connected facts.

This is deliberately offline: embeddings come from the local all-MiniLM-L6-v2
model (same one the rest of the graph uses) and no cloud calls are made.

Run:
  python3 knowledge_graph_sync.py ingest --source notes.md --kind markdown --title "My notes"
  python3 knowledge_graph_sync.py ingest --source session.json --kind opencode_session
  cat memory.txt | python3 knowledge_graph_sync.py ingest --stdin --kind memory
  python3 knowledge_graph_sync.py link-similar --top-k 5      # build SIMILAR edges
  python3 knowledge_graph_sync.py search "how does the fleet router route" --k 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List

from lms_agent_bench.graph_common import (
    Embedder,
    get_driver,
    ensure_constraints,
    vector_query,
)


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    """Split into overlapping chunks on paragraph/sentence boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    # Prefer paragraph breaks, then sentences, then hard splits.
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 2 <= max_chars:
            buf = (buf + "\n\n" + para).strip()
            continue
        if buf:
            chunks.append(buf)
        if len(para) <= max_chars:
            buf = para
        else:
            # hard split long paragraph by sentences then by chars
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(buf) + len(sent) + 1 <= max_chars:
                    buf = (buf + " " + sent).strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = sent
    if buf:
        chunks.append(buf)
    # apply overlap by merging tails is optional; keep simple non-overlapping here
    return chunks


def _extract_concepts(text: str, max_concepts: int = 12) -> List[Dict[str, str]]:
    """Lightweight concept extraction: capitalized noun phrases + known signals.

    Deliberately deterministic (no LLM) so it runs offline and is reproducible.
    """
    concepts: Dict[str, str] = {}
    # CamelCase / TitleCase identifiers and known technical terms
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+)\b", text):
        concepts[m.group(1)] = "identifier"
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", text):
        phrase = m.group(1)
        if phrase.lower() not in ("the fleet", "this is", "we need", "i am"):
            concepts[phrase] = "entity"
    # explicit technical keywords
    for kw in re.findall(r"\b(lm studio|neo4j|tailscale|auto-router|fleet|orchestrator|circuit breaker|outbox|embedding|kgraph|hermes|router|loadout|reporter)\b", text, re.IGNORECASE):
        concepts[kw.lower()] = "keyword"
    out = [{"name": k, "kind": v} for k, v in concepts.items()]
    return out[:max_concepts]


def cmd_ingest(args: argparse.Namespace) -> int:
    if args.stdin:
        text = sys.stdin.read()
        source_id = args.source or f"stdin:{hashlib.sha1(text[:200].encode()).hexdigest()[:8]}"
        title = args.title or "stdin input"
    else:
        path = Path(args.source)
        text = path.read_text(encoding="utf-8", errors="replace")
        source_id = args.source or str(path)
        title = args.title or path.name

    kind = args.kind
    node_id = str(uuid.uuid4())
    chunks = _chunk_text(text)
    concepts = _extract_concepts(text)

    driver = get_driver()
    try:
        ensure_constraints(driver, [
            "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KgNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT kg_concept_name IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
        ])
        with driver.session() as s:
            s.run(
                """
                MERGE (kn:KgNode {node_id: $node_id})
                SET kn += $props
                """,
                node_id=node_id,
                props={
                    "node_id": node_id,
                    "source_id": source_id,
                    "title": title,
                    "kind": kind,
                    "char_count": len(text),
                    "chunk_count": len(chunks),
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            for i, ch in enumerate(chunks):
                cid = f"{node_id}:{i}"
                emb = Embedder.embed(ch)
                s.run(
                    """
                    MERGE (c:Chunk {chunk_id: $cid})
                    SET c.text = $text, c.embedding = $emb, c.index = $i,
                        c.updated_at = $now, c.source_id = $source_id
                    WITH c
                    MATCH (kn:KgNode {node_id: $node_id})
                    MERGE (kn)-[:HAS_CHUNK]->(c)
                    """,
                    cid=cid, text=ch, emb=emb, i=i, now=dt.datetime.now(dt.timezone.utc).isoformat(),
                    source_id=source_id, node_id=node_id,
                )
            # Concepts + RELATES_TO within this source
            created = []
            for c in concepts:
                rec = s.run(
                    """
                    MERGE (cn:Concept {name: $name})
                    SET cn.kind = $kind, cn.updated_at = $now
                    RETURN cn
                    """,
                    name=c["name"], kind=c["kind"], now=dt.datetime.now(dt.timezone.utc).isoformat(),
                ).single()
                created.append(rec["cn"])
            for a in created:
                for b in created:
                    if a["name"] != b["name"]:
                        s.run(
                            "MATCH (x:Concept {name:$a}), (y:Concept {name:$b}) "
                            "MERGE (x)-[:RELATES_TO]->(y)",
                            a=a["name"], b=b["name"],
                        )
            # Link concepts to the source node
            for a in created:
                s.run(
                    "MATCH (kn:KgNode {node_id:$nid}), (cn:Concept {name:$cn}) "
                    "MERGE (kn)-[:MENTIONS]->(cn)",
                    nid=node_id, cn=a["name"],
                )
    finally:
        driver.close()

    print(f"Ingested {title!r} -> KgNode {node_id[:8]} | {len(chunks)} chunks, {len(concepts)} concepts")
    return 0


def cmd_link_similar(args: argparse.Namespace) -> int:
    driver = get_driver()
    try:
        with driver.session() as s:
            rows = s.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN c.chunk_id AS id, c.embedding AS emb").data()
        linked = 0
        with driver.session() as s:
            for r in rows:
                emb = r["emb"]
                # client-side top-k over the (small) chunk set
                scored = []
                for o in rows:
                    if o["id"] == r["id"]:
                        continue
                    dot = sum(x * y for x, y in zip(emb, o["emb"]))
                    na = sum(x * x for x in emb) ** 0.5
                    nb = sum(y * y for y in o["emb"]) ** 0.5
                    sim = dot / (na * nb) if na and nb else 0.0
                    scored.append((o["id"], sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                for oid, sim in scored[: args.top_k]:
                    if sim < 0.55:
                        continue
                    s.run(
                        "MATCH (a:Chunk {chunk_id:$a}), (b:Chunk {chunk_id:$b}) "
                        "MERGE (a)-[:SIMILAR {score:$s}]->(b)",
                        a=r["id"], b=oid, s=round(sim, 4),
                    )
                    linked += 1
    finally:
        driver.close()
    print(f"Created {linked} SIMILAR edges (top_k={args.top_k})")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    driver = get_driver()
    try:
        qvec = Embedder.embed(args.query)
        results = vector_query(driver, "Chunk", qvec, k=args.k)
        for node, score in results:
            text = node.get("text", "") if hasattr(node, "get") else str(node)
            snippet = (text or "")[:240].replace("\n", " ")
            print(f"[score={score:.3f}] {snippet}")
    finally:
        driver.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract knowledge from memories into Neo4j.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="chunk + embed + extract concepts from text")
    i.add_argument("--source", help="file path or source id")
    i.add_argument("--stdin", action="store_true")
    i.add_argument("--kind", default="markdown")
    i.add_argument("--title", default=None)
    i.set_defaults(func=cmd_ingest)

    l = sub.add_parser("link-similar", help="build SIMILAR edges between chunks")
    l.add_argument("--top-k", type=int, default=5)
    l.set_defaults(func=cmd_link_similar)

    s = sub.add_parser("search", help="semantic search over Chunk nodes")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
