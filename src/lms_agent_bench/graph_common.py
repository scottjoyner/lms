#!/usr/bin/env python3
"""graph_common.py — shared Neo4j + embedding helpers for the lms graph tooling.

All lms->graph modules (fleet_graph_sync, session_graph_sync, knowledge_graph_sync,
delegation_graph_sync) import from here so driver creation, schema constraints,
and the local embedding model are configured in exactly one place.
"""
from __future__ import annotations

import os
from typing import List, Optional

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "knowledge_graph_2026")
NEO4J_DB = os.environ.get("NEO4J_DB", "neo4j")

LOCAL_EMBEDDING_MODEL = os.environ.get("LMS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LOCAL_EMBED_DIM = int(os.environ.get("LMS_EMBED_DIM", "384"))


def get_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def ensure_constraints(driver, statements: List[str]) -> None:
    with driver.session(database=NEO4J_DB) as s:
        for stmt in statements:
            try:
                s.run(stmt)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] constraint failed: {exc}", file=__import__("sys").stderr)


class Embedder:
    """Lazily-loaded local sentence-transformers embedder (offline-friendly)."""

    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        return cls._model

    @classmethod
    def embed(cls, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * LOCAL_EMBED_DIM
        vec = cls.model().encode(text, normalize_embeddings=True)
        return vec.tolist()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def vector_query(driver, label: str, query_vec: List[float], k: int = 5, db: Optional[str] = None):
    """Approximate nearest neighbours via the existing Neo4j vector index.

    Falls back to client-side cosine scan if the vector index is unavailable.
    """
    db = db or NEO4J_DB
    index_name = {
        "Chunk": "chunk_embedding",
    }.get(label)
    with driver.session(database=db) as s:
        if index_name:
            try:
                recs = s.run(
                    f"CALL db.index.vector.queryNodes('{index_name}', $k, $qvec) "
                    "YIELD node, score RETURN node, score",
                    k=k,
                    qvec=query_vec,
                ).data()
                return [(r["node"], r["score"]) for r in recs]
            except Exception:
                pass
        # Fallback: pull embeddings and rank client-side (small fleets only).
        rows = s.run(f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL RETURN n, n.embedding AS emb").data()
        scored = []
        for r in rows:
            emb = r.get("emb")
            if emb:
                scored.append((r["n"], cosine(query_vec, emb)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
