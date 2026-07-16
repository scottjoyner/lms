"""Smoke tests for lms graph-sync modules (mocked Neo4j, no live DB).

These verify the shared driver path and that the graph modules import and build
their schema/constraints against a fake driver. They do NOT require a running
Neo4j. See docs/LLD_UNIFIED_FLEET.md W-73.
"""

import importlib

import pytest

import lms_agent_bench.neo4j as neo4j
from lms_agent_bench import graph_common


class _FakeSession:
    def __init__(self, db=None):
        self.db = db
        self.ran = []

    def run(self, stmt, **params):
        self.ran.append(stmt)
        return self

    def data(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, *a, **k):
        self.sessions = []

    def session(self, database=None):
        s = _FakeSession(database)
        self.sessions.append(s)
        return s

    def close(self):
        pass


def test_neo4j_config_reads_env(monkeypatch):
    monkeypatch.setattr(graph_common, "NEO4J_URI", "bolt://example:7687")
    monkeypatch.setattr(graph_common, "NEO4J_USER", "neo4j")
    monkeypatch.setattr(graph_common, "NEO4J_PASSWORD", "secret")
    monkeypatch.setattr(graph_common, "NEO4J_DB", "neo4j")
    cfg = neo4j.neo4j_config()
    assert cfg["uri"] == "bolt://example:7687"
    assert cfg["password"] == "secret"


def test_graph_common_get_driver_requires_password(monkeypatch):
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        graph_common.get_driver()


def test_shared_get_driver_uses_fake(monkeypatch):
    fake = _FakeDriver()
    monkeypatch.setattr(neo4j, "_graph_common_get_driver", lambda: fake)
    assert neo4j.get_driver() is fake


def test_graph_modules_import_cleanly():
    """All four graph-sync modules must import without a live Neo4j."""
    for mod in (
        "lms_agent_bench.fleet_graph_sync",
        "lms_agent_bench.session_graph_sync",
        "lms_agent_bench.knowledge_graph_sync",
        "lms_agent_bench.delegation_graph_sync",
    ):
        importlib.import_module(mod)


def test_ensure_constraints_runs_against_fake_driver(monkeypatch):
    fake = _FakeDriver()
    monkeypatch.setattr(neo4j, "_graph_common_get_driver", lambda: fake)
    # ensure_constraints lives in graph_common and runs statements on a session.
    graph_common.ensure_constraints(fake, ["CREATE CONSTRAINT x FOR (n:Y) REQUIRE n.id IS UNIQUE"])
    assert any("CREATE CONSTRAINT" in s for s in fake.sessions[0].ran)
