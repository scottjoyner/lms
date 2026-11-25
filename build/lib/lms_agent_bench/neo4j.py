"""neo4j.py — single source of truth for Neo4j driver/config in lms.

Wraps :func:`lms_agent_bench.graph_common.get_driver` so every graph-sync
module (fleet_graph_sync, session_graph_sync, knowledge_graph_sync,
delegation_graph_sync) reuses one driver factory instead of each re-declaring
NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD / NEO4J_DB. This is the W-69
de-duplication of the Neo4j client.

Secrets are never hardcoded here — they come from the environment (see
docs/LLD_UNIFIED_FLEET.md W-71).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from lms_agent_bench.graph_common import (
    NEO4J_DB,
    get_driver as _graph_common_get_driver,
)


def neo4j_config() -> Dict[str, str]:
    """Return the resolved Neo4j connection config (no secrets defaulted).

    Reads the values live from :mod:`lms_agent_bench.graph_common` so environment
    overrides are reflected even after import.
    """
    from lms_agent_bench import graph_common

    return {
        "uri": graph_common.NEO4J_URI,
        "user": graph_common.NEO4J_USER,
        "password": graph_common.NEO4J_PASSWORD,
        "db": graph_common.NEO4J_DB,
    }


def get_driver():
    """Return a configured Neo4j driver (delegates to graph_common)."""
    return _graph_common_get_driver()


def session(driver: Any, db: Optional[str] = None):
    """Open a session against ``db`` (defaults to NEO4J_DB)."""
    return driver.session(database=db or NEO4J_DB)
