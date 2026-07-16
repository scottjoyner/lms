# Neo4j Schema Contract (lms)

This document is the **canonical schema contract** for the labels, properties, and
constraints that the `lms` graph-sync modules write into Neo4j. It exists so the
schema can be reconciled with `knowledge-graph` / `neo4j-mcp-server` without
reading the code. See `docs/LLD_UNIFIED_FLEET.md` W-76.

All writes go through the shared driver in `src/lms_agent_bench/neo4j.py`
(delegating to `graph_common.get_driver`). Modules:

- `fleet_graph_sync.py` → fleet health snapshots
- `session_graph_sync.py` → opencode session capture
- `knowledge_graph_sync.py` → chunks / concepts / KgNodes
- `delegation_graph_sync.py` → delegation records

## Databases

| DB | Purpose |
|---|---|
| `neo4j` (default) | unified memory graph (chunks, sessions, KgNodes, fleet snapshots) |
| `assistx` | control-plane (owned by AssistX; not written by lms) |

`NEO4J_DB` selects the target; lms defaults to `neo4j`.

## Labels written by lms

### Fleet health (`fleet_graph_sync.py`)
| Label | Key properties | Uniqueness constraint |
|---|---|---|
| `FleetSnapshot` | `snapshot_id`, `captured_at`, `previous_snapshot_id` | `fleet_snapshot_id` on `snapshot_id` |
| `FleetNodeState` | `node`, `hostname`, `ip`, `reachable`, `loaded_models` | `fleet_node_state` on `node` (+ snapshot) |
| `FleetModelState` | `node`, `model`, `status`, `tps` | `fleet_model_state` on `model`/`node` |
| `FleetLoadout` | `node`, `mount`, `ram_gib`, `max_concurrency` | `fleet_loadout_id` on `node` |
| `FleetTaskProfile` | `task_profile_id`, `kind` | `fleet_task_profile_id` on `task_profile_id` |

Relationships:
```
(FleetSnapshot)-[:HAS_NODE_STATE]->(FleetNodeState)
(FleetSnapshot)-[:HAS_MODEL_STATE]->(FleetModelState)
(FleetSnapshot)-[:HAS_LOADOUT]->(FleetLoadout)
(prev:FleetSnapshot)<-[:PREV_SNAPSHOT]-(FleetSnapshot)
```

### Sessions (`session_graph_sync.py`)
| Label | Key properties | Uniqueness constraint |
|---|---|---|
| `session` | `id`, `title`, `directory`, `project_id`, `summary_*` | `sg_session_id` on `id` |
| `message` | `id`, `session_id`, `role`, `content` | `sg_message_id` on `id` |
| `toolcall` | `id`, `session_id` | `sg_tool_id` on `id` |
| `reasoning` | `id`, `session_id` | `sg_part_id` on `id` |

Relationships: `(session)-[:HAS_MESSAGE]->(message)`, `(session)-[:HAS_TOOLCALL]->(toolcall)`, `(session)-[:HAS_REASONING]->(reasoning)`, `(message)-[:NEXT]->(message)`.

### Knowledge graph (`knowledge_graph_sync.py`)
| Label | Key properties | Uniqueness constraint |
|---|---|---|
| `Chunk` | `chunk_id`, `text`, `embedding[]`, `source` | `kg_chunk_id` on `chunk_id` |
| `Concept` | `name` | `kg_concept_name` on `name` |
| `KgNode` | `id`, `label`, `embedding[]` | `kg_node_id` on `id` |

Relationships: `(Chunk)-[:MENTIONS]->(Concept)`, `(Chunk)-[:HAS]->(KgNode)`, `(KgNode)-[:RELATES]->(KgNode)`, `(KgNode)-[:SIMILAR]->(KgNode)`.

Vector index (used by `graph_common.vector_query`): `chunk_embedding` on `Chunk.embedding`
(dimensions from `LOCAL_EMBED_DIM`, default 384).

### Delegation (`delegation_graph_sync.py`)
| Label | Key properties | Uniqueness constraint |
|---|---|---|
| `delegation` | `id`, `status`, `task`, `assignee` | `del_id` on `id` |
| `Agent` | `name` | `del_agent` on `name` |
| `DELEGATED` | (rel marker) | — |

Relationships: `(Agent)-[:DELEGATED]->(delegation)`.

## contract envelope shim (trace correlation)

Per `docs/LLD_UNIFIED_FLEET.md` §2, every graph write that originates from an
outbound fleet/event SHOULD carry a `correlation_id`. Until lms imports the shared
`assistx.contracts` envelope (cross-repo import deferred), lms mirrors the
required contract fields locally: any node properties that record an event carry a
`correlation_id` (UUID) so the writes can be linked to the unified `TraceGroup`
owned by AssistX. **Do not** introduce a second envelope shape — reuse this field
name verbatim.

## Reconciliation notes for knowledge-graph / neo4j-mcp-server

- `KgNode` here is the lms-side analogue of the knowledge-graph `KgNode`
  (`docs/LLD_UNIFIED_FLEET.md` §4.1). Same label + `embedding[]` + vector index
  name `chunk_embedding` are intentional so the two repos share one index.
- `Chunk`/`Concept` are lms-authored; knowledge-graph may also write `Chunk`.
  Coordinate uniqueness constraint `kg_chunk_id` so both writers use the same key.
- Fleet labels (`Fleet*`) are lms-only and safe to keep separate.
