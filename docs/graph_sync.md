# Fleet → Graph Sync

These modules mirror the live LM Studio fleet, Hermes sessions, extracted
knowledge, and agent delegations into the running Neo4j knowledge graph
(`bolt://localhost:7687`, db `neo4j`). Other agents query all of it
programmatically instead of scraping router health or stale JSON.

## Modules (in this repo)

| Module | What it does | Graph labels |
| --- | --- | --- |
| `fleet_graph_sync.py` | Publishes live fleet health/loadouts (reads auto-router `/api/fleet/nodes` + `/health` + `fleet_state.json`) | `FleetSnapshot`, `FleetNodeState`, `FleetModelState`, `FleetLoadout` |
| `session_graph_sync.py` | Mirrors opencode (Hermes) sessions from `~/.local/share/opencode/opencode.db` | `session`, `message`, `reasoning`, `toolcall` |
| `knowledge_graph_sync.py` | Chunks + embeds text/memories, extracts concepts | `KgNode`, `Chunk`, `Concept` (+ `SIMILAR`, `RELATES_TO`) |
| `delegation_graph_sync.py` | Agent coordination inbox (create/claim/complete/find) | `delegation`, `Agent` (+ `DELEGATED_TO`) |

Shared helpers (driver, local embedder, vector query) live in `graph_common.py`.
Embeddings use the local `all-MiniLM-L6-v2` model (offline, 384-dim) and reuse
the existing `chunk_embedding` Neo4j vector index.

## Persisting the sync (systemd --user)

The old `fleet-task-dispatcher.service` is broken (source missing) and has been
replaced by these units in `~/.config/systemd/user/`:

- `fleet-graph-sync.service` — runs `fleet_graph_sync.py watch --sleep 60`,
  refreshing fleet health into the graph every minute.
- `fleet-session-sync.timer` + `fleet-session-sync.service` — incremental
  session sync every 5 min (`session_graph_sync.py sync --since-days 1`).

Enable:

```bash
systemctl --user daemon-reload
systemctl --user enable --now fleet-graph-sync.service
systemctl --user enable --now fleet-session-sync.timer
```

Set `NEO4J_PASSWORD=knowledge_graph_2026` in the unit Environment (already done).

## Common commands

```bash
# fleet health (one-shot)
python3 fleet_graph_sync.py publish

# sessions
python3 session_graph_sync.py sync --since-days 7
python3 session_graph_sync.py sync --watch --sleep 300

# knowledge
cat memory.md | python3 knowledge_graph_sync.py ingest --stdin --kind memory
python3 knowledge_graph_sync.py link-similar --top-k 5
python3 knowledge_graph_sync.py search "how does the fleet router route"

# delegation (agent coordination)
python3 delegation_graph_sync.py create --goal "..." --from-agent build --tags fleet
python3 delegation_graph_sync.py find "fleet loadout planning"
python3 delegation_graph_sync.py claim --id <id> --agent orchestrator
python3 delegation_graph_sync.py list --status open
```

## Querying from another agent

```cypher
// current fleet health
MATCH (snap:FleetSnapshot)-[:HAS_NODE_STATE]->(n:FleetNodeState)
WHERE snap.snapshot_id = <latest> RETURN n.node_name, n.online, n.loaded_models

// what a node currently serves
MATCH (snap:FleetSnapshot)-[:HAS_MODEL_STATE]->(m:FleetModelState)
WHERE m.node_name = 'x1-370' AND m.loaded RETURN m.model_id

// hermes prior sessions on a directory
MATCH (s:session) WHERE s.directory CONTAINS 'lms' RETURN s.title, s.session_id

// related delegations
MATCH (d:delegation) WHERE d.status = 'open' RETURN d.goal, d.id
```
