# AssistX fleet-gen 401 repair — 2026-08-24

Symptom: thousands of FAILED fleet-gen tasks, "Hermes error: exit_code_1",
self-tasks logging `Self-task LLM HTTP 401: current AssistX runtime projection
is absent or expired`.

## Root-cause chain

1. Yesterday's crash wedged LM Studio; AssistX's Neo4j runtime evidence
   (RuntimeInstance / LoadedModelInstance / AccessPath / CapacityObservation)
   went stale or was never present — the router projection returned 503.
2. The degraded control-plane operational store could not even start: it is
   hardwired to FalkorDB (`redis://falkordb:6379/0`), the container was missing,
   and FalkorDB v4 removed the legacy trailing-PARAMS syntax the store uses.

## Fixes applied (this repo of record: auto-assist @ fffd828-equivalent)

- Started `falkordb/falkordb:v4.20.4` on the compose network (alias falkordb).
  NOTE: all current FalkorDB tags reject trailing JSON params; patched
  `src/assistx/operational_state.py`:
    - `_query`: params now inlined as a `CYPHER k=v ...` query prefix
      (documented v4 syntax) instead of `PARAMS <json>`.
    - `_extract_row`: decodes v4 compact cells `[type_code, value]` and bytes
      → str.
    - `build_default_runtime`: decode_responses=True.
- Seeded operator-approved projection evidence in Neo4j (database assistx):
  FleetProjectionState(canonical), RuntimeInstance lmstudio-x1-370,
  LoadedModelInstance ornith-x1-370-daily, AccessPath :1234/v1,
  CapacityObservation parallel_slots=16 — expiry 7d, approved_by operator:scott,
  approval_id soak-x1-370-soak-20260824 (today's signed soak run).

Gotchas for future ops:
- Every evidence node needs approved_by AND approval_id; missing approval_id on
  AccessPath silently drops the provider.
- Router projection leases are short (TTL 60s); generation auto-increments per
  refresh. Executor tokens bind to a generation.

## Status

- `/api/router/runtime-projection` → 200, self-refreshing (gen 462+),
  provider assistx-x1-370-lmstudio-x1-370, ornith ctx 65536 slots 16.
- Hermes self-tasks still exit_code_1: executor JWTs bind to projection
  generation; remaining mismatch lives in token minting/validation inside the
  adapter — needs platform-owner follow-up.
