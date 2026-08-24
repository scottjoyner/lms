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

## Round 2 — full chain restored (2026-08-24)

Layer-by-layer repair of the inference path (auto-router :8088 → assistx-api
projection → provider):

1. Router was stuck at projection generation 460 while the source advanced in
   jumps. Generation must advance EXACTLY +1; skipped numbers deadlock the
   router ("generation must advance exactly by one"). Fixed by rewinding
   FleetProjectionState.generation to router+1 and letting it re-apply.
2. Access path 127.0.0.1:1234 is only valid ON x1-370. The auto-router
   container probes from outside — repointed to host.docker.internal:1234.
3. Claim fence required AUTO_ROUTER_EXECUTOR_CLAIM_STATUS_URL +
   AUTO_ROUTER_ASSISTX_EXECUTOR_SERVICE_TOKEN (router) matching
   ASSISTX_EXECUTOR_CLAIM_STATUS_TOKEN (assistx-api). Wired with shared secret.
4. Result: POST /v1/chat/completions through the full chain returned **200**
   (7.6s round trip incl. queue).

Evidence-schema requirements learned (for future seeding):
- LoadedModelInstance needs: model_key, provider_model, admitted=true,
  expires_at_ts, artifact_fingerprint, quantization, context_length.
- AccessPath/CapacityObservation need approved_by AND approval_id.
- Projection generation advances exactly +1 per apply; never skip.

Remaining known issues:
- hermes self-tasks still exit_code_1 on some tasks: embeddings calls
  (/v1/embeddings) return 401 — caller lacks executor auth; separate wiring.
- xwing offline (operator training run); macbook-air re-bench pending its LM
  Studio LAN-serving toggle (LAN IP moved to 192.168.1.233).

## Round 3 — pipeline fully restored (2026-08-24)

Final layers fixed:
1. Router attempt timeout raised 45s→240s / deadline 300s (auto-router .env):
   slow-node cold JIT loads were tripping circuit breakers with empty
   httpx timeout errors ("unexpected provider error: ").
2. optiplex qwen3.5-0.8b pinned resident (TTL 31536000) — no per-task cold loads.
3. Executor token minting only allowed [task model] +
   ASSISTX_EXECUTOR_DEFAULT_MODEL_ALIASES env (was unpopulated → scope 401s).
   Set to the full serving catalog; api restarted.
4. swarm_memory indexer authenticates to router embeddings via internal token.
5. auto-router image rebuilt with executor scope-reject diagnostics
   (`executor scope reject: requested=… allowed=…` in logs).
6. hermes adapter logs stderr/stdout tails on non-zero exits.

Verified: zero exit_code_1 over a 3.5-min self-task window; chat completions
200 through router incl. 126s optiplex cold-load; embeddings 200; projection
gen 468 fresh with 4 providers.

Residual known-failures: tasks requesting ornith-1.0-35b (model files exist
only on offline xwing) fail dispatch until xwing returns.
