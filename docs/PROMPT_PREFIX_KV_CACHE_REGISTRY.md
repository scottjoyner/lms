# Prompt-prefix graph and KV-cache artifact registry

## Decision

Use a two-plane architecture:

1. **Control plane:** a graph or lightweight relational registry containing prompt-prefix identities, compatibility metadata, lineage, locations, reuse observations, TTLs, and policies.
2. **Data plane:** engine-native KV-cache artifacts stored outside the graph in RAM, local NVMe, shared filesystem, Redis/Valkey-compatible storage, LMCache, or another content-addressed blob store.

Do not store large raw KV tensors directly as FalkorDB node properties. FalkorDB is the catalog and relationship engine; the payload store owns the bytes.

## Recommended rollout

### Phase 1: SQLite registry plus local content-addressed files

Make SQLite the default backend for the first implementation because it:

- is in the Python standard library;
- supports the project's Python 3.10+ floor;
- requires no daemon;
- supports WAL, transactions, indexes, recursive CTEs, and durable local files;
- lets the cache semantics stabilize before adding distributed coordination.

Store engine-native artifacts beneath a content-addressed root:

```text
~/.cache/lms-kv/
  blobs/sha256/ab/cd/<artifact-sha256>.bin
  staging/
  quarantine/
  registry.sqlite3
```

### Phase 2: FalkorDB adapter

Add a FalkorDB backend with the same registry interface when shared fleet discovery is needed. Use:

- FalkorDBLite for local development and CI where Python 3.12 is available;
- a normal persistent FalkorDB deployment for shared production metadata.

FalkorDB remains the metadata and graph layer, not the raw tensor store.

### Phase 3: engine-specific KV movement

Add adapters rather than one fake portable KV format:

- `llama.cpp`: slot prompt-cache save/restore files and in-memory prefix reuse;
- `vLLM`: automatic prefix caching and, where useful, LMCache connectors;
- `SGLang`: native radix/prefix cache and LMCache-compatible movement where supported;
- LM Studio or other servers: only when the runtime exposes an exact, testable cache API.

Engine-native serialized KV artifacts are treated as opaque and non-portable unless an adapter proves compatibility.

## Core graph model

```text
(:PromptSequence)
(:PrefixBlock)
(:PromptSegment)
(:Tokenization)
(:Tokenizer)
(:ChatTemplate)
(:SystemPrompt)
(:ToolSchema)
(:ModelArtifact)
(:Loadout)
(:EngineBuild)
(:KVArtifact)
(:StorageLocation)
(:FleetNode)
(:CacheObservation)
(:Policy)
```

Relationships:

```text
(PromptSequence)-[:STARTS_WITH]->(PrefixBlock)
(PrefixBlock)-[:NEXT]->(PrefixBlock)
(PrefixBlock)-[:ENCODES]->(PromptSegment)
(PrefixBlock)-[:TOKENIZED_AS]->(Tokenization)
(Tokenization)-[:USES]->(Tokenizer)
(PromptSequence)-[:USES_TEMPLATE]->(ChatTemplate)
(PromptSequence)-[:USES_SYSTEM_PROMPT]->(SystemPrompt)
(PromptSequence)-[:USES_TOOL_SCHEMA]->(ToolSchema)
(KVArtifact)-[:MATERIALIZES]->(PrefixBlock)
(KVArtifact)-[:FOR_LOADOUT]->(Loadout)
(Loadout)-[:LOADS]->(ModelArtifact)
(Loadout)-[:RUNS_ON]->(EngineBuild)
(KVArtifact)-[:STORED_AT]->(StorageLocation)
(StorageLocation)-[:ATTACHED_TO]->(FleetNode)
(CacheObservation)-[:REQUESTED]->(PromptSequence)
(CacheObservation)-[:HIT|MISSED|RESTORED|EVICTED]->(KVArtifact)
(PromptSequence)-[:FORKS_FROM]->(PromptSequence)
(Policy)-[:GOVERNS]->(KVArtifact)
```

A prompt sequence should form a Merkle-style prefix DAG rather than duplicating every full conversation. Shared system prompts, tool definitions, repository context, and conversation histories then converge on the same prefix blocks.

## Prefix block identity

Token IDs, not normalized text, define reusable model input. Divide a tokenized sequence into fixed-size blocks, initially 256 tokens.

```text
compatibility_hash = SHA256(canonical compatibility manifest)
block_hash[0] = SHA256(compatibility_hash || token_ids[0:256])
block_hash[n] = SHA256(compatibility_hash || block_hash[n-1] || token_ids[n])
sequence_hash = final block hash plus exact token count
```

This creates deterministic prefix lookup and prevents a block from being reused under an incompatible loadout.

## Compatibility manifest

The KV compatibility hash must include every field that can change prefill activations or serialized layout:

```text
model content SHA-256
model architecture metadata
tokenizer artifact SHA-256
chat-template SHA-256
system-prompt SHA-256
tool-schema SHA-256
adapter or LoRA SHA-256
multimodal encoder and preprocessing identity when present
engine name, version, build commit, and KV serialization ABI
backend and device layout when the artifact is backend-specific
weight quantization and tensor overrides
KV key dtype and value dtype
attention type and KV-head layout
configured context and slot context
RoPE configuration and position handling
sliding-window configuration
attention-mask mode
parallel-slot layout when serialized artifacts depend on it
```

Sampling temperature, top-p, and similar decode-only controls do not change the prompt-prefix KV state. Record them on observations, but do not put them into the prefix compatibility hash unless the cached object includes generated continuation tokens whose identity must also be fixed.

## Artifact record

Each `KVArtifact` should contain only metadata and a payload reference:

```json
{
  "artifact_id": "sha256:...",
  "prefix_block_hash": "sha256:...",
  "compatibility_hash": "sha256:...",
  "engine": "llama.cpp",
  "engine_build": "git:...",
  "serialization_format": "llama-slot-cache",
  "serialization_version": 1,
  "token_count": 8192,
  "key_dtype": "q8_0",
  "value_dtype": "q4_0",
  "size_bytes": 536870912,
  "payload_uri": "file:///.../sha256/...bin",
  "payload_sha256": "sha256:...",
  "created_at_utc": "...",
  "last_verified_at_utc": "...",
  "expires_at_utc": "...",
  "state": "ready",
  "sensitivity": "private",
  "owner_namespace": "user-or-project",
  "admission": {"admitted": false}
}
```

Artifact lifecycle:

```text
staging -> verified -> ready -> stale -> evicted
                  \-> quarantined
```

Only an atomically written artifact with a matching payload hash may enter `ready`.

## Lookup algorithm

For each request:

1. Canonicalize the exact ordered message list, system prompt, tools, attachments, and template inputs.
2. Tokenize with the exact tokenizer artifact.
3. Calculate the compatibility hash.
4. Calculate chained block hashes.
5. Find the longest matching ready prefix whose compatibility hash is exact.
6. Ask the engine adapter whether the artifact can be restored on the current node and build.
7. Restore or use the engine's in-memory prefix cache.
8. Process only the unmatched suffix.
9. Record hit/miss, bytes loaded, tokens skipped, TTFT delta, load time, and any verification failure.
10. Optionally materialize a new artifact at a policy-approved block boundary.

A semantic vector match may recommend likely related prefixes, but it must never authorize raw KV reuse. KV reuse requires exact token-prefix and compatibility matches.

## Policy and privacy

Prompt caches can contain secrets, personal data, proprietary source, and tool outputs. Default policy:

- prompt text is optional and encrypted when retained;
- token IDs and hashes may be stored without plaintext;
- raw KV payloads inherit the sensitivity of the originating prompt;
- cache sharing is namespace-scoped;
- private user or project prefixes never cross namespaces;
- public reusable prefixes require explicit classification;
- tool outputs with volatile credentials are non-cacheable;
- deletion must remove graph metadata and payload bytes;
- TTL and byte quotas apply per namespace and node;
- cache poisoning or restore failure quarantines the artifact;
- no artifact changes live routing or admission by itself.

The cache key must include the system prompt and tool schema. Otherwise an apparently identical user prompt could restore state created under different instructions or capabilities.

## Cache value scoring

Do not retain every prefix. Calculate a value score from observed reuse:

```text
saved_prefill_seconds
x expected_reuse_probability
x successful_restore_rate
x task-value weight
-
load_seconds
-
byte_cost weight
-
network-transfer cost
-
staleness and privacy penalties
```

Track separately:

```text
hit count
near-miss count
restoration failures
tokens skipped
prefill milliseconds saved
TTFT reduction
artifact load bandwidth
bytes per saved prefill second
last access
reuse by node and loadout
```

Eviction should be weighted-LRU or GreedyDual-Size-Frequency rather than simple LRU, because large long-context artifacts can be expensive to keep but also extremely valuable when repeatedly reused.

## SQLite MVP schema

```sql
CREATE TABLE loadouts (
    compatibility_hash TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE prefix_blocks (
    block_hash TEXT PRIMARY KEY,
    compatibility_hash TEXT NOT NULL,
    parent_block_hash TEXT,
    token_count INTEGER NOT NULL,
    token_ids_blob BLOB,
    token_ids_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (compatibility_hash) REFERENCES loadouts(compatibility_hash),
    FOREIGN KEY (parent_block_hash) REFERENCES prefix_blocks(block_hash)
);

CREATE TABLE kv_artifacts (
    artifact_id TEXT PRIMARY KEY,
    block_hash TEXT NOT NULL,
    payload_uri TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    serialization_format TEXT NOT NULL,
    serialization_version TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    state TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    last_verified_at_utc TEXT,
    expires_at_utc TEXT,
    FOREIGN KEY (block_hash) REFERENCES prefix_blocks(block_hash)
);

CREATE TABLE cache_observations (
    observation_id TEXT PRIMARY KEY,
    artifact_id TEXT,
    requested_sequence_hash TEXT NOT NULL,
    outcome TEXT NOT NULL,
    matched_tokens INTEGER NOT NULL,
    tokens_skipped INTEGER NOT NULL,
    restore_ms REAL,
    prefill_ms_saved REAL,
    ttft_ms REAL,
    node_id TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    details_json TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES kv_artifacts(artifact_id)
);

CREATE INDEX idx_prefix_parent ON prefix_blocks(parent_block_hash);
CREATE INDEX idx_artifact_block_state ON kv_artifacts(block_hash, state);
CREATE INDEX idx_observation_artifact_time ON cache_observations(artifact_id, created_at_utc);
```

## FalkorDB query examples

Longest known path candidate:

```cypher
MATCH path=(root:PrefixBlock {block_hash: $first})-[:NEXT*0..]->(tail:PrefixBlock)
WHERE tail.compatibility_hash = $compatibility_hash
  AND tail.block_hash IN $request_block_hashes
OPTIONAL MATCH (artifact:KVArtifact)-[:MATERIALIZES]->(tail)
WHERE artifact.state = 'ready'
  AND artifact.namespace = $namespace
RETURN tail.block_hash, tail.token_count, artifact
ORDER BY tail.token_count DESC
LIMIT 1
```

Hot shared prefixes:

```cypher
MATCH (observation:CacheObservation)-[:HIT]->(artifact:KVArtifact)-[:MATERIALIZES]->(block:PrefixBlock)
WHERE observation.created_at_utc >= $window_start
RETURN block.block_hash,
       artifact.artifact_id,
       count(observation) AS hits,
       sum(observation.prefill_ms_saved) AS saved_ms,
       artifact.size_bytes AS bytes
ORDER BY saved_ms DESC
LIMIT 100
```

## Failure boundaries

The registry must fail closed when:

- payload or model hashes differ;
- tokenizer, template, tools, adapter, RoPE, KV dtype, or engine ABI differ;
- the artifact is incomplete, expired, stale, or quarantined;
- namespace authorization fails;
- the target runtime cannot prove compatible restore support;
- the restored slot fails a deterministic canary;
- the database is unavailable.

Database failure must only cause a cache miss and normal prefill. It must never prevent inference or silently accept an unverified artifact.

## Initial deliverable

The first implementation should contain:

1. a backend-neutral `PromptCacheRegistry` interface;
2. a SQLite implementation;
3. a content-addressed local artifact store with atomic rename and SHA-256 verification;
4. a `llama.cpp` slot save/restore adapter behind an experimental flag;
5. record-only instrumentation for hit candidates, misses, token overlap, and estimated savings;
6. deterministic tests proving compatibility mismatches always miss;
7. no automatic sharing, routing, or admission.

After local evidence demonstrates useful hit rates and correct restores, add the FalkorDB backend and fleet-aware placement policy.
