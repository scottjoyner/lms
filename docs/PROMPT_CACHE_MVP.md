# Prompt-cache registry MVP

This release implements the first safe prompt-prefix/KV-cache registry slice.
It is **record-only**: it can identify, store, verify, and report reusable cache
candidates, but it never restores KV state, skips prefill tokens, changes routing,
or admits a loadout.

## Components

- `prompt_cache_identity.py`
  - exact loadout compatibility manifests;
  - tokenizer, chat-template, system-prompt, tool-schema, adapter, multimodal,
    preprocessing, engine serialization ABI, and device-layout identity;
  - deterministic Merkle-style prefix blocks from exact token IDs.
- `prompt_cache_store.py`
  - local content-addressed opaque payload storage;
  - atomic staging, `fsync`, hard-link publication, SHA-256 verification,
    deduplication, and quarantine.
- `prompt_cache_registry.py`
  - backend-neutral `PromptCacheRegistry` contract;
  - SQLite implementation with WAL, foreign keys, indexes, namespace isolation,
    expiry filtering, longest-prefix lookup, and observation statistics;
  - `PromptCacheRecorder`, which explicitly never restores payloads.
- `prompt_cache_cli.py`
  - installed as `lms-prompt-cache`.

The default root is `~/.cache/lms-kv`:

```text
~/.cache/lms-kv/
  registry.sqlite3
  blobs/sha256/<2>/<2>/<sha256>.blob
  staging/
  quarantine/
```

## Initialize

```bash
lms-prompt-cache --root ~/.cache/lms-kv init
```

## Token-ID input

Commands consume exact token IDs rather than normalized prompt text:

```json
{
  "token_ids": [1, 15043, 29871, 13]
}
```

The token IDs must be produced with the exact tokenizer and rendered prompt
associated with the supplied compatibility hashes.

## Register an opaque engine-native artifact

```bash
lms-prompt-cache --root ~/.cache/lms-kv register-artifact \
  --loadout loadout.json \
  --token-ids prompt-token-ids.json \
  --artifact slot-cache.bin \
  --namespace project/example \
  --node-id x1-370 \
  --serialization-format llama-slot-cache \
  --serialization-version 1 \
  --engine-serialization-abi llama.cpp-slot-v1 \
  --tokenizer-sha256 <sha256> \
  --chat-template-sha256 <sha256> \
  --system-prompt-sha256 <sha256> \
  --tool-schema-sha256 <sha256>
```

The payload is copied into the content-addressed store and must pass SHA-256
verification before its metadata is registered as `ready`.

## Observe a request

```bash
lms-prompt-cache --root ~/.cache/lms-kv observe \
  --loadout loadout.json \
  --token-ids request-token-ids.json \
  --namespace project/example \
  --node-id xwing \
  --engine-serialization-abi llama.cpp-slot-v1 \
  --tokenizer-sha256 <sha256> \
  --chat-template-sha256 <sha256> \
  --system-prompt-sha256 <sha256> \
  --tool-schema-sha256 <sha256>
```

A verified exact match reports `candidate_hit`. It still reports:

```json
{
  "mode": "record_only",
  "restoration_attempted": false,
  "tokens_skipped": 0,
  "admission": {"admitted": false}
}
```

This prevents estimated savings from being mistaken for measured restored-cache
savings.

## Statistics and verification

```bash
lms-prompt-cache --root ~/.cache/lms-kv stats \
  --namespace project/example

lms-prompt-cache --root ~/.cache/lms-kv verify \
  --payload-sha256 sha256:<digest>
```

## Safety behavior

A lookup requires all of the following to match:

- exact token-prefix block hashes;
- exact compatibility hash;
- namespace;
- ready state;
- non-expired artifact;
- valid local payload SHA-256.

A corrupt candidate is marked `quarantined`, moved out of the blob tree, and
reported as `verification_failed`. A miss or registry failure must fall back to
normal prefill in the eventual runtime adapter.

## Not implemented yet

- automatic KV restoration;
- llama.cpp slot endpoint integration;
- vLLM, SGLang, or LMCache adapters;
- FalkorDB shared-registry adapter;
- cross-node payload transfer;
- measured prefill/TTFT savings;
- cache-driven routing or admission.

Those capabilities require physical runtime evidence and remain separate,
reviewed phases.
