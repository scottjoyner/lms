# EXP-053 isolated SQLite high-water test fixture — frozen protocol

**Prospective synthetic-fixture protocol only; not a deployable witness or a production adapter.** Frozen before the implementation or any live model forward. The independent external witness, key custody and operator authority remain unapproved.

## Why this fixture is necessary

The existing mock witness uses a Python tuple and a thread lock. It exercises
request/response validation, but not durable storage, reconnects, fresh
connections, cross-process locking, primary-key uniqueness, or transaction
rollback. A real external witness must establish those properties.

## Fixture implementation boundaries

Implement strictly under `tests/fixtures/`, not as a callable production
`_external_cas` adapter. No network service, production endpoint,
discovery, deployment, NAS/Neo4j write, or new model run.

A synthetic test creates a temporary, private SQLite database explicitly,
using `BEGIN IMMEDIATE` transactions, `PRAGMA synchronous=FULL`,
`journal_mode=DELETE`, and a fixed exact schema:
- `witness_state`: exactly one row, `id=1`, `sequence INTEGER`,
  `claim_sha256 TEXT`, with genesis `(0, 64 lowercase zeros)`;
- `witness_claims`: `sequence` PRIMARY KEY, `execution_id_sha256 TEXT`
  UNIQUE, `nonce_hex TEXT` UNIQUE, `claim_sha256 TEXT` UNIQUE, and
  `previous_claim_sha256 TEXT`.

The private synthetic Ed25519 signing key is generated only in disposable
test context and is NOT persisted to SQLite. Each separately opened
connection reads the same SQLite state. A CAS request with the exact frozen
`REQUEST_FIELDS` and `REQUEST_SCHEMA` is accepted only if:
- the fresh nonce is 64 lowercase hex, the execution ID and hashes are
  64 lowercase hex;
- sequence is an exact positive integer, not boolean or float;
- sequence is `state.sequence+1`, expected_previous_sequence equals current
  state.sequence, previous_claim_sha256 equals state.claim_sha256;
- `nonce_hex`, `claim_sha256`, `execution_id_sha256` have never appeared.

Within the same transaction insert the unique claim row, update state,
and COMMIT. After the transaction completes, sign the exact frozen
`RESPONSE_FIELDS` with `status=committed`, copying all request bindings
and committed witness counter. The verifier separately checks this signature
and nonce before local journal mutation.

On invalid, duplicate, stale, concurrently contested, interrupted or
malformed requests: ROLLBACK and raise a refusal, never a signed success.
Do not automatically repair/delete reset state. A failed signing operation
AFTER a successful COMMIT must not roll back the durable counter; the caller
conservatively fails closed until independent recovery.

## Required synthetic negative controls

Check two separate connections and process-restart persistence; two racing
clients contending for the SAME state; reused execution ID with a different
sequence; reused nonce; reused claim SHA; wrong previous hash; malformed
field or bool sequence; a simulated local operator state/marker rollback
while the fixture DB survives; invalid witness signature; database write
failure and no half-committed state.

The SQLite file is **local, temporary, and controlled by the same test**.
This cannot establish a separate trust domain, a truly remote network
endpoint, independent host/operator custody, immutable off-host storage,
rollback prevention if the SQLite DB itself is snapshotted, or native
inference provenance. CI MUST use it only as a synthetic adversarial fixture.

All real witness root and adapter pins remain EMPTY, the default
`_external_cas` method DENIES. No independent human review or actual
EXP-053 model forward is authorized by these software tests.
