# EXP-053 isolated SQLite CAS fixture handoff — 2026-10-17

## Completed source-only research test

Added a **nondeployable** reference fixture under
`tests/fixtures/exp053_sqlite_cas_fixture.py`. It is not imported by
`research/exp053_external_witness.py` or any other production-facing
research replay entry point and opens no sockets or remote service.

Prospective fixture protocol was frozen before implementation:
`research/EXP053-SQLITE-WITNESS-FIXTURE-SPEC-2026-10-17.md`, SHA256
`1d4589ba32c26ab6abc5b5f8e3659f1e863dbd69a8ce443c1b1400ede426209a`.

The fixture makes actual SQLite `BEGIN IMMEDIATE` transactional
compare-and-swap updates with `synchronous=FULL`, `journal_mode=DELETE`,
one persistent high-water record, and unique constraints on
`sequence`, `nonce_hex`, `claim_sha256`, and
`execution_id_sha256`. Every synthetic CAS opens a fresh connection,
commits the new witness state and unique run claim together, and only
then produces a signed receipt using an ephemeral **test-only** Ed25519 key.

Tests verify: state persists across separate connections and a separate
Python process; a successful two-step chain; stale/forked or malformed
requests rejected without advancement; a new authorization cannot reuse
an earlier execution ID, claim digest, or nonce; competing threads and
separate POSIX processes agree **exactly one wins**; interrupted
transactions roll back; a deliberately lost response after committed
storage cannot be retried as if the first CAS never occurred.

A further integrated negative test feeds the isolated SQLite fixture
through the existing operator-claim verifier using test-time dependency
injection. After a successful claim, a simulated privileged actor restores
the **local operator journal** to genesis and deletes its local run marker.
The previously persisted SQLite witness rejects the replay. This is the
actual software interface being exercised, not just matching two fake
Python dictionaries.

## What was not done

This is still a **local test database** managed by the same experiment.
It is not an independently administered, off-host, rollback-resistant
service. If the SQLite file itself is restored from an earlier snapshot,
the fixture cannot stop replay. The request has no authenticated
operator-to-witness transport; malicious callers who can reach a future
backend could attempt counter exhaustion. A genuine service therefore
needs independent operator request authorization, authenticated transport,
source-pinned adapter, separate signer custody, off-host durability and
external high-water auditing. These are **NOT satisfied** here.

All actual witness root/adapter approval constants remain **EMPTY**,
and the real `_external_cas` continues to deny every request. The
previous human reviewer, checkpoint-scientific admission, native scorer,
and operator key/journal authorization pins also remain empty.

No native EXP-053 model forward, actual human review, model deployment,
RACE answer exposure, NAS/Neo4j write, training or scientific performance
claim was performed. PR #32 remains draft/unmerged.

## Regression correction uncovered by full hosted-scope tests

The initial full pytest run caught a test-only global-state leak. The
previous concurrent operator-claim test applied `unittest.mock.patch.object`
to module-level trust roots independently from both racing threads. Those
overlapping patches could restore a transient synthetic fingerprint instead
of the original EMPTY pin when tests were executed together.

This was fixed by applying ONE shared patch scope outside the two-thread
competition, with workers calling the implementation directly. A final
assertion verifies that all mock roots return to EMPTY after both workers
join. The new SQLite process-competition test uses `spawn` rather than
`fork` from a multithreaded pytest process; only an ephemeral 32-byte
**synthetic test key** is serialized into each subprocess.

After these corrections, the complete hosted-scope suite passes with
all trust roots disabled. No real operator, external witness, or native
model validation has been introduced.
