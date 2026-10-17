# EXP-053 external high-water anti-rollback research handoff — 2026-10-17

## Finding and research-only software change

The local operator single-use ledger and execution-ID markers prevent
duplicates while their private filesystem has not been rolled back.
A privileged actor could restore a journal snapshot together with markers
and then reuse a previous, valid signed claim.

The prospective contract
`research/EXP053-EXTERNAL-HIGHWATER-WITNESS-SPEC-2026-10-17.md` was frozen
before implementation at SHA256
`3aa0547ee07a9c271fd67b6542544a147790fef052337f1ce1764a8c2cbb5f1e`.

New `research/exp053_external_witness.py` requires an independently
pinned Ed25519 witness key and separately reviewed external CAS adapter.
It generates a fresh 256-bit random challenge nonce on each claim, asks
an independently administered witness to atomically advance its OWN
persisted monotonic sequence and accepted signed operator-claim SHA256,
then validates a complete, canonical signed `committed` receipt bound
to that nonce, score-run ID, exact claim, previous hash, and counter.
The witness key must be different from scorer and operator signing keys.

`research/exp053_operator_run.py` invokes this future external witness
**after** checking the operator-signed authorization and locked local
journal sequence, but **before** writing the local used-ID marker or
monotonic journal state. An unavailable, stale, malformed, forged or
non-committed witness returns an explicit HOLD with no local claim advance.
If remote CAS commits and local mutation fails, the system conservatively
HOLDs until independent operator recovery; no auto-reset, key generation
or remote bypass is implemented.

The ONLY future aggregate additions are the public-safe signed witness
receipt SHA256 and committed external sequence number. The raw operator
and witness credentials, signed receipts, native runtime logs and private
model scores remain off GitHub.

### Verified in synthetic tests, NOT in production

The expanded test suite runs an ephemeral, **in-memory mock CAS**
alongside disposable local operator files and synthetic signer keys.
It tests a simulated privilege rollback: after accepting run 1, restore
the LOCAL operator journal to 0, remove the local used-ID marker and
resubmit the same signed authorization. The unchanged mock external CAS
still reports that high-water is 1, so the replay is denied.
Additional tests reject stale challenge nonces, invalid signatures,
wrong counter/status/execution ID, unexpected fields, missing backend,
and duplicates or races.

These tests establish the **software refusal contract** only.
They DO NOT demonstrate that a real independent service exists, that
a CAS operation is durable outside x1-370, or that a trusted operator
has approved any signing key or counter.

### Hard NO-GO parameters

The two new authority pins remain INTENTIONALLY EMPTY:

- `APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256=""`
- `APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256=""`

The default `_external_cas` adapter is an explicit refusal with NO
network requests, dynamic endpoint selection or implicit mock fallback.

All previous real authorization pins for the human reviewer roster,
scientific weight identity, scorer key/source, and local operator
key/journal likewise remain empty.

**The remaining material dependency is a separately administered
physical or remote atomic CAS witness, with independent operator
review of its source, key custody, append-only high-water and durable
native receipts.** Even this cannot by itself prove model inference
or correct log probabilities. There are zero independently approved
RACE articles and zero EXP-053 model neural forward passes. PR remains
DRAFT and unmerged; no model training, deployment, NAS/Neo4j writes
or scientific performance assertions.
