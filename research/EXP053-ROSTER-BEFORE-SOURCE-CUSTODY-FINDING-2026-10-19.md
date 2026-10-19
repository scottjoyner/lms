# EXP-053 — pinned owner reviewer roster must precede licensed corpus access

**2026-10-19. Source-only engineering finding. NO_GO_RESEARCH_ONLY.**

## Reproduced flaw and scope

After a previous patch started checking that the configured reviewer-roster
SHA256 was syntactically well formed, `verified_admission()` still hashed the
full licensed RACE source before reading and verifying the actual owner-pinned
roster. With a fake but plausible `"a"*64` pin, a synthetic source-hash hook
proved that the code called the source hasher before discovering the roster was
missing. This is a **premature sensitive read and wasted I/O**, not evidence
of an identity or signature bypass. No licensed RACE bytes were accessed in
the reproduction.

## Change and verification

`verified_admission()` now first refuses absent/malformed owner pins, then
hashes and parses the preregistered, **source-free public manifest**. Its
previously pinned packet SHA256 is passed to the unchanged `load_roster()`
validator, which checks the separately pinned 0600 private roster's exact
bytes, packet binding, blind reviewer assignments and distinct public keys.
Only after that verification does the function hash the licensed source and
verify protocol, private packet, sealed answer key, signed A/B human reviews
and any independent third-party adjudication. The actual private packet SHA
must still match the public manifest; roster trust is thus transitively bound.

Tests use a real source-free public manifest, a disposable fake source path,
and synthetic pseudonyms/public-key values, never a human signer key or licensed
RACE passage. Instrumented tests prove failure **before source hashing** for
missing roster, wrong pin, wrong packet digest, world-readable file, invalid
blind assignments or damaged public manifest. A valid-form synthetic roster
reaches the real source SHA requirement but receives `HOLD` on a deliberate
fake-source digest mismatch. No private packet, key, or review form is read.

## Unchanged governance requirements

The live `APPROVED_REVIEWER_ROSTER_SHA256` is empty, and this patch does not
populate it. No independently verified person has been enrolled, no signed
adjudication has been accepted, and the >=40 source-distinct human-review gate
remains unfulfilled. A 64-hex pin alone is never a human approval. External
independent custody, source licensing, native rollback tests and the model
weights/run receipt approval are separately blocked. No Git push, merge,
model forward or production deployment is authorized.

Reproduce:

`python3 -m pytest -q tests/test_exp053_roster_before_source.py`
