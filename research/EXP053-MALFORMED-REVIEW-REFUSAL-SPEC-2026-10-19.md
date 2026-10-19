# EXP-053 — malformed signed-review input refusal (research-only)

**Frozen source-only negative-test specification, October 19, 2026.**
`NO_GO_RESEARCH_ONLY`. The existing signer, frozen candidate cohort, owner
signer roster, scientific plan, witness roots, and model paths remain closed.

Confirmed locally using synthetic public-safe objects: review `case_id=[]`
or `case_id={}` causes uncaught Python `TypeError` in `decisions_for` during
set/dict membership. An unhashable reviewer public-key object in `trust`
causes the same uncaught exception at `len(set(trust.values()))` in
`verify_signed`. This is **error handling**, not verified privilege escalation
or a signature bypass. Unexpected tracebacks may contain implementation detail
and complicate offline human handoff.

Required minimal repair:

- Before building an index, require the source-free packet to have a cases
  array of unique, nonempty **string** case IDs, each associated with a
  dictionary containing a string source article. Do not inspect or emit case
  content in tests. Refuse duplicate/unhashable packet IDs with deterministic
  `HOLD EXP053`, not uncaught exceptions.
- Before checking decision IDs against the indexed packet, require each
  decision's `case_id` to be a nonempty string. On malformed, duplicate,
  foreign or missing IDs fail closed; preserve the exact-subset requirement
  for independent A/B and third adjudicator forms. Required-ID subsets must
  also consist of valid string IDs, with no duplicates supplied as a list.
- Before checking key uniqueness, require reviewer-trust mapping keys and
  values to be nonempty strings. Duplicate string public-key values remain
  rejected. Malformed list/dict/numeric values must produce sanitized HOLD,
  not unhashable-object errors or raw traceback. Cryptographic signature
  validation is not weakened.
- All synthetic tests use invented one-line articles, fake review data,
  optionally ephemeral test Ed25519 keys, and no real human signature,
  licensed RACE text, enrollment records or output writes. Preserve existing
  valid reviewer and adjudicator behavior, and all source/authorization pins.
- Do not change the public manifest, source schema, cohort case order,
  threshold of 40 human adjudications, server admission authority, or PR #32.

A passing negative test does NOT demonstrate an actual independently reviewed
article or independent key custody. No live production or scientific execution
is authorized by this patch.
