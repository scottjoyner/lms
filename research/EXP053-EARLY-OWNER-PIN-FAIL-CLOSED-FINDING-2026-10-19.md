# EXP-053 — early owner reviewer-roster HOLD before sensitive source read

**2026-10-19. Synthetic read-boundary test, NO_GO_RESEARCH_ONLY.**

## Reproduced consequence

`verified_admission` previously performed a SHA256 of the full licensed RACE
source, then read the manifest, blinded packet and sealed exam answer key,
**before** checking whether an independent human study owner had approved a
reviewer roster. On the real research checkout, the configured approval SHA256
is the empty string. When approval is absent, none of the private reads are
necessary, and admission is guaranteed to fail later in `load_roster`.
A synthetic test with a non-RACE file and a deliberately raising source-hash
stub showed the old code called the hash function before returning a HOLD.
No real source contents were opened for the reproduction.

## Limited code correction

`verified_admission` now rejects a roster approval pin that is not exactly 64
lowercase hex characters **at function entry**, before probing or hashing
any corpus/manifest/private packet/reviewer/answer-key paths. All malformed
approval shapes (empty, upper case, too short/long, nonstring) return a
sanitized `HOLD EXP053` with no path/source text output. A syntactically valid
pin allows execution to proceed to the unchanged source/protocol/manifest
checks and the independently owner-pinned roster validator. It is not by
itself evidence of a real, authorized human roster. Existing signature,
blind-slot, third-adjudicator, minimum-40 and downstream score gates remain.

Synthetic regression tests instrument file hashing, `Path.is_file` and
`Path.read_text` to reject any access under an absent or malformed owner pin,
verify `gate` creates no output, and verify a valid-shape fake pin cannot
bypass the expected source SHA comparison. Tests do not open licensed RACE
passages, actual reviewer forms, private signing keys or scientific model
runtimes. A source-only test does not attest human identity or custody.

## Remaining actions

A real accountable study owner must independently enroll real blinded A/B
reviewers with verified key custody, oversee >=40 independent co-reviews and
third-party adjudication, and independently validate external witness and
rollback controls before considering model inference. A reviewer must inspect
this exact diff. No production or scientific authorization has changed.

Reproduce:
`python3 -m pytest -q tests/test_exp053_early_roster_pin_no_source.py`
