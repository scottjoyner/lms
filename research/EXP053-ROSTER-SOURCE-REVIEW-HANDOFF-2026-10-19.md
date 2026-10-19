# EXP-053 — isolated roster filesystem safety review

**Draft technical code review only. NO_GO_RESEARCH_ONLY.**

This stacked patch is based on the exact head of draft PR #32,
`d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28`, and contains only a
private reviewer-roster reader and **synthetic-only** tests. It is not
an approved human reviewer roster or evidence of any completed study review.

## Focused external review request

Review `exp053_reviewer_roster.load_roster` and
`_open_no_symlink_ancestors`: final-component `O_NOFOLLOW` alone did not
protect symlinked or substituted ancestor directories. The research host
also denies `O_RDONLY` on `/`, so non-reading Linux `O_PATH` directory
handles with per-component `O_DIRECTORY|O_NOFOLLOW` are used. The function
checks the owner, single-link mode-0600 regular-file identity, SHA256 of
private roster bytes, source-free packet digest binding, exact slots and
public-key uniqueness. The code requires relevant OS flags and fails closed.

Review especially the limits of the final path identity check, ambiguous
`close(2)` failures, identity-checked cleanup, same-UID adversaries, and
any risk from directories that change between the two path traversals.
These are source-code controls, not independent reviewer enrollment or
third-party custodian evidence.

## Reproduction (Linux with O_PATH)

```
python3 -m pytest -q tests/test_exp053_reviewer_roster.py \
  tests/test_exp053_roster_nofollow_failclosed.py \
  tests/test_exp053_roster_ancestor_race.py
python3 -m ruff check research/exp053_reviewer_roster.py \
  tests/test_exp053_roster_nofollow_failclosed.py \
  tests/test_exp053_roster_ancestor_race.py
```

All tests use disposable synthetic JSON, invented reviewer pseudonyms,
and artificial public key bytes. No RACE passage, exam answer key, private
human review form or real private signing key is contained in this patch.

## Scope and safety

Do not merge, deploy, create enrollment pins, assert human independence,
load models or import private source data. The scientific experiment still
requires two owner-vetted, individually key-controlled blinded reviewers,
>=40 source-distinct independently co-reviewed/adjudicated articles,
independent custody/rollback evidence and separate model-weight approval.
Draft PR #32 and the original research worktrees remain unchanged.

**Decision: NO_GO_RESEARCH_ONLY; zero independently admitted human reviews.**

## Critical follow-up: Ed25519 public-key subgroup (October 19)

Before approval, independently review the **weak public-key signature issue**
documented in `EXP053-ED25519-KEY-SUBGROUP-FINDING-2026-10-19.md` and the
preregistered source-only test specification in
`EXP053-ED25519-KEY-SUBGROUP-SPEC-2026-10-19.md`.

`load_roster()` now refuses noncanonical, low-order and mixed-torsion Ed25519
public keys, and direct `exp053_race_review.verify_signed()` callers enforce
that same prime-order subgroup check. The installed OpenSSL provider accepted
an identity-point synthetic signature for two unrelated messages; this could
otherwise defeat private-key-possession evidence. An independent check against
libsodium matched all 228 synthetic key classifications. It remains essential
to review the point-decoding and group-arithmetic implementation carefully.

Focused reproduction:

```
python3 -m pytest -q tests/test_exp053_roster_weak_keys.py \
  tests/test_exp053_reviewer_roster.py \
  tests/test_exp053_roster_ancestor_race.py \
  tests/test_exp053_roster_nofollow_failclosed.py \
  tests/test_exp053_human_review_gate.py
python3 -m ruff check research/exp053_reviewer_roster.py \
  research/exp053_race_review.py tests/test_exp053_roster_weak_keys.py
```

This correction does **not** authorize reviewer enrollment or promote draft
PR #37; approval remains NO_GO_RESEARCH_ONLY.
