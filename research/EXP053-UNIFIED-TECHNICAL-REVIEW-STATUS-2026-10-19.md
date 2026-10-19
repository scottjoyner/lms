# EXP-053 unified technical review / owner-readiness handoff

**Research-only. NO_GO_RESEARCH_ONLY. October 19, 2026.** This combined
change deliberately unifies two *source-only* paths under the exact frozen
remote GitHub PR #32 head:

`d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28`

1. The deterministic seven-document source-free custodian packet and its
   directory/path/partial-file/interrupt negative tests and review checklist.
2. The independent study-owner readiness CLI reporting publicly pinned cohort
   counts and **metadata only** for existing private local reviewer forms.

The first produces a **non-authorizing** ZIP; the second never opens RACE
licensed text, private reviewer forms, the answer key or any signing keys.
The fact that the original registered PR is draft, the owner approval
fingerprint is empty, and no independent human review has been admitted is
**not corrected** by unifying the code changes.

## Required independent technical review

- Compare the exactly named local candidate SHA and recorded SHA256 of the
  review patch against the precise PR #32 head, not main or a shallow clone's
  artificial history boundary. Review only the explicitly listed source-only
  changed paths and provenance/permissions prior to any public publication.
- Reproduce both synthetic suites in an isolated checkout:

  `python3 -m pytest -q tests/test_exp053_custodian_handoff_packet.py tests/test_exp053_review_owner_readiness.py`

- Inspect the source-only ZIP packager's required `NO_GO_RESEARCH_ONLY` and
  exit-code-2 behavior and all final-entry/no-follow/failure cleanup paths.
- Review the owner readiness metadata-only implementation, including refusal
  of unsafe parent-directory swaps and no-follow directory traversal, denial
  on missing source pins, and lack of any path that could grant human review
  approval or modify the private roster.
- Independently examine exact licensing, privacy and key-custody claims,
  then provide a reviewer-authored, attributable response out of band.
  Programmatically created reviews do not count as human independence.

## Still blocked

- Scientific review: no owner-approved A/B key-custody fingerprint, no 40
  independently adjudicated article-disjoint RACE cases; 64 source-free
  cohort records are candidates, not completed human reviews.
- Witness: no independently controlled signing/admin/restore root, no
  separately persisted high-water evidence, no native negative-control
  witness approval. Fleet SSH connectivity alone is not independent custody.
- Inference and release: NO model-forward passes, merges, GitHub publication,
  deployed service updates, key pin modifications or scientific approval.

**The source-only reviewer packet is NOT an approval object.** Only an
independent accountable human study owner/custodian can establish external
trust, consent and custody. Until then keep both the scientific workflow and
production deployment blocked.

## Source-free readiness public-file descriptor failure (automated follow-up)

One reproducible **resource leak** existed in the owner-readiness checker:
`_public_bytes` opened its frozen public specification/manifest, then called
`os.fdopen` without closing the original descriptor if Python rejected the
wrapper. An injected `OSError` left a descriptor open despite a sanitized
HOLD response; repeated failures could consume the worker's file descriptor
limit. The revision gives the new descriptor to the file wrapper only after
successful ownership transfer, and otherwise closes it even on an injected
`KeyboardInterrupt`. It also requires `O_NOFOLLOW|O_NONBLOCK` rather than
silently degrading to unprotected public-file open flags on unsupported
platforms. The public-file readback metadata check now includes size and
ctime alongside device/inode/mtime.

Negative tests inject descriptor-wrap failures, KeyboardInterrupt, missing
kernel flags and symlinked public protocol paths; all fail closed. They do
not read licensed RACE passages, reviewer decisions or private keys, and
cannot grant human authority. The frozen source-only manifest, review count,
and researcher approval pin remain unchanged.

## Scientific label-validation parity repair (October 19)

Review UI and Python signed-review intake both accepted empty/multi-character
A/B/C/D **substrings** as plausible labels. Downstream metrics, replay and
candidate-cohort selection contained the same error. The source-only fix
requires one exact A/B/C/D label at all five boundaries and adds synthetic
Node/Python tests, including a pre-signing denial test. No human reviewer was
enrolled, no exam texts were disclosed, and no score/model admission was run.
This is a review candidate requiring independent technical scrutiny.

## Offline human signer key-read correction (2026-10-19)

A synthetic Ed25519 private key was previously accepted through a final-
component symlink because the human-operated signing helper used separate
path-following `stat()` and `read_bytes()` calls. The corrected helper uses
one private non-following, nonblocking fd with owner/mode/type/nlink and stable
metadata checks before parsing a bounded PEM. Negative tests use only
synthetic keys, and all personal/owner permissions remain unapproved. See
`EXP053-HUMAN-REVIEW-SIGNER-KEY-READ-FINDING-2026-10-19.md` and the frozen
`EXP053-HUMAN-SIGNER-KEY-READ-PREFLIGHT-SPEC-2026-10-19.md`.

## Invalid reviewer case/trust object refusal (October 19)

Offline signed-review parsing previously surfaced uncaught `TypeError` on
JSON object/list case IDs and malformed reviewer trust-store values. Source-
only validation now checks case IDs and public-key string types before
set/dictionary membership and preserves strict third-party disputed subset
validation. Synthetic tests ensure deterministic HOLD and prohibit signing-
key access for invalid drafts. See the frozen
`EXP053-MALFORMED-REVIEW-REFUSAL-SPEC-2026-10-19.md` and findings report.

## Private artifact creation window closed (October 19)

An injected `fchmod` failure demonstrated that the old `open("xb")` then
`fchmod(0600)` private writer could leave a mode-0644 file. It has been
replaced with parent-dirfd-anchored `O_EXCL` creation in mode 0600 before any
bytes are written, full readback and inode-scoped cleanup; the human signer
export reuses this private writer. Unit tests use synthetic JSON and ephemeral
keys only. See `EXP053-PRIVATE-REVIEW-WRITER-SPEC-2026-10-19.md` and
`EXP053-PRIVATE-REVIEW-WRITER-FINDING-2026-10-19.md`. This does not amend
human or scientific authorization.

## Offline reviewer packet/draft read hardening (October 19)

The human-operated signer previously used path-following unbounded JSON
reads, permitted duplicate JSON fields, and raised raw exceptions for
malformed adjudicator subsets. The local input path now performs a stable,
bounded one-descriptor read and strict JSON/decision validation before
opening a signing key. Synthetic adversarial tests cover the refusal
boundary and a valid synthetic signature; owner enrollment and model
admission remain unapproved. See `EXP053-SIGNER-INPUT-BOUNDARY-SPEC-2026-10-19.md`
and `EXP053-SIGNER-INPUT-BOUNDARY-FINDING-2026-10-19.md`.

## Unapproved reviewer roster refuses before source hashing (October 19)

The offline scientific review gate previously hashed licensed RACE source and
read private packet/sealed answer-key inputs before detecting the empty human
owner enrollment pin. The corrected entrypoint refuses blank/malformed owner
pins first; source/protocol/manifest and owner-pinned roster checks remain
mandatory if an exact hex pin is configured. Negative tests instrument
public-safe synthetic paths to prove no sensitive-file access without
approval. See `EXP053-EARLY-OWNER-PIN-FAIL-CLOSED-SPEC-2026-10-19.md` and
`EXP053-EARLY-OWNER-PIN-FAIL-CLOSED-FINDING-2026-10-19.md`.

## Authenticate exact owner roster before licensed source reads (October 19)

A syntactically plausible but incorrect owner roster SHA previously allowed
`verified_admission` to hash licensed source before roster byte validation.
The local research branch now uses only the preregistered source-free manifest
packet digest to verify the exact private owner roster first. Malformed,
mismatched or unsafe synthetic rosters refuse before any source access; a
valid synthetic roster still cannot bypass the source hash. See
`EXP053-ROSTER-BEFORE-SOURCE-CUSTODY-SPEC-2026-10-19.md` and
`EXP053-ROSTER-BEFORE-SOURCE-CUSTODY-FINDING-2026-10-19.md`.

## Reviewer roster reader refuses missing non-following kernel flags (October 19)

The private human owner-roster loader previously permitted opening an owner-
pinned roster without `O_NOFOLLOW` when that OS feature was unavailable.
The source-only fix now requires `O_NOFOLLOW|O_NONBLOCK` and checks matching
owner uid and single-link identity before reading. Synthetic tests cover
missing flags, symlinks, hardlinks, owner mismatch, invalid pins and public-
repository placement. The actual owner approval pin is still EMPTY; this
is NOT independent human enrollment. See
`EXP053-ROSTER-NOFOLLOW-FAILCLOSED-SPEC-2026-10-19.md` and
`EXP053-ROSTER-NOFOLLOW-FAILCLOSED-FINDING-2026-10-19.md`.
