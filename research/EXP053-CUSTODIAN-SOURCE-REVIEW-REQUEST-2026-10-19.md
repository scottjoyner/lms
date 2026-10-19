# EXP-053 — independent source-code review request (October 19, 2026)

**Decision: NO_GO_RESEARCH_ONLY.** This document requests a review; it is not
independent approval, operator authorization, model admission, proof of native
rollback protection, or evidence that any human has reviewed source text.

## What is being reviewed

Repository: `scottjoyner/lms`.

Comparison base: **the exact PR #32 head**
`d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28`, *not* `main`, the EXP-052
base, or the unrelated shallow worktree's presumed root. GitHub PR #32 remains
DRAFT and stacked; its registered head/base must be rechecked immediately
before any reviewer uses this patch.

Scope: a new source-only custodian handoff request, frozen protocols, a source-
free Lenovo observation explicitly marked untrusted, the deterministic local
ZIP packager `research/exp053_custodian_handoff_packet.py`, its synthetic
negative tests, and the security findings report. A separate file-level
manifest, patch checksum and source hashes belong in the external offline
review package. The developer alone cannot attest reviewer identity or
independence merely by writing a checklist or generating signatures.

This technical source review **does not** automatically constitute any of the
>=40 independently blinded human RACE item adjudications required by issue
#31. No original source passages, answer keys, model logits, true native
privilege logs, reviewer identities/consent, passwords, keys or approvals
should be present in the public review package.

## Additional exact-label scientific intake review (October 19)

The unified review patch also repairs a confirmed Python/JavaScript
substring-validation defect: a blank string or `"AB"` is considered a
substring of `"ABCD"`. Before this repair, the offline UI could falsely mark
a malformed item complete, and Python review/analysis could accept a
multi-character foil. Independently inspect the strict single-character
A/B/C/D checks in the offline UI, `exp053_race_review.py` (including candidate
cohort intake), `exp053_verified_analysis.py` and `exp053_metrics.py`. Review
the synthetic Node/Python parity and signing-denial tests and ensure no
change was made to approved reviewer roster pins, sealed corpus, scientific
preregistration, source-passage content or model execution. See
`EXP053-OFFLINE-REVIEW-VALIDATION-FINDING-2026-10-19.md` for the confirmed
failure and exact conditions. This source review is not human adjudication.

## Human-operated signer key custody review (October 19)

A separate synthetic key-file test confirmed that the offline human signing
helper previously followed a symlink into a private Ed25519 PEM and performed
its permission and data reads through separate paths. The candidate now adds
an owner-only, single-descriptor non-following key reader to
`exp053_sign_review.py` and synthetic tests in
`tests/test_exp053_signing_key_read.py`. The source-only documentation is
`EXP053-HUMAN-REVIEW-SIGNER-KEY-READ-FINDING-2026-10-19.md`.

An independent technical reviewer should check POSIX flag refusal,
regular-file ownership/private permissions/nlink, bounded file size,
stable metadata, safe descriptor closure and no private-key logging. A
successful synthetic Ed25519 public-key extraction **does not** prove that
any reviewer is human, properly blinded or owner enrolled. Keys used in
unit tests are transient synthetic fixtures, not eligibility evidence.

## Reviewer's required decisions, in priority order

1. **Trust and authority claims**: verify that `--inspect`, `--verify`, and
   `--create` always return `NO_GO_RESEARCH_ONLY`/exit 2; no output can be
   misinterpreted as permission to deploy, merge, train, evaluate or release.
   Packet hashes prove bytes only, not administrative separation or freshness.
2. **Read-only versus mutating scope**: inspect all files and source imports;
   confirm the only optional write is a single explicitly requested, exclusive
   user-home-local ZIP. No network calls, SSH, Git updates, credential
   operations, scientific inputs, service installation or signer enrollment.
3. **Provenance and licensing**: independently inspect each allowlisted
   document and the JSON claims before public sharing. Confirm the claims
   record contains no real identity, private hostname/IP, login credential,
   source passage or hidden grant. Explicitly record any license or privacy
   question as a blocker; absence of a scanner finding is not proof.
4. **Filesystem paths and races**: review dirfd-relative no-follow traversal,
   parent ownership/permissions, `O_EXCL`, no-overwrite, the final inode and
   exact-byte readback, and inode-aware cleanup on failure/interruption.
   Consider symlink exchanges, hardlinks, substitute entries, permission
   changes, partial writes, failure to `fdopen`/`fsync`, and forced kill.
5. **ZIP and manifest verification**: review duplicate/unknown ZIP member
   refusal, canonical ZIP byte matching, bounded decompression and JSON,
   source pins, deterministic rendering and tamper failure. Ensure an
   adversary cannot swap manifest+source pins and present a forged history
   as an independently authenticated checkpoint.
6. **Negative-test quality**: reproduce focused and broad tests from a
   disposable, private checkout; inspect potential mock weaknesses and
   missing real kernel/user namespace controls. Synthetic path mutation is
   NOT a physical privilege denial against an independent witness.
7. **Human custody prerequisites**: keep any decision to *conduct* disposable
   nonproduction negative tests separate, with distinct admin/root/signer/
   hypervisor/backup/observer control and externally held root/head pins.
   Require actual native denial and rollback evidence before considering
   custodian authority. This source package does not create that authority.

## Reviewer response to collect out of band

An accountable real human reviewer should produce a private signed or
otherwise independently attributable response containing exact patch
SHA256, base SHA, candidate SHA, reviewed file list, review date, reviewed
controls and test environment, blocking findings and severities, and one of:
`CHANGES_REQUESTED`, `SOURCE_REVIEW_COMPLETE_WITH_LIMITATIONS`, or
`DECLINE_UNVERIFIED`. No software token may automatically convert such a
response into independent observer custody, reviewer enrollment, 40-item
adjudication or scientific inference permission.

**Automatic release decision is always NO-GO.** If identities, approvals or
custody evidence are unavailable, report the request as awaiting human
review. Do not manufacture a review identity or a signature to satisfy this
handoff. The local source-only change may be prepared for independent review,
but do not force-update PR #32, merge the stacked research history or run
EXP-053 model forwards on this basis.

## Malformed-review refusal follow-up (October 19)

The current unified source review also includes a narrow denial repair for
unhashable JSON case IDs and malformed reviewer trust mappings. Review exact
source and negative tests in `exp053_race_review.py` and
`tests/test_exp053_malformed_review_refusal.py` before approval. This is an
error-handling fix, **not** evidence of an independently enrolled signer or
signed RACE decisions. See `EXP053-MALFORMED-REVIEW-REFUSAL-FINDING-2026-10-19.md`.

## Private output mode and interruption review (October 19)

The candidate fixes the initial mode-0644 window in private JSON and signed
review creation. An independent reviewer should inspect `save_private()` in
`exp053_race_review.py`, its use by `exp053_sign_review.py`, and tests in
`tests/test_exp053_private_writer.py` for exact-mode `O_EXCL` creation,
no-follow parent binding, readback, cleanup and no-overwrite. This is a
local source-only safety improvement; it does not constitute signed human
review evidence or proof of independent custody.

## Human signer offline draft input review (October 19)

Review `exp053_sign_review.py` and `tests/test_exp053_signer_input_boundary.py`
for bounded no-follow packet/draft reads, duplicate JSON keys, invalid
adjudicator case IDs, and denial **before** private signer key access. This
is a source-only integrity hardening patch, not evidence that a real reviewer
has been enrolled, agreed or signed. See the frozen specification and
reproducible finding named `EXP053-SIGNER-INPUT-BOUNDARY-*.md`.

## Reviewer admission early-denial source access (October 19)

Inspect the entry of `verified_admission` and tests in
`tests/test_exp053_early_roster_pin_no_source.py`. A blank or invalid
approved reviewer-roster pin must HOLD before any source hash, public
manifest, private packet, review form or answer-key read. A valid-format
fake pin does not bypass downstream checks. Synthetic tests and the status
report do NOT attest independently enrolled humans, 40 approved articles,
or external witness custody.

## Exact roster first, licensed source second (October 19)

Review the new `verified_admission` order and synthetic tests in
`tests/test_exp053_roster_before_source.py`. The code now checks the pinned
source-free public manifest, then the exact privately owner-pinned roster
before probing/hashing the licensed RACE corpus, private packet or sealed
answers. Confirm that wrong owner/packet pins and unsafe rosters HOLD and that
real source SHA, human signatures and 40-item thresholds remain mandatory.
Passing unit tests cannot independently enroll humans or approve model runs.
