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
