# EXP-053 — source-only admission denial and private JSON reader

**Draft technical security review request, October 19, 2026.**
**Decision: NO_GO_RESEARCH_ONLY.**

This is a **stacked draft** against source-only draft PR #37, which contains
the reviewed-on-host (not yet human-reviewed) owner roster and Ed25519
signature checks. No signed human review, RACE source, credential, or private
experimental data is included in this diff.

## Small, reviewable source-only change

Three mutually dependent denial boundaries are included:

1. Before touching licensed source bytes or private source/answer-key files,
   `verified_admission` requires a syntactically valid owner-approved roster
   pin and verifies the independently pinned, public **source-free** manifest
   and the exact private reviewer roster via its packet identity. Only then
   does the existing source hash/protocol/packet exam check run.
2. Replace raw path-following, unbounded reads of the blinded packet,
   sealed exam key, signed A and B reviews, and optional third adjudicator
   with a single `_read_private_admission_json()` guarded reader.
3. That reader refuses symlinked ancestors, final-component symlinks,
   hardlinks, weak owner/mode, oversized/nonregular content, unstable files,
   substituted path/inode, duplicate JSON fields, invalid UTF-8, nonfinite
   numbers and non-object roots. It reads at most 8 million bytes and
   never treats ambiguous content as evidence.

Source-free reproduction:

```
python3 -m pytest -q tests/test_exp053_private_admission_json.py \
  tests/test_exp053_early_roster_pin_no_source.py \
  tests/test_exp053_roster_before_source.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_race_review.py \
  tests/test_exp053_private_admission_json.py
```

Every new test uses only synthetic public-safe JSON, fake reviewer labels,
and a disposable private temporary directory. No actual RACE article,
private key, human response or model inference is needed.

### Independent reviewer requests

Verify the owner-pin/roster/source-check ordering, fixed public-manifest
binding, all private-path entry points, descriptor/ancestor identity checks,
JSON numeric and duplicate-field semantics, cleanup on read failures, and
preservation of existing 40-item human signoff and Ed25519 restrictions.
Review Linux O_PATH/O_NOFOLLOW assumptions, same-UID race caveats, and the
explicit 8 MB size limit. Sign-off on code alone does NOT attest reviewer
independence or external custody, and does not authorize scientific scoring.

### Hard HOLD

Owner-approved human roster SHA and model weights approval SHA remain empty.
Zero independently admitted human reviews, no third-party adjudication,
no 40 approved source-distinct articles, and no independently controlled
witness/replay authority. PR #37 and this stacked PR must remain draft.
Do not merge or deploy. **NO_GO_RESEARCH_ONLY**.

## Public manifest hash/read atomicity correction (October 19)

A synthetic swap between the two historical `filehash(MANIFEST)` and
`MANIFEST.read_text()` operations showed that the roster could receive a
substitute `packet_sha256` despite a successful check of *other* manifest
bytes. The draft now calls `_read_pinned_public_manifest()`, which checks
exact pinned SHA-256 and strict JSON against the same bounded, stable fd
buffer before private roster lookup. Added no-follow ancestry, final inode
recheck and synthetic mismatch/duplicate/mutation tests. Review
`EXP053-PINNED-PUBLIC-MANIFEST-ATOMIC-READ-SPEC-2026-10-19.md`, finding,
and `tests/test_exp053_public_manifest_atomic.py`. No science approval.
