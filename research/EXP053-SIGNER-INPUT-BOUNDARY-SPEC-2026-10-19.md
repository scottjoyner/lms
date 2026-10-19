# EXP-053 human-operated reviewer signer input boundary

**Frozen before implementation, 2026-10-19 — NO_GO_RESEARCH_ONLY.**

The research-only human signing CLI validates an offline packet and draft
BEFORE opening a private signing key. Synthetic inputs demonstrated that
malformed adjudicator `decisions` (null, object or case IDs that are lists)
raised raw `TypeError`/`AttributeError`, rather than the required `HOLD`.
The packet and draft were read by unbounded path-following `read_text()` and
JSON's default parser silently discards duplicate object keys.

This local-only input boundary must:

- Open the final component of the caller-selected packet and draft with
  `O_NOFOLLOW|O_NONBLOCK`; refuse if either OS protection is unavailable.
  Require a regular file, nonzero size <= 8,000,000 bytes (same limit as the
  offline HTML importer). Read through a single file descriptor with bounded
  reads, and compare device/inode/size/mtime/ctime before and after.
  Always close fd, including on injected `OSError` and `KeyboardInterrupt`.
  Never log text, JSON snippets, absolute paths or private human identities.
- Reject duplicate keys at any nested JSON depth, invalid numeric constants,
  malformed UTF-8, incomplete JSON, non-finite numeric values (including
  exponent overflows), lists where object expected, invalid
  review schema, unknown review fields, previously signed drafts, malformed
  `decisions` and non-string/empty adjudicator case IDs. A missing or invalid
  reviewer ID and missing attestation must remain HOLD, not automatic approval.
- Preserve signing/key-access order: reject bad packet, review, decision,
  packet SHA or adjudicator scope BEFORE `get_key` is called. Preserve exact
  signature canonicalization and private 0600 exclusive signed output writer.
  A valid human-led review can still sign after the person verifies and
  supplies their own independent key. No reviewer keys are enrolled or
  generated here; no real review or scientific admission is authorized.
- Imported browser draft permissions are not claimed to be secure by this
  *read* preflight; the human reviewer must store licensed input and draft
  files in an appropriately private directory. Private signed output remains
  0600-from-creation via the already tested shared writer.
- Tests use only synthetic JSON and fake identities, optionally ephemeral
  synthetic keys, and are local, non-networked, research-only. Include
  malformations, duplicate keys, symlinks, FIFO, oversized files, read
  failures, adversarial filename swaps, and exactly one valid synthetic
  signed review. All offline UI, source, scorer and custody gates stay HOLD.

A passed test is NOT a human attestation, independent review, key-custody
approval, model inference receipt or evidence that 40 source-distinct RACE
articles have been independently adjudicated. Public PR #32 remains draft.
