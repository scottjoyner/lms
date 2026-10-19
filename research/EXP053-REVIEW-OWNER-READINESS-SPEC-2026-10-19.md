# EXP-053 — independent human review owner preflight (source-free)

**Frozen before implementation, October 19, 2026.** This is a read-only
readiness diagnostic for the human study owner, **NOT** a human reviewer,
enrollment, independent witness, or permission to start scientific inference.

- Inputs: exact public cohort manifest, frozen protocol, Python source approval
  pin, and metadata-only `lstat` of four expected private handoff file names
  under the existing user-controlled local private directory. The optional
  Parquet corpus presence is also probed by metadata only. Hugging Face
  snapshots may contain a legitimate symlink into their blob store: report
  `SYMLINK_PRESENT_TARGET_UNVERIFIED` separately, without resolving/following
  the target or declaring it a verified regular corpus. Never open, read,
  hash, render, or copy private packet/exam key, reviewer forms or corpus bytes.
  Never output private absolute paths, RACE text, key materials or identities.
- Public reviewed manifest/protocol/spec bytes must be read through a
  regular-file, size-bounded `O_NOFOLLOW|O_NONBLOCK` open. Require both OS
  safety flags rather than silently using unprotected path-following reads.
  Transfer file-descriptor ownership to `fdopen` only on success; close the
  raw descriptor if wrapping fails. On all refusal paths do not leak an open
  descriptor or emit private source bytes. Check stable device, inode, size,
  mtime and ctime surrounding readback. An unrecognized/symlinked public
  input is a source-free HOLD, never a fallback to unpinned content.
- Validate public SHA pin, schema, exact 64 candidate records, distinct case
  and article hash identities, human_reviewed_articles==0 in frozen manifest,
  min_reviewed==40, evaluation_admitted==false, source/protocol pin binding,
  and public manifest's source-free rows only. This does NOT validate what
  any human read or any private current review progress.
- Require private directory to be a non-symlink directory owned by the local
  euid and mode without any group/other bits, and each named private asset
  to be non-symlink regular file, matching the account's euid and mode with
  no group/other bits. The checker must use a non-following directory
  descriptor and `stat(..., dir_fd=..., follow_symlinks=False)` so directory
  substitution between listing and inspection does not attest unrelated files.
  After the check it must verify that a freshly opened directory descriptor
  still identifies the same private directory inode, otherwise report all
  private files BLOCKED. Missing/unreadable/unsafe means BLOCKED. Do not
  open or inspect inode contents even for the presence check.
- Separate status of private asset availability from owner approval: the
  reviewer roster digest `APPROVED_REVIEWER_ROSTER_SHA256` remains empty, so
  enrollment is UNAPPROVED even if all four private assets exist. A helper
  cannot nominate an independent person or assert key custody.
- Print one deterministic source-free JSON report, including 64 candidate
  cases, minimum 40 distinct co-reviewed cases, minimum 80 independent
  case-level review decisions by two human A/B reviewers before considering
  disputes. Third-party adjudication, real identities, licenses and owner
  approval remain separately required. Every outcome is `NO_GO_RESEARCH_ONLY`
  exit 2, including an intact manifest and safely present private assets.
- Use bounded read-only operations; no network, Git mutation, source packet
  generation, model loading/forwards, approvals, signer enrollment, token
  operations, production deployment, NAS/Neo4j writes or private content export.
- If the public manifest is malformed, swapped, contains extra fields or a
  changed digest, print sanitized `PUBLIC_SOURCE_FREE_MANIFEST_INVALID` and
  fail closed. Do not print Python tracebacks containing source/path data.
  Tests use ephemeral synthetic, public-safe manifests and fake empty private
  files only; they must never access the real licensed source text.

The upstream scientific gate is `scottjoyner/lms#31`. A source-only diagnostic
and a static public manifest containing `human_reviewed_articles=0` are not
an authoritative count of current, signed human review receipts. Given the
owner approval fingerprint is currently empty, the true number of reviews
**admitted by the approval gate** is zero, even if files have been filled
privately; never overwrite or create owner roster in this tool.
