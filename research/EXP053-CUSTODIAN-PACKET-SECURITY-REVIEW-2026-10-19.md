# EXP-053 source-only custodian packet — automated security review (2026-10-19)

**NO_GO_RESEARCH_ONLY. This is automated adversarial code review, NOT
independent human reviewer approval or custody attestation.** The PR #32-based
review branch remains local. This report contains no privileged fleet logs,
private scientific sources, receipt signatures, signing keys or RACE passages.

## Confirmed defect: packet destination directory race

The initial `exp053_custodian_handoff_packet.create()` implementation validated
its intended user-local output directory using path-based `Path.is_dir()`,
`Path.is_symlink()` and `Path.stat()`, but created the packet afterward with
`os.open(absolute_path, O_CREAT|O_EXCL|O_NOFOLLOW)`. O_NOFOLLOW only protected
the **final path component**; its directory parents were followed again.

A disposable, synthetic local reproduction exchanged the validated output
parent with a symlink immediately before `O_CREAT`. **The original code
reported `CREATED_USER_LOCAL_ONLY` while writing into the wrong directory.**
The test did not touch any production directories, services, fleet nodes, or
research evidence.

## Remediation and reproducible negative controls

The hardened packager now opens the account home and each relative destination
directory component using `O_DIRECTORY|O_NOFOLLOW` descriptors and validates
actual directory owner and non-writable permissions through `fstat`. The ZIP
is created with `O_CREAT|O_EXCL|O_NOFOLLOW` **relative to the retained parent
file descriptor**. The bytes are read back using the same new file descriptor,
then the intended output directory is reopened and its device/inode compared
to the retained descriptor before reporting success. Unexpected changes
produce source-free HOLD, with best-effort same-inode cleanup *relative to the
opened parent* rather than following a substituted pathname.

Regression cases cover swaps immediately before creation and between
validation and creation, symlinked parent components, world-writable targets,
output exclusivity, exact deterministic packet hash, and platform refusal if
no-follow/secure-directory primitives are not available. This mitigates a
symlink-swap attack **against the directory path at publication time**. It
does not claim security against an attacker who already has the SAME effective
user credentials or kernel-level control and can alter the open file itself;
actual independent administrative custody remains external.

The ZIP remains the same deterministic **source-only** seven-document packet
as previously frozen. Expected SHA256:
`a0f2458fca44269fa17e27cbec0b34ea78d3a4cb9a8bb35a05807b02156b75b3`.
The local file has never been transmitted to a custodian by this tool.

## Additional demonstrated final-entry race (follow-up)

A further disposable test exploited a **separate** problem after the directory
race remediation: the process read back its original file descriptor without
checking that the final published directory entry still referred to that
file. Replacing the entry with a symlink to unrelated bytes after the first
file `fsync` led to a **false `CREATED_USER_LOCAL_ONLY` success**. Replacing
it with an ordinary forged file likewise produced success in the old code.

The revised code reopens the **final output entry relative to the securely
retained directory descriptor** with `O_NOFOLLOW|O_NONBLOCK`, checks device and
inode against the originally created descriptor, verifies `0600` permissions,
regular-file type, `nlink == 1`, exact expected size and content, and checks
before/after stable readback metadata. It finally checks that the pathname
still names the same inode and that no metadata drift was observed. Errors
remain sanitized HOLD with best-effort, inode-scoped cleanup; it never deletes
an unrelated substituted file in its cleanup handler.

Adversarial regression now includes (a) replacement with a symlink to decoy
bytes, (b) replacement with an ordinary fake file, (c) deletion of the final
entry, (d) creating an unexpected hardlink, and (e) permission weakening to
world-readable before verification. The original bytes and hash of the
seven-document ZIP remain unchanged.

**Limit:** This is a bounded, point-in-time consistency check, not an OS-level
permanent lock on a pathname. A same-UID or privileged process could change
an entry *after* the final check; a genuinely independent custodian and
protected immutable publication channel are still necessary for trustworthy
long-term custody. These local synthetic negative tests do not count toward
the real 12 physical custody challenges or the 40 blinded RACE reviews.

## Reproduction

```
python3 -m pytest -q tests/test_exp053_custodian_handoff_packet.py
python3 research/exp053_custodian_handoff_packet.py --inspect
```

The second command deliberately returns **exit code 2** even if the packet
passes exact-byte validation. Every acceptance/authorization status remains
`NO_GO_RESEARCH_ONLY`, `scientific_execution_authorized=false`,
`independent_administration_approved=false`, and
`witness_durability_approved=false`.

This finding requires human source review before any proposed publication.
The broader PR #32 stacked branch and its model/adjudication gates remain
unapproved. Zero human article reviews and no native EXP-053 model forwards
were authorized or executed by this change. No external witness setup,
software deployment, NAS/Neo4j write, model loading, GitHub push or merge.
