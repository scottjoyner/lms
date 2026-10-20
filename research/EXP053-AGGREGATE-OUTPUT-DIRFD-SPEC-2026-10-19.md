# EXP-053 — pinned public aggregate output location and single-use claim

**Frozen preregistered source-only security contract, 2026-10-19.**
**NO_GO_RESEARCH_ONLY.** No real scorer, operator receipt or private source.

## Reproduced issue

The scoring replay validated that `output.parent.resolve()` was the trusted
research root, then performed many checks and consumed an irreversible
single-use independent operator authorization. Finally it called
`with output.open('x') as ...`. A disposable synthetic directory renamed
AFTER the initial validation and replaced by a symlink to another directory
caused the old output call to create an invented JSON aggregate in the
substitute directory. No licensed data or real operator authorization used.
The aggregate is source-free by intent, so this is a location/custody and
one-use-result-availability risk, NOT evidence of actual reviewer leakage.

## Required fix

- Preserve existing public output naming/location policy: one fresh file in
  research root, exclusive no-overwrite creation, no actual results in tests.
- Before irreversible operator claim, bind the accepted repository research
  directory to a stable file descriptor that refuses every symlinked
  ancestor component and checks effective-user ownership and mode.
- After claim, create source-free aggregate using the existing 0600-from-
  birth, `O_EXCL|O_NOFOLLOW|O_NONBLOCK`, fd-readback/fsync/inode-bounded
  rollback `save_private` path, **but only if its freshly opened parent inode
  matches the root inode authorized before the claim**. Reject any parent
  redirection without creating a file. Recheck parent pathname before
  claiming and after successfully writing. Do not risk removing unrelated
  replacement files on failure. Exact summarized JSON only, no private
  source, per-item score rows or human text in public output.
- Keep single-use claim consumed on failure after witness CAS (fail closed).
  Never replay authorization or reset a high-water marker automatically.
- Unit tests: synthetic valid owner-only directory succeeds with mode 0600
  and parseable aggregate; wrong directory, symlinked ancestor/leaf, existing
  file, parent path replaced by symlink AFTER first check, and same-UID
  parent inode changed AFTER claim refuse without writing outside or
  overwriting a decoy. Any mock stand-in for the operator is explicitly fake
  and must not create real signed claims.
- Preserve all source-signature, independent human, operator and weight
  approval pins as empty. No merge, deployment, actual RACE/model run.

Filesystem identity checks do not provide tamper-resistant custody against
post-check same-UID/privileged changes; separately controlled witness and
signer authority remain independent hard release gates. NO_GO_RESEARCH_ONLY.
