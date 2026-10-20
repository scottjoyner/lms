# EXP-053 — operator claim journal and replay marker parent custody

**2026-10-20. Synthetic-only finding. NO_GO_RESEARCH_ONLY.**

The single-use authorization verifier had a lexical-ancestor trust gap. It
checked that the operator journal's FINAL leaf was not a symlink and required
owner-only mode, but it opened both journal and marker through absolute
paths that could follow symlinks in intermediate directory components. Its
`getattr(os,"O_NOFOLLOW",0)` silently tolerated a missing nofollow flag. A
fully synthetic run (generated ephemeral operator and witness keys; local
mock high-water CAS; no real signing identity) reached an invented owner-only
journal via an **intermediate symlink** and advanced both journal and
`.used-<execution>` marker. This is a custody/path-integrity defect, NOT a
successful bypass of any approved real operator/witness authority: all real
operator and external-witness roots are still EMPTY by design.

## Implemented source-only correction

- The operator journal parent is opened through the existing verified
  non-following per-component directory walker, requiring Linux O_PATH,
  O_DIRECTORY, O_NOFOLLOW and O_NONBLOCK. A missing flag returns HOLD.
- The operator directory must be a real directory controlled by the
  effective uid without group/other write permission. The journal file
  opens by `dir_fd` with O_NOFOLLOW; its inode is compared by `os.stat`
  relative to that same parent fd, while preserving existing flock, owner,
  0600, regular/single-link and bounded stable-read checks.
- The lexical parent path is reopened and rechecked against the originally
  pinned `(st_dev,st_ino)` **before** the external CAS request, **after** CAS
  returns and before creating any local used-marker, and **before** returning
  a success. A substituted symlink, different inode or inaccessible path
  returns HOLD.
- The used marker is exclusive and created with mode 0600 via the same
  pinned `dir_fd`; the parent fsync is likewise done using the pinned fd,
  not a new absolute path. Directory fd is closed on all normal/exception
  paths. The preexisting normal replay prevention and interrupted marker
  semantics remain intact.

Fully invented tests reject symlinked parent before the fake CAS; reject
missing kernel flags; reject a symlink or same-owner different directory
substituted **during** the fake CAS. After a synthetic post-CAS substitution,
the fake witness's high-water advances, but neither the original nor the
decoy directory receives a new marker or changed journal, accurately
preserving fail-closed manual reconciliation requirements. Existing normal,
negative, rollback, and two-concurrent-caller operator tests still pass.

This does not solve same-uid or privileged adversarial changes occurring
**after** the final check, nor does a synthetic signed CAS prove a genuine
separately hosted, human-controlled high-water service. No real operator
key, private reviewer, licensed RACE passage, model weight, or model
inference was used. Both research weight and human roster approval pins
remain empty, and operator and witness key roots are still unconfigured.

`python3 -m pytest -q tests/test_exp053_operator_run.py`

**Scientific decision: NO_GO_RESEARCH_ONLY. No merge or deployment.**
