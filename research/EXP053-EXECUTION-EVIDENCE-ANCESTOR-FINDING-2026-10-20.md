# EXP-053 — scorer evidence reader must reject symlinked ancestors

**2026-10-20. Synthetic-only security finding. NO_GO_RESEARCH_ONLY.**

The previous execution evidence reader `private_regular_file` resolved the
private root and evidence file and checked their resolved relationship, then
opened the pathname with `O_NOFOLLOW` on the final leaf only. An invented
owner-private evidence file remained accessible through a symlinked
intermediate directory under that private root. The same path could be
silently followed for signed execution attestation, score archives, attempt
logs and operator authorization, even though a normal `st_mode 0600`
regular-file check succeeded. The reader also used the unsafe fallback
`getattr(os,"O_NOFOLLOW",0)` which could drop symlink protection on a host
lacking that flag. This is a filesystem path provenance issue, **not** proof
of a live forged model execution or reviewer admission.

## Source-only repair

The shared `private_regular_file` now requires Linux O_PATH, O_DIRECTORY,
O_NOFOLLOW and O_NONBLOCK; canonicalizes the caller's lexical absolute
pathname before validating it lies under the selected private root; and uses
the earlier audited `_open_no_symlink_ancestors` to open each directory
component and the final evidence leaf without symlink following. It demands
one regular, single-link, effective-user-owned and mode-private nonempty
file. It hashes exact bytes from one fd, retaining bytes only for an
explicit bounded JSON request, and checks unchanged fd identity and exact
byte count. Before accepting the result, it securely reopens the original
lexical path and checks its inode still matches the one hashed.

Fifteen synthetic tests cover stable owner-private evidence, stream hashing,
intermediate/final/private-root symlinks, traversal outside the selected
root, hardlinks, FIFO, weak permissions, changed inode with identical bytes,
midread symlink substitution, midread rewrite, missing Linux flags, wrong
UID, and invalid/oversized JSON limits. The existing signed execution-
attestation tests are rerun unchanged. Tests include no real RACE passages,
model weights, human reviewer signatures or real operator grants.

The helper still cannot guarantee custody in a hostile same-UID or privileged
environment after the last directory/path identity check. It cannot replace
an independently administered external witness or the model-weight and
human-review approval gates. All real approval pins remain intentionally
empty. **NO_GO_RESEARCH_ONLY, no merge/deployment/inference.**
