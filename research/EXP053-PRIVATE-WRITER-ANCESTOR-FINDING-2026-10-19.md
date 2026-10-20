# EXP-053 — source-only private writer ancestor path control

**2026-10-19. Synthetic-only demonstration. NO_GO_RESEARCH_ONLY.**

The existing draft PR #39 still contained a legacy `save_private()` writer:
`open(path, "xb")`, followed by `os.fchmod(fd,0o600)`, and then the JSON
write. The more robust exclusive descriptor-bound writer that immediately
opens mode-0600 and verifies its readback and inode had been implemented
on the unpublished unified research branch, but not yet extracted into the
stacked public source-review PRs.

In addition, the unified writer's open of `path.parent` with O_NOFOLLOW
protected only the *immediate* parent from a symlink. It did not protect
intermediate directories. In disposable synthetic tests, a symlinked
ancestor redirected a supposedly private JSON write into another directory.
The resulting file was still mode-0600; this is a pathname/custody failure,
not a demonstration of world-readable data or exposure of any real key.

This focused patch exposes the previously source-tested secure writer and
adds the missing ancestor protection. The writer now uses the existing
kernel-enforced `_open_no_symlink_ancestors()` traversal on the first and
final parent verification, checking parent directory type and effective-user
ownership before leaf creation. File creation remains `O_EXCL` and
`O_NOFOLLOW`, mode-0600 from birth; partial writes, interruption rollback,
fsync, descriptor readback, readback inode and parent checks all remain.
Interrupted writes unlink only the inode created by this attempt, never a
substituted unrelated file.

New synthetic tests cover static intermediate and direct parent symlinks,
owner UID mismatch, missing O_PATH, an ancestor symlink replacement during
write, intact existing decoys, and a successful nested owner-private path.
The prior private writer and ephemeral signer integration tests are included
in the focused source-only review so both behaviors are independently
reproducible. No RACE article, actual review, real private signing key, or
scientific model execution was accessed or created.

The POSIX path rechecks do not give absolute protection against hostile
same-UID code changing the path *after* final verification. Independent
checkpoint authority, human signer identities, true blinded annotation and
model-weight approval remain outside this patch.

Local source-only reproduction:

```
python3 -m pytest -q tests/test_exp053_private_writer.py \
  tests/test_exp053_private_writer_ancestors.py
```

All owner and model approval pins remain unset. **NO_GO_RESEARCH_ONLY.**
