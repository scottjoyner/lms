# EXP-053 — source-free aggregate output path substitution after operator claim

**2026-10-19. Synthetic-only, no scientific execution. NO_GO_RESEARCH_ONLY.**

Before this source change, `replay()` checked that `output.parent.resolve()`
was the intended research folder and output was absent. It then performed
multiple receipt checks, including consuming a single-use external operator
claim, before using `with output.open('x')`. In an ephemeral synthetic
reproduction, we renamed the previously validated research directory and
substituted a symlink to a different invented directory. The old exclusive
pathname writer created source-free aggregate JSON in the substituted
location. This is an output location/custody issue and risks losing the
result after irreversible claim consumption; it is NOT evidence of any
real RACE/scoring disclosure or of bypassed operator controls.

## Source-only correction

The scorer calls `_aggregate_output_parent_identity(output)` BEFORE any
private scorer or operator authorization is consulted. It compares the
output parent to the resolved research source directory, checks for any
existing/symlinked output, safely opens the parent via existing nonfollowing
component-by-component directory traversal, verifies owner identity and no
group/other directory write permissions, and records `(st_dev,st_ino)`.

After the operator claim is consumed, `_persist_source_free_aggregate`
passes that exact approved directory identity to the existing audited
`save_private` writer. The writer requires the freshly opened output parent
to match the pre-claim inode **before creating a file**, and preserves
mode 0600 from birth, exclusive O_EXCL|O_NOFOLLOW creation, fd readback,
fsync, final parent/leaf checks, and unlink-limited-to-our-inode rollback.
The output JSON schema is unchanged. The source-free aggregate is initially
owner-only; any deliberate public release remains a separate authorized
publication operation.

The synthetic tests verify: ordinary exclusive owner-only output,
repeated filename refusal, symlinked parent/leaf, post-validation symlink
swap and same-user directory inode replacement, mismatched expected inode,
unsafe root mode, unavailable Linux flags, and no invented output written
to an external substitute directory. No real operator claim or independent
witness was constructed or consumed.

The fix mitigates detected path substitutions; no software-only same-UID
path validation makes output immutable after the final check. The off-host
witness/high-water authority and human reviewer credentials remain
independent HOLD gates.

Reproduce without licensed data:

```
python3 -m pytest -q tests/test_exp053_aggregate_output_dirfd.py
```

**NO_GO_RESEARCH_ONLY. No merge, deploy or model forward.**
