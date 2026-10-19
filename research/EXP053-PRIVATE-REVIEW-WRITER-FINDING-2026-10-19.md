# EXP-053 — private review artifact creation and cleanup finding

**2026-10-19. Source-only security repair. NO_GO_RESEARCH_ONLY.**

## Confirmed failure

Using a disposable synthetic output and the ordinary `022` umask, the old
`exp053_race_review.save_private()` created its target with `open("xb")`,
then `fchmod(0600)` to protect it. The file's actual mode at creation was
**0644**. An injected error at `fchmod` left an **0644 named artifact** behind.
The same `open("xb")` then `fchmod()` pattern was used in the human-operated
`exp053_sign_review.sign()` output. This is a concrete local privacy/custody
risk if the output parent is accessible to other users, not evidence of
an actual disclosure. No private RACE or real reviewer content was inspected
for this reproduction.

## Scoped remediation

`save_private()` now opens the immediate parent directory with
`O_DIRECTORY|O_NOFOLLOW` and creates the final file relative to that fd using
`O_CREAT|O_EXCL|O_NOFOLLOW|O_NONBLOCK`, with requested mode `0600` **at
creation**. It checks regular-file type, owner, single link and no group/other
permissions before writing any bytes. `fchmod(0600)` can then only preserve
or tighten privacy from the secure initial mode. Full canonical JSON+newline
is written, synchronized and read back through the same fd. The code checks
the final directory-entry device/inode and content metadata, then confirms
that the parent path still names the original directory before reporting
success. No overwriting of preexisting files is allowed.

After an ordinary error or KeyboardInterrupt following creation, the writer
attempts inode-guarded cleanup **through the already opened parent fd**.
A third-party substituted file is not deleted by this attempt. Uncertain
inodes or irrecoverable kernel errors can still leave a failed file for manual
review, never a claim of a completed review. The human signing helper now
uses the same exclusive writer for its signed export; signing the synthetic
unit-test fixture **does not** represent an independent reviewer.

## Validation and limits

Synthetic tests exercise default umask, initial permissions, fchmod failure,
short writes, fsync errors, overwrite/symlink refusal, output replacement,
parent-directory swap and sanitized refusal if required kernel flags are
unavailable. A purely ephemeral Ed25519 key is used only for a test proving
the signed-export code calls the same private writer. No human approval or
real RACE passages enter public tests or artifacts.

The opened immediate parent is anchored, but symlinks earlier in the ancestor
path are not independently proven trustworthy; same-user/privileged mutation
after the last check remains out of scope. SIGKILL, power loss, hardware
rollback, external witness custody, human authorship and model behavior are
not covered by these synthetic tests.

Nothing was published, merged or deployed. Real independent code review,
human reviewer enrollment, 40 co-reviewed/adjudicated articles and external
witness/rollback evidence remain entirely separate and blocked. **NO-GO.**

Reproduction:

`python3 -m pytest -q tests/test_exp053_private_writer.py`
