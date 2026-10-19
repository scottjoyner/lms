# EXP-053 — owner roster symlinked-ancestor substitution

**2026-10-19. Synthetic-only security reproduction. NO_GO_RESEARCH_ONLY.**

## Confirmed

The old `load_roster` used `path.resolve(strict=True)` to exclude public repo
files, then `os.open(path, O_NOFOLLOW)` for the final roster. An attacker can
rename an earlier path directory and replace it with a symlink into the
public repo after the resolve but before the open. A *different, identical-
byte synthetic roster* inside the repo passed the exact SHA-256 pin and was
accepted. No real reviewer data was read, and there is no demonstrated
cryptographic signature bypass. This is a repository-exclusion/custody-origin
race even when the roster bytes happen to match an approved fingerprint.

## Fix

`exp053_reviewer_roster._open_no_symlink_ancestors` now securely walks the
absolute lexical pathname starting at `/` with `O_PATH|O_DIRECTORY|O_NOFOLLOW`
per ancestor; the leaf is opened relative to that parent with
`O_RDONLY|O_NONBLOCK|O_NOFOLLOW`. An unavailable OS flag is a HOLD. `O_PATH`
is necessary because the fleet research host does not permit a normal
read-only open of `/`, but does allow non-reading directory traversal.
The original exact byte pin, owner uid, mode 0600, link count, size and
before/after fstat checks remain in place; roster file contents are read via
one descriptor with bounded `os.read`. On completion the code re-resolves
and securely reopens the named file, comparing the original and final device
and inode, rejecting moved/replaced/symlinked ancestors or changed leaves.
The repo-exclusion rule remains mandatory.

Synthetic negative tests reproduce path swaps **before descriptor traversal**,
while reading and while replacing an identical-byte leaf, including a distant
ancestor and static ancestor symlink. Tests also cover missing `O_PATH` and
the positive single-link private roster case. They do NOT establish that a
human actually owns a roster, prove public-repo history absence, prevent an
attacker with same-user privileges from racing after the final check, or
create independent high-water/root custody evidence.

## Still blocked

Real owner-approved reviewer roster pin empty; no independent human review,
40 accepted/adjudicated source-disjoint articles, witness independence, model
forward or deployment. This local candidate requires independent accountable
technical code review. The worktree and review package are not approvals.

Run: `python3 -m pytest -q tests/test_exp053_roster_ancestor_race.py`.
