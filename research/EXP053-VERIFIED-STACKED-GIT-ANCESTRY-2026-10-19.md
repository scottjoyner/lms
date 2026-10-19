# EXP-053 — verified upstream stack topology and local shallow clone correction

**Read-only research finding, 2026-10-19. NO_GO_RESEARCH_ONLY.**

The shared multi-worktree `scottjoyner/lms` object repository at
`/home/scott/git/lms/.git` was checked and reports `git rev-parse
--is-shallow-repository = true`. Its shallow boundary contains the PR #32
head commit `d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28`. That commit's raw
object names a parent even though `git rev-list` treats it as a root.
Consequently, an absent local merge-base against `main` or PR #32 in the
shared checkout does **not** prove disconnected history. The earlier v1
source-only ancestry preflight inaccurately returned DISCONNECTED. Its
interpretation is superseded; v2 returns UNVERIFIED_SHALLOW_HISTORY when
history truncation prevents finding a common ancestor.

An **independent, separate Git repository** was created solely for fetching
public upstream commit objects. No shared `.git/shallow`, source checkout,
branch refs, GitHub PR, production system or fleet host was altered. Fetch
only, no checkout/merge/cherry-pick/rebase/push. Exact authentic GitHub PR #32
and PR #30 metadata independently identified these branch heads:

| Reference | Verified commit |
|---|---|
| PR #32 head | `d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28` |
| PR #32 declared EXP-052 base / PR #30 head | `a04c9bb9262be8bf39012ba7b9c85ef0f505e486` |
| origin/main | `1f64776b3b31c581f88635a5ed3bc2a98a48f0c4` |
| PR #32 / origin/main merge base | `d0cf7a242f9d57bf068892eb825127d6c5e3cc3c` |

In the isolated complete Git commit graph:
- EXP-052 base -> PR #32: 0 commits behind / **95 ahead**, with **107** changed
  source-tree files (matching PR #32 remote metadata).
- `main` -> PR #32: **2 behind / 161 ahead**. Actual common base exists.
- A clean, source-only review candidate `c944409ccc84374523b376d66f18a1f9e0bf1412`
  has exact parent `d14b1dcc...` and adds ONLY eight allowlisted custodian
  handoff files, already tested in its dedicated worktree. It is not pushed.

## What this establishes and what it does not

The source-only review candidate **does not need history surgery** to descend
from PR #32; its Git parent is exact. This does not make PR #32, which stacks
95 commits on an unmerged EXP-052 dependency, ready for main. Publication
needs separate live GitHub head/base revalidation and human code review.
External custodian independence, native high-water CAS, separately retained
and authenticated observer roots and 12 actual native negative tests are
**not** established by any ancestry or local source-only packet. There are
still zero independently reviewed RACE cohort articles admitted and no new
EXP-053 model forwards. Scientific inference remains NO-GO.

Avoid asserting "disconnected" from an incomplete `.git/shallow` checkout.
Do not unshallow the shared fleet clone merely to change a diagnostic, since
concurrent repositories/worktrees may be using it. Do not force-push,
merge, promote a PR, deploy a witness or run scientific evaluation based on
these source-only findings.
