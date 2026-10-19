# EXP-053 — source-only branch-lineage integration preflight (2026-10-19)

**Frozen before implementation. RESEARCH-ONLY / ALWAYS NO-GO.** This tool
checks local Git object relationships between the explicitly selected EXP-053
checkout, `refs/remotes/origin/main` and
`refs/remotes/origin/research/exp053-race-human-review-gate-20261016` (PR #32).
It cannot prove independently controlled custodianship or scientific safety,
and cannot authorize merging, pushing, deploying, inference, or signer pin edits.

Read-only subprocess calls MUST use an argument array, not `shell=True`.
Allow ONLY: `git rev-parse --verify`, `git merge-base`,
`git diff-tree --no-renames --name-status -r`,
`git status --porcelain=v1 --untracked-files=no`, and `git cat-file -e`
with fixed code-defined revision and path arguments. No `fetch`, `push`,
`checkout`, `cherry-pick`, `merge`, `remote`, `update-ref`, `config`,
network, SSH, file installation, or repository mutation is permitted.
The Git repository path must be an existing, absolute local path; a user-
supplied revision, file-list, executable, Git config snippet or shell command
is never accepted. Report no remote URLs, account data, private paths,
model passages, source bytes, fingerprints or content of diffs.

For each reference, return only FOUND or MISSING. For HEAD vs `origin/main`
and HEAD vs PR #32 return CONNECTED, DISCONNECTED or UNRESOLVED. A missing
remote ref must remain UNRESOLVED, never interpreted as a clean relationship.
If references exist, compare blob trees with Git's read-only `diff-tree` and
summarize changed file counts by A/M/D/T; do not export file bodies, patch
hunks, diff text or private filenames. Check presence of exactly three known
source-only prospective local handoff paths in HEAD and PR #32:
`research/exp053_custodian_handoff_packet.py`,
`research/EXP053-INDEPENDENT-CUSTODIAN-REQUEST-2026-10-19.md`, and
`research/EXP053-CUSTODY-BOUNDARY-NEGATIVE-AUDIT-SPEC-2026-10-18.md`.
The PR branch's remote-tracking ref is cached local evidence, NOT proof that
GitHub has not moved; a separate authenticated GitHub API read is required to
confirm the current head before making a human-reviewed patch.

Output JSON EXACT keys: `schema` (value
`exp053-branch-lineage-negative-preflight-v1`), `references`, `lineage`,
`tree_delta_counts`, `source_only_handoff_file_presence`,
`tracked_worktree_modification_count`, `target_pr_base_ref_status`,
`integration_authorized`, `scientific_execution_authorized`, `decision`,
`next_action`. `references` EXACT keys `head`, `origin_main`, `pr32`, with
values FOUND/MISSING; `lineage` EXACT keys `main`, `pr32`; `tree_delta_counts`
EXACT keys `main`, `pr32` each either null or A/M/D/T integer counts;
`source_only_handoff_file_presence` EXACT keys `local`, `pr32`, each a list
of the above three fixed file names that were found there. The
`target_pr_base_ref_status` is FOUND/MISSING for
`refs/remotes/origin/research/k2-minicpm-v47-exp052-20261016` and MUST
NOT be construed as proof that the actual PR base is still unchanged.

Under ALL circumstances:
- `integration_authorized=false`
- `scientific_execution_authorized=false`
- `decision=NO_GO_RESEARCH_ONLY`
- `next_action=HUMAN_REVIEW_SOURCE_ONLY_DELTA_WITH_CORRECT_ANCESTRY`
- CLI returns exit status **2** even for fully connected branches.

Missing/ambiguous Git refs, unavailable CLI, unexpected status, Git errors,
output parse failures, or wrong repo paths must fail closed with a source-free
NO-GO error object (also exit code 2). Never auto-reparent or force-push an
unrelated root. Do not modify concurrent uncommitted `exp053_preflight_release`
files, change candidate witness trust roots, or run a model. Scientific HOLD
remains independent of Git ancestry and cannot be addressed by this preflight.
