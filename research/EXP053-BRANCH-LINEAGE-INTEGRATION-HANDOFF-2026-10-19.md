# EXP-053 integration lineage — read-only NO-GO handoff (2026-10-19)

> **CORRECTION (2026-10-19):** The earlier DISCONNECTED conclusions below
> were **false negatives from shallow Git history**, NOT evidence of unrelated
> branch roots. The shared repository has a `.git/shallow` file containing
> PR #32 head `d14b1dcc...`; Git's merge-base does not traverse beyond that
> boundary. An isolated read-only GitHub fetch recovered the PR #32 parent
> history and the EXP-052 base `a04c9bb...`: PR #32 is **95 commits ahead,
> 0 behind** its declared EXP-052 base, with exactly 107 files changed. It
> shares ancestor `d0cf7a242f9d57bf068892eb825127d6c5e3cc3c` with
> `main` (PR #32 is 161 ahead/2 behind current `main` in the fetched graph).
> **Use `EXP053-BRANCH-LINEAGE-NEGATIVE-PREFLIGHT-V2-SPEC-2026-10-19.md`**
> and v2 `exp053_branch_lineage_preflight.py`; it explicitly returns
> `UNVERIFIED_SHALLOW_HISTORY` instead of the misleading `DISCONNECTED`.
> Original below is retained as historical context but its disconnected-ancestry
> interpretation and resulting recommendation to avoid all integration are
> superseded. The 8-file isolated handoff review candidate is safely based on
> exact PR #32 head. This correction does NOT approve publication, merging or
> scientific execution. PR #32 is still a **DRAFT stacked research PR**.


**NO_GO_RESEARCH_ONLY. No pushes, merges, approvals, model execution or remote
host changes are authorized by this report.** Research code and a source-free
human custodian packet exist in a LOCAL shallow Git object history whose common ancestry was initially
unknown before the isolated history fetch confirmed it (see correction above).

## Verified local object-graph relationship

From the isolated `x1-370` research worktree at commit
`9357c25e605b728720dd078f8dd19eea8f7bda91`:

| Read-only check | Result |
|---|---|
| Local HEAD -> cached `origin/main` merge base | **None: DISCONNECTED** |
| Local HEAD -> cached PR #32 head merge base | **None: DISCONNECTED** |
| `origin/main` object present | Yes |
| PR #32 head cached locally | Yes |
| PR #32 base (`research/k2-minicpm-v47-exp052-20261016`) cached locally | **No** |
| Difference from cached PR #32 tree | **34 added, 4 modified, 1 deleted** |
| Difference from cached `origin/main` tree | **178 added, 5 modified, 5 deleted** |
| User worktree tracked modifications | **2 unrelated concurrent preflight edits** |
| Source-only handoff packager and request in published PR #32 | **Absent** |

The remote PR #32 metadata was separately checked through authenticated
GitHub tools, reporting it still **DRAFT**, head SHA
`d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28`, with the issue #31 research
hold intact. Local `origin/*` refs are cached and are not a freshness proof.
The remote PR #32 head here is a single-root commit, while `origin/main` and
local HEAD have different root commits. Directory overlap does not provide
Git ancestry or safe automatic merge semantics.

`research/exp053_branch_lineage_preflight.py` runs ONLY bounded read-only local
Git metadata calls with hardcoded refs and source-free output. It never
reparents, merges, pushes or creates a PR. Frozen source protocol:
`EXP053-BRANCH-LINEAGE-NEGATIVE-PREFLIGHT-SPEC-2026-10-19.md`, SHA256
`ec5805ba4b51d2d318250d61fea0f184ed07bf8ff75a199163c270b978a53fe8`.
Every output, even connected history, has
`integration_authorized=false`, `scientific_execution_authorized=false`,
`decision=NO_GO_RESEARCH_ONLY` and CLI exit 2.

## The smallest safe review path

1. Independent technical reviewer must confirm exact live GitHub PR #32 head
   and its base revision through authenticated API. Don't infer fresh state
   from stale `origin/` refs or recreate the PR base blindly.
2. Branch owner selects **only** the reviewed source-free custodian request,
   protocol descriptions and handoff packager; examine all file additions and
   modifications against the exact PR base, explicitly excluding unrelated
   source, model, research raw input and production files. The local branch
   has 39 tree changes from PR #32 and must NOT be cherry-picked as a unit.
3. Prepare a *new isolated review branch with the correct commit ancestry*,
   never force-update PR #32 or `main`; retain draft and NO-GO state.
   Do not copy private exam passages, owner/reviewer identities, keys,
   approvals, secrets, self-signed synthetic receipts or simulated evidence.
4. Run synthetic tests and independent source review; publish only after a
   human explicitly approves the exact source-only delta. A public comment,
   approval badge or GitHub review is NOT evidence of independently controlled
   witness state; technical review and human custodian approval are separate.
5. Share the local source-only packet with an independently accountable human
   **out of band**, only after reviewing its contents and organizational
   permissions. Its manifest proves byte consistency, not trusted
   publication, custody separation or source authenticity.
6. Only that separate human can designate/deploy an approved *nonproduction*
   witness and conduct privilege, backup/restore, signer, checkpoint and
   native rollback negative tests; scientific PR #32 remains draft until its
   separate blinded 40-article review and model/runner byte custody gates.

## Safe local reproduction

```
python3 research/exp053_branch_lineage_preflight.py \\
  --repo /home/scott/git/wt-lms-exp053-ci-remote-validation-20261016
python3 -m pytest -q tests/test_exp053_branch_lineage_preflight.py
```

All output is source-free, no raw patch/diff content, no account identity,
no remote URLs or private paths. No unauthorized branch mutation occurs.
