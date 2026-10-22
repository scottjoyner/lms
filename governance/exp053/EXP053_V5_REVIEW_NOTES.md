# EXP-053 Governance V5 — tracked follow-up to draft PR #48

This review-only increment hardens the proposed GitHub branch-ruleset deployment.
It does not activate the policy or permit merges or inference.

## Security changes

- The original policy JSON and historical branch/CI evidence must both match immutable SHA-256 digests before the administrator script can proceed.
- The read-only policy audit verifies the historical evidence digest before any GitHub API reads. A modified branch roster cannot silently shrink coverage.
- Two new negative tests alter evidence in a temporary directory and assert denial without remote calls.
- A dedicated GitHub Actions workflow runs all governance synthetic tests and the zero-write administrator dry run for governance-related changes.
- The workflow uses read-only contents permissions and does not pass an administrator token or call live service endpoints.
- The new workflow is supplemental, not one of the five previously verified required checks.

## Reproduce locally

From governance/exp053, run: `python3 -m unittest -q test_exp053_admin_activation test_exp053_pr_head_ci_audit test_exp053_merge_ref_ci_audit test_exp053_verify_live_enforcement`.
Then run `python3 EXP053_validate_ruleset_candidate_2026-10-20.py --dry-run` and `python3 exp053_admin_activation.py`.

## Remaining holds

Repository rules are NOT active merely because this PR or CI passes.
Only an authorized human administrator may decide to activate them, using authenticated GitHub verification and a non-destructive negative merge challenge.
Real independent human code review, off-host custody and blinded scientific adjudication remain separate gates.

- Governance: NO_GO_GOVERNANCE.
- EXP-053 science: NO_GO_RESEARCH_ONLY.
- No merges, signer enrollment, licensed RACE processing, model inference, production deployment, or ruleset mutation is authorized.

Tracking: governance issue #47, scientific issue #31, draft PR #48.

## V6 follow-up — branch-inventory race detection (2026-10-20)

Commits [3388119](https://github.com/scottjoyner/lms/commit/3388119939c4b5981e8674b50e4cd011e8c85199) and [5cbdcfc](https://github.com/scottjoyner/lms/commit/5cbdcfc49b987aa4257e0b3dd3ecacee4f75756a) harden the existing read-only verifier, without modifying the proposed GitHub ruleset.

- After auditing each observed target branch and re-reading the active ruleset, the verifier now lists target branches a second time.
- A newly created or deleted `main` / `research/exp053*` target fails closed with `branch_inventory_stable=false`; the audit must be restarted rather than accepting an incomplete branch snapshot.
- Two new network-free negative tests simulate branch creation/deletion between the inventory reads.
- The exact code head `5cbdcfc49b987aa4257e0b3dd3ecacee4f75756a` passed **48/48** tests and all three commands in [EXP-053 Governance Offline run 39505848581](https://github.com/scottjoyner/lms/actions/runs/39505848581). The admin helper reported `OFFLINE_DRY_RUN_VALID` / `writes_performed: 0`.

**Limits:** Two successful inventory reads are not a transactionally consistent snapshot; a branch or policy may change immediately after the last read. The verifier explicitly claims metadata verification at most. The `do_not_enforce_on_create=true` status-check exception in the candidate still needs accountable administrator approval and observed branch-creation behavior. Effective rules, negative merge-eligibility challenges, bypass visibility, and independent human reviews remain prerequisites for governance release.

**HOLD remains** `NO_GO_GOVERNANCE` / `NO_GO_RESEARCH_ONLY`. No ruleset activation, merge, code-owner/reviewer assertion, production deployment, or scientific execution.

## V7 follow-up — current-head CI preflight before an authorized ruleset activation (2026-10-20)

Source-only commits [41c9b61](https://github.com/scottjoyner/lms/commit/41c9b61504fa9de074dbc000af0567fca69ec53f) and [848758a](https://github.com/scottjoyner/lms/commit/848758a1cbb09f3d7cb62bf7f08cd388733a6211) add an activation-time, **read-only** check over current `main` and all existing `research/exp053*` branch HEADs.

Before any authorized POST, the installer now requires the five pinned required status checks on each current HEAD to be uniquely present, successful and source-bound to GitHub Actions app `15368`; it verifies each branch identity/protection state, reads each HEAD again for stability, and rejects target-roster changes mid-scan. Missing, duplicated, pending, wrong-app or wrong-head checks fail closed. Unexpected existing protection fails closed for independent policy review. The installer still performs **zero writes by default**; its explicit opt-in path is unchanged.

- Local Fleet Commander x1-370: **54/54** synthetic offline tests PASS, candidate validator PASS, default dry run `OFFLINE_DRY_RUN_VALID` / `writes_performed: 0`, diff check clean.
- [Hosted EXP-053 Governance Offline run 39507559334](https://github.com/scottjoyner/lms/actions/runs/39507559334): **54/54 tests PASS**, policy preflight PASS, dry run zero writes.
- A **read-only, public GitHub API probe of `main`** with the added method passed. Other target-head paths were not independently probed live using administrator credentials; production activation remains unattempted.
- GitHub connected updates match the SHA-1 Git blob hashes of locally tested source files byte-for-byte. Candidate JSON and pinned evidence unchanged.

**Limitations:** preflight API reads are not atomic and cannot guarantee GitHub state after the last GET. A passing CI check is not approval, and `do_not_enforce_on_create=true` still needs an explicit owner decision and post-install negative enforcement challenge. No independent reviewer custody or science authorization is implied.

Still draft/unmerged; `NO_GO_GOVERNANCE` / `NO_GO_RESEARCH_ONLY`. No ruleset activation, bypass, merge, signing, licensed RACE processing, model execution or fleet deployment.
