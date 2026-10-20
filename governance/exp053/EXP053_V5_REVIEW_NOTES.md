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
