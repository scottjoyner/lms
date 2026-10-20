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
