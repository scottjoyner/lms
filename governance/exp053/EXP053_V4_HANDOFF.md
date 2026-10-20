# EXP-053 Governance V4 — source-bound rule verification and correct check-source selection

**2026-10-20 | LMS issue #47 | NO_GO_GOVERNANCE | NO_GO_RESEARCH_ONLY**

## What changed

- `exp053_verify_live_enforcement.py`: binds each effective-branch rule to **the selected ruleset ID**, `ruleset_source=scottjoyner/lms`, and `ruleset_source_type=Repository`. An unrelated overlapping ruleset cannot falsely satisfy the rule matcher. Also SHA-256-pins the original candidate and re-reads its details at the end to fail closed on mid-scan policy mutations. The verifier still refuses redacted/missing bypass permissions.
- `exp053_merge_ref_ci_audit.py`: implements GitHub's documented check selection: **test-merge commit has checks/statuses → evaluate test-merge; otherwise evaluate PR head**. Separately handles legacy commit statuses that collide with named checks and checks snapshot stability/merge-parent identity. Does not assert GitHub will allow merging.
- Supplemental negative tests cover foreign effective rules, wrong rule IDs, policy changes mid-snapshot, head fallback, queued merge reruns, unrelated merge activity and failing same-name legacy statuses.

## Live verification

Connected Fleet Commander on x1-370 queried public GitHub API. `GET /rulesets` and `GET /rules/branches/main` both returned empty arrays. PR #46 is still draft; its generated merge SHA has **zero checks and zero legacy statuses**, while its HEAD has **11 successful checks including all five required names from GitHub Actions app 15368**. Per official GitHub troubleshooting documentation, HEAD checks can be considered when no status is on the test-merge commit. This does **not** prove strict base freshness, enforcement or mergeability under the still-missing ruleset.

The local container cannot reach GitHub directly in this session; live observations are recorded from Fleet Commander, not a direct run of the V4 scripts against the public API.

## Validation commands (read-only by default)

```bash
python3 -m unittest -v test_exp053_admin_activation test_exp053_pr_head_ci_audit test_exp053_merge_ref_ci_audit test_exp053_verify_live_enforcement
python3 EXP053_validate_ruleset_candidate_2026-10-20.py --dry-run
python3 exp053_admin_activation.py
# When run on an authorized Internet-connected administrator machine:
python3 exp053_verify_live_enforcement.py --json-out policy_readback.json
python3 exp053_pr_head_ci_audit.py --json-out head_checks_readback.json
python3 exp053_merge_ref_ci_audit.py --all --json-out merge_selection_readback.json
```

The activation helper **has not been invoked with `--activate`**. Owner must first intentionally authorize policy rollout, select/confirm independent human reviewers and emergency access, and use authenticated GitHub admin controls. The active ruleset must be checked for its effective branch coverage, no bypass actors, and actual rejection of unreviewed merges (a **non-destructive** eligibility challenge). Do not consider the helper or the GitHub API to be a substitute for human review or scientific admissibility.

**No rule created or changed; zero PR approvals or merges; zero scientific model forwards.**

## Official references

- GitHub: [Troubleshooting required status checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
- GitHub: [REST API endpoints for rules](https://docs.github.com/en/rest/repos/rules)
- GitHub: [Rulesets in public repositories](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets)
