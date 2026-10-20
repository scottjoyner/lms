# EXP-053 — draft review: operator journal/marker inode and symlink provenance

Stacked on draft PR #43 (which itself rests on #42 through #32); isolated
technical source/test/docs only. Review `exp053_operator_run._consume_claim`,
`_pinned_operator_parent`, and `_require_operator_parent_stable`.

Review checklist: test all lexical ancestor paths with O_PATH and O_NOFOLLOW;
check final journal leaf O_NOFOLLOW and relative inode; check who owns the
parent directory; verify the locked descriptor remains unchanged across
CAS/marker/state writes; inspect `dir_fd` exclusive used-marker creation,
fsync and process shutdown behavior; ensure original human-owned journal
and *remote* independent high-water are NEVER automatically reset on
interrupted work. Determine whether same-UID post-validation races and
failure-to-return after remote CAS require independent operator recovery
protocol additions. Do not treat a synthetic local CAS as independent.

Tests use generated fake Ed25519 keys and ephemeral, fictional state/marker
bytes ONLY, with no actual reviewer or model. Reproduction:

```
python3 -m pytest -q tests/test_exp053_operator_run.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_operator_run.py \
    tests/test_exp053_operator_run.py
```

Independent human technical reviews, separately held operator/witness keys,
real off-host high-water service, two blinded reviewers with at least 40
co-reviewed/adjudicated source-distinct articles, and operator weight
approval remain missing. Owner and weight pins remain EMPTY. Do not deploy,
merge, create live keys, replay an operator authorization, rebuild a licensed
packet or perform model inference. **NO_GO_RESEARCH_ONLY.**
