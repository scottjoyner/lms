# Archive — lms repository

This directory holds dead, orphan, or duplicate scripts removed from the
repository root during the lms packaging/unification remediation
(docs/LLD_UNIFIED_FLEET.md §3.6, work-items W-67/W-68).

They are MOVED here (not deleted) so history is preserved. Each entry notes why.

## Archived files

- `lms.py` — 34KB duplicate of `lms_cli.py` (canonical CLI). Superseded by
  `src/lms_agent_bench/lms_cli.py`. Archived 2026-07-16.
- `benchmark_lmstudio_inventory_concurrent.py` — overlapping benchmark runner;
  `benchmark_lmstudio_inventory.py` + `bench_fleet.py` + `bench_concurrency_probe.py`
  are the canonical runners. Not imported anywhere. Archived 2026-07-16.
- `benchmark_lmstudio_two_phase_concurrent.py` — overlapping benchmark runner;
  same canonical set as above. Not imported anywhere. Archived 2026-07-16.

To restore any of these, move them back to the repo root and update references.

- `bootstrap_keys.sh` — installed runner PUBLIC SSH keys via inline
  `authorized_keys` append. Moved to archive 2026-07-16 (W-71): keys/paths should
  be templated or managed out-of-band rather than committed inline.
