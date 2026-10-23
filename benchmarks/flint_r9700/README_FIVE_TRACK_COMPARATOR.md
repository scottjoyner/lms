# Flint benchmark comparison research — October 23, 2026

`compare_receipts.py` only checks two offline, locally supplied JSON-like
measurement structures. It does not run models, query providers, attest
weights, or control GPU allocation. Even structurally comparable receipts
return `production_admission=NO_GO`.

Reconciled conditions: pinned Flint source revision, opposite known IQ2/IQ3
weight hashes and byte counts, exact R9700/Vulkan1 runtime label, threads,
GPU layers, flash attention, valid profile and paired prompt/decode token
counts. Invalid/partial/truncated/failed receipts, non-finite metrics,
fractional counts or absent data never produce a comparison.

```bash
cd /home/scott/git/wt-lms-flint-metrics-validation-20261023/benchmarks/flint_r9700
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -p 'test_*.py' -q
```

Comparisons are **not** full-versus-Q4 weight quality evaluations, and no
physical source/model identity, steady-state thermals, independent grading,
or complete prompt trace custody was established.
