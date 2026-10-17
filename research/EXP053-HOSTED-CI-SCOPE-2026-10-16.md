# EXP-053 hosted CI recovery / scientific scope (2026-10-16)

## Observed remote failure and resolution
Draft PR #32 GitHub Actions run 38997754551 (head fb38606b0b5ccaeca140a8a73eb00572b01c1ddd) failed **Ruff lint before pytest** on Python 3.10; 3.11 and 3.12 matrix jobs cancelled. The archived runner log reported 18 simple E401/F401 import errors in older K2 experiment tests. The same full scope under local Ruff 0.14.3 found 26 E401/F401 issues (differences are Ruff release/repository tree), all resolved by automatic unused-import deletion / import separation without scientific test logic edits (22 affected tests). Full `ruff check src tests` now passes locally.

## Explicit hosted / local scientific separation
Normal hosted `.github/workflows/ci.yml` installs `.[test]` but intentionally does **not** install PyTorch / NumPy / Transformers, download checkpoint weights, mount local QASPER/QuAC/RACE source, or create human-review identities. Direct `pytest -q` on this lean environment cannot collect the older `test_k2_*.py` set: 29 collection errors on local reproduction (torch, numpy, other research-only requirements).

For ordinary hosted CI, run `pytest -q --ignore-glob='tests/test_k2_*.py'`, printing the excluded file list as an explicit receipt. Under isolated x1-370 Python 3.12 with `.[test]`, this command passed **274 tests**, unchanged semantic test assertions. It includes all EXP051/052/053 source-free audits and the offline browser source-free contract checks; optional local-source checks skip when unavailable.

The `test_k2_*.py` scientific integration suites remain in the repository **unchanged except non-semantic import cleanup**. They are not claimed as passing in hosted CI. Execute them only on the research host with independently pinned source files, optional Torch/Transformers/NumPy and checkpoint identity, while preserving no-production-inference policies. Do not simulate missing source files or auto-download corpora in a public GitHub Action. The normal hosted pass does **not** qualify EXP053 science promotion, new model forwards, private data exposure, RACE license compliance, or owner review of independent human attestations.

## Gate status
At this checkpoint: the source-only tokenization preflight (384 prompts/model, 1,536 label boundary checks/model) passed, 43 EXP053 tests plus 10 prior archive audits passed on the research host, 274 hosted-scope tests passed locally, **0** new EXP053 native model forwards, **0** signed independently reviewed articles; minimum 40 still required. PR remains draft, no deployment/weights/NAS/Neo4j writes.
