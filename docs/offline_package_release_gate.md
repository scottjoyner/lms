# LMS 0.33.0 offline wheel validation and non-admitted staging

**Issue:** #5 production CLI/package hardening. **Baseline:** clean Git
commit `d0cf7a242f9d57bf068892eb825127d6c5e3cc3c`. October 18, 2026.

## Reproduced defect

The LMS 0.33.0 wheel itself built correctly. When the wheel was installed
into a virtual environment and `lms-bench --help` was invoked from
`/tmp` (outside the source checkout), Python raised
`ModuleNotFoundError: No module named 'lms_agent_brief'`.

The installed `src/lms_agent_bench/lms_cli_v2.py` imported legacy
top-level checkout shims instead of its package's own canonical modules.
A source-checkout or editable installation masked the missing module.

## Fix and acceptance

Refactored the packaged CLI to import its canonical internal modules
relatively. `lms_history_decide` supports both the canonical packaged
import and legacy direct-module test import; neither path requires
an uninstalled top-level compatibility shim.

`scripts/offline_wheel_smoke.py` is a source-only non-execution test:
it builds a real wheel with `pip wheel --no-index --no-deps
--no-build-isolation`, installs that exact wheel into a temporary
virtual environment, invokes eight `--help` CLI entrypoints from
an unrelated temporary working directory, asserts the imported
`lms_agent_bench` module resolves under the disposable installed
environment, and checks packaged schemas/benchmark JSON resources.
It uses local dependencies made available by the CI base environment;
it performs **no model forward, network request, fleet connection,
model download, routing change or production activation**.

GitHub's Python 3.10/3.11/3.12 CI now builds both sdist and wheel,
executes its normal tests, then runs this *offline wheel* acceptance.
Unit tests also statically reject checkout-only absolute `lms_*`
imports in the two affected packaged modules.

Locally, Ruff passed; full clean-source pytest passed **222/222**;
offline installed-wheel CLI smoke passed all eight entrypoints;
and sdist/wheel built using PEP517 with an existing local
setuptools backend. Builds did not access a model runtime.

## Release and open-issue separation

- **#5 package quality:** fixes one real installation defect.
  Source-only PR and CI must pass before a production package release.
- **#7 signed physical canary:** this packaging check is NOT the
  independent three-signed-root observation/loadout/canary proof,
  nor a 30-minute physical soak/rollback acceptance.
- **#4 agent evaluation:** no auto-routing or agent model loading.
- **#31 EXP053:** completely independent human review, source/score
  admission, key custody and external high-water approval gates remain
  EMPTY; zero approved articles and zero new EXP053 model forwards.

An operator may install the source-pinned wheel into a separate,
explicitly named controller-local staging venv for `--help` smoke.
Do NOT modify the shared active Python environment, service units,
public PATH, admission/routing profiles, release tags, or fleet
deployments. A staging installation is not a production rollout.
