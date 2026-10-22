# LMS open-issue execution plan and research-safe release — 2026-10-17

## Live backlog

Audited live GitHub open issues in `scottjoyner/lms` before planning:
`#31`, `#7`, `#5`, `#4`. PRs `#32`, `#30`, `#29` and earlier
research drafts remain distinct and unmerged. Do not equate passing source
CI with scientific or runtime admission.

| Priority | Issue | Deliverable / acceptance next |
|---|---|---|
| P0 scientific custody | #31 — EXP053 | 40+ source-locked, independently double-human-reviewed articles; signed identities, distinct plausible foils, genuine adjudication, authentic native model-score receipts and provenance; separately independently approved checkpoint/scorer keys and operator-run identity; true off-host rollback-resilient witness with independent administration and native runtime proof |
| P0 fleet release safety | #7 — first signed noncritical canary | Confirm pinned `lms` 0.32.0 and profile source commits, out-of-band recovery, stable runtime, independent allowed signers. Produce signed observation, exact-loadout qualification, sustained canary and verified automatic stable-runtime rollback. Pass negative drills; import profile with `admission.enabled=false`. No activation until external review |
| P1 package quality | #5 — production CLI hardening | Audit existing `src/lms_agent_bench` package, wheel/sdist, entrypoints, resource data, JSON schemas and release workflow. Run no-network clean install and hermetic smoke tests on nonproduction node before declaring done |
| P1 agent-evaluation quality | #4 — self-evaluation | Audit manifest-driven deterministic evaluators, exact run receipts and output isolation, context sweep, capability synopsis, benchmark comparison and evidence-grounded route rules; no automatic route changes |

## Implemented and deployed in this slice

**Only** the source-safe EXP053 physical witness qualification CLI is
authorized for a user-local **controller-side** deployment. It is not
the witness daemon and cannot serve CAS, sign credentials or run models.

- Prospective source-only host-scanning contract SHA256
  `523cbece51e6d966e868e859e728946fe7c15990e83f428143d1a08fb5dceeb5`.
- `research/exp053_witness_host_preflight.py` uses existing host-key-pinned
  SSH aliases to execute strict bounded read-only Python probes with
  fresh echoed challenge and typed responses. It never copies anything
  to the remote nodes.
- `research/exp053_preflight_release.py` installs the reviewed CLI and
  frozen protocol into a unique 20-hex content-addressed directory
  under the OS account home (not environment HOME) in a user-local prefix, with owner-only parent permissions. It
  validates SHA256 manifests, refuses symlinks, tampering or drift,
  creates a staging release and atomically renames it after fsync.
  Existing release paths are not overwritten; clean reinstall is
  idempotent. No systemd unit, ports, certificate/key files, model
  processes, privileged code or production changes.

### Controller deployment protocol

From the reviewed clean research worktree, run (the installer uses the OS account home, never an isolated session HOME):

```bash
ACCOUNT_HOME=$(python3 -c 'import os,pwd; print(pwd.getpwuid(os.getuid()).pw_dir)')
python3 research/exp053_preflight_release.py
python3 research/exp053_preflight_release.py --install
python3 research/exp053_preflight_release.py --verify \
  "$ACCOUNT_HOME/.exp053-readonly-preflight/<verified-release-id>"
python3 "$ACCOUNT_HOME/.exp053-readonly-preflight/<verified-release-id>/exp053_witness_host_preflight.py" lenovo destroyer xwing
```

The operator must verify the actual release ID and source hashes, not
copy this example path. Output is source-free and deliberately includes
`decision=NO_GO_RESEARCH_ONLY` and all four approval fields FALSE.

### Live read-only physical preflight

On Oct 17, 2026, already host-key-verified SSH reached Lenovo,
Destroyer and Xwing; all expose Python 3.12.3, SQLite 3.45.1,
cryptography, OpenSSL and ext4. Approx free GiB: Lenovo 40.4,
Destroyer 74.2, Xwing 81.2. TPM device reported: Lenovo no,
Destroyer/Xwing yes (NOT proof of usable TPM custody).
All three meet local tooling checks but **NONE** meets the independent
administration, trusted key, anti-rollback witness or scientific authority
gates. They use existing controller SSH administration.

## Gates deliberately NOT crossed

No model forwards, reviewer enrollment, key creation, witness daemon,
production network listener, runtime canary, routing, fleet benchmark
load, model downloads, production deployments, NAS/Neo4j writes,
profile admission, scientific promotion, PR merge, or release tag.
Independent physical canary and scientific evaluations are separate
future operations, not implied by deploying a read-only diagnostic.

## Verified physical controller-side deployment

On October 17, the Fleet Commander session's `HOME` pointed through
an isolated, symlinked research environment; the original installer
correctly refused that route without creating files. The code was
refactored to anchor user-local releases to the canonical OS account
home directory via `pwd.getpwuid(os.getuid()).pw_dir`, not untrusted
session `HOME`. The existing `.local` symlink was untouched.

After rerunning the entire local regression suite, source release ID
`76aaf673b47b925e76e8` installed successfully under the
OS account home's `.exp053-readonly-preflight/` directory. Its parent
and version directory both have permission 0700; frozen protocol and
Python runner SHA256 match exact tested source. The deployed copy
independently re-ran read-only, host-key-verified diagnostics on the
three explicit aliases. All three returned
`status=READ_ONLY_CANDIDATE` while the report and all per-host
scientific authority fields remained `NO_GO_RESEARCH_ONLY`/FALSE.

No witness service was started and no model, router, reviewer
enrollment or scientific backend was enabled. This is a controller
research diagnostic release only.
