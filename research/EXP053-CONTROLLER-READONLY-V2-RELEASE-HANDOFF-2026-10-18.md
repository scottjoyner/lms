# LMS open-issue next slice — content-addressed controller read-only toolkit

**Date:** October 18, 2026. **Status:** research-only controlled deployment;
**not** scientific model approval, fleet runtime rollout, or independent custody.

## Live backlog and gates

Four live open issues in `scottjoyner/lms` were checked before execution.

| Priority | Issue | Next acceptance boundary | Authorized in this slice? |
|---|---|---|---|
| P0 — science | #31 EXP053 human-grounding falsification | Two independent humans, >=40 article-disjoint genuine source verdicts, separately signed checkpoint/scorer/runner custody, independent durable external witness | **No**, keep draft PR #32 and scoring on HOLD |
| P0 — operations | #7 signed physical canary / rollback | Reviewed exact noncritical node/model, signer policy, stable recovery, three independent signed evidence stages, full rollback, *non-admitted* profile | **No**, no runtime process, model load or persistent routing changes |
| P1 — package | #5 production CLI | Hermetic clean wheel/sdist + install, assets, schemas, CI, launch smoke tests | Read-only audit/build checks only |
| P1 — agent evaluation | #4 skill/capabilities | Deterministic manifest fixtures, self-contained run evidence, measured routing synopsis, no autonomous routing | Review/tests only |

## Implemented refactor

The controller-only `research/exp053_preflight_release.py` now packages
**both** existing research-safe diagnostics in one immutable, content-addressed
user-local release:

- `exp053_witness_host_preflight.py` + frozen protocol, which uses
  explicit SSH aliases, pinned host keys, fresh echo challenge, and strict
  read-only Python probes; it CANNOT create a witness or model run.
- `exp053_ssh_custody_overlap.py` + controller read-only custody audit
  handoff, which uses `ssh -G` on trusted local SSH config and never
  connects to a host. It intentionally returns **exit 2** and marks
  independence **NO_GO_RESEARCH_ONLY**, even when config overlap is checked.

All four payload files are hashed into the release manifest. Any payload
tampering, added file or hash drift fails verification. Existing releases
are never overwritten; changing source produces a new release ID. The
installer writes only to the canonical OS account home under
`~/.exp053-readonly-preflight/<content-hash-id>/`, with a staged fsynced
atomic rename, no privilege elevation or systemd. The old controller
release is preserved for reproducibility. No remote agent deployment.

This work is in a **separate clean worktree**; a concurrent uncommitted
SSH auditor extension in the original worktree was deliberately NOT
included in the reviewed release or overwritten.

## Essential caveats

Existing controller aliases `lenovo`, `destroyer`, and `xwing`
show configured SSH key path overlap and use the same fleet controller.
No evidence grants independent administration, separate key/certificate
custody, robust native logs, monotonic off-host witness or scientific
execution authority. A physical host alone is NOT a rollback-resistant
trust boundary.

A user-local read-only research diagnostic is the **only** authorized
deployment here. Do not merge draft EXP053 PR #32, start a model forward,
create/enroll keys, install external witness software, alter NAS/Neo4j,
start an experimental canary, import admitted profiles, alter production
routing, or publish scientific findings.
