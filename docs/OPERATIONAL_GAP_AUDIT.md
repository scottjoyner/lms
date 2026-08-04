# Operational gap audit

This audit separates repository-enforced controls from physical controls that
still require execution on real fleet nodes. The system remains fail-closed and
non-admitting throughout observation, qualification, canary, and profile import.

## Closed repository-side gaps

| Failure class | Enforced control | Evidence |
|---|---|---|
| Partial or stale controller writes | Atomic JSON publication with fsync and replacement | Run state and manifests |
| Concurrent controller execution | Host/boot/PID-owned locks with explicit stale recovery | Lock owner record |
| Concurrent remote execution | Remote host/boot/PID lock owned by the actual shell process | Remote owner record |
| Moving Git branch | Fetch branch, require exact configured commit, fast-forward only to that commit | Source-control provenance |
| Dirty or wrong source | Clean branch/commit checks before and after execution | Preflight/postflight evidence |
| SSH trust bypass | Strict known-hosts by default; explicit first-contact acknowledgement only | Operator trust mode |
| Transient SSH/SCP failure | Bounded transport-only retries; workloads are not silently rerun | Attempt records |
| Partial archive promotion | Per-attempt partial file and atomic final promotion | Collected archive manifest |
| Unsafe paths or symlinks | Input, run-ID, archive, manifest, and artifact containment validation | Verification report |
| Disk/file-limit/clock readiness | Controller and remote readiness gates | Preflight reports |
| Hung subprocess | Process-group timeout, TERM/KILL escalation, and state finalization | Command records |
| Manual qualification sequencing | One locked throughput -> base Hermes -> context Hermes -> bind -> verify command | Qualification-run manifest |
| Secret in Hermes argv | API keys inherited by environment name, never placed in subprocess arguments | Secret-safe command tests |
| Self-hash without authentication | OpenSSH detached signatures and external allowed-signers policy | Fleet and qualification attestations |
| Candidate passes short tests but degrades under load | Sustained canary soak gate with success, latency, RSS, TPS, and temperature controls | Soak samples and summary |
| Candidate cannot be removed safely | Transactional candidate stop, rollback, and rollback health verification | Runtime-canary state |
| Failure after candidate startup | Automatic best-effort stop and rollback; original run remains failed | Recovery command records |
| Shell injection in lifecycle plan | Installed canary entrypoint rejects shell interpreters and secret-bearing flags | Plan validation |
| Canary evidence rewrite | Manifest hashes every artifact; optional OpenSSH attestation binds rollback state | Canary manifest/attestation |
| Qualification/profile evidence splicing | Profile importer verifies independent signed evidence roots and the raw internal chain | Attested profile import |
| Accidental admission | All producer and consumer artifacts explicitly remain non-admitted | Admission fields and tests |

## Execution stages and authority

```text
fleet observation
  authority: observe only
  output: signed fleet evidence

exact-loadout qualification
  authority: exercise an already running loopback candidate
  output: signed throughput + Hermes qualification

runtime canary
  authority: start one reviewed candidate and restore the previous runtime
  output: signed soak + rollback evidence

profile import
  authority: write static desired-state evidence only
  output: admission.enabled=false profile

live admission
  authority: external and independent
  output: runtime may receive routed work
```

No earlier stage may infer the authority of a later stage.

## Physical failure-injection matrix

The following items cannot be proven by repository unit tests. They must be run
on a noncritical canary node, with the previous runtime and an out-of-band access
path available.

| Drill | Expected behavior | Required proof |
|---|---|---|
| Unknown SSH host key | Preflight stops before remote execution | Failed preflight and unchanged node |
| Network loss during preflight | Bounded transient retry, then failure | Attempt records and no run directory on node |
| Network loss during archive transfer | Retry transfer only; no workload rerun | Multiple transfer attempts, one final archive |
| Network loss during canary | Canary fails and local recovery executes where possible | Failed canary plus rollback evidence |
| Candidate process crash | Health/soak fails, candidate stop is idempotent, rollback succeeds | Crash log and rollback health proof |
| Qualification timeout | Process group terminates, canary enters rollback | Timeout record and restored service |
| GPU/accelerator reset | Candidate fails closed and previous runtime can restart | Device/runtime logs and rollback health proof |
| Disk exhaustion before run | Readiness gate prevents execution | Preflight failure |
| Disk exhaustion during run | Artifact write fails; run cannot verify or sign as successful | Failed manifest/state and no trusted signature |
| Controller interruption | Atomic state remains parseable; stale lock requires explicit recovery | Archived stale lock and new run ID |
| Node reboot during run | Boot identity invalidates prior lock; evidence remains failed | Boot ID mismatch and stale-lock archive |
| Thermal throttling | Terminal TPS or temperature gate fails | Soak telemetry and rollback proof |
| Memory leak | RSS growth gate fails | Soak sample series and rollback proof |
| Intermittent endpoint errors | Success/consecutive-failure gates fail | Probe sample series |
| Rollback command failure | Run ends `rollback_failed`; must not be signed with `--require-success` | Recovery failures and external intervention |
| Rollback health failure | Run remains failed even when rollback command exited zero | Health log and failed state |
| LAN path loss | Tailscale or loopback path is checked independently; no hidden path substitution | Per-path health evidence |
| Tailscale path loss | LAN fallback is explicit and policy-controlled | Per-path health evidence |
| Power interruption | Service manager restores the reviewed stable runtime or remains fail-closed | Boot-time service and health evidence |
| Multi-node rollout interruption | Canary cohort halts; unaffected nodes retain previous profiles | Cohort controller evidence |

## First physical acceptance sequence

Use one noncritical node and one exact reviewed loadout:

1. Verify out-of-band access and the previous stable runtime.
2. Run and sign fleet observation.
3. Start the reviewed candidate and run signed exact-loadout qualification.
4. Run `lms-runtime-canary` with at least a 30-minute soak.
5. Verify and sign canary evidence only after rollback health succeeds.
6. Repeat with an intentional qualification failure.
7. Repeat with an intentional soak-probe failure.
8. Repeat with candidate process termination.
9. Repeat with a controlled network interruption.
10. Import the three signed evidence roots into a non-admitted profile.
11. Independently review rollback and live-access evidence.
12. Grant time-bounded canary admission externally, not from this repository.

## Remaining architecture improvements after first physical proof

These should not be enabled before the first signed canary succeeds:

- cohort canary rollout with one-node-at-a-time promotion;
- automatic pause on fleet-wide error-budget consumption;
- systemd-user and launchd lifecycle adapters with fixed typed configuration;
- signed live-admission leases with expiry and revocation;
- continuous profile freshness and rollback-health monitoring;
- record-only prompt-cache observations under sustained workload;
- experimental KV save/restore only behind an independent compatibility gate;
- FalkorDB metadata replication only after local cache behavior is proven.

## Current boundary

Repository execution now covers deterministic observation, exact-loadout
qualification, authenticated evidence, sustained canary testing, and verified
rollback. The remaining blockers are physical execution and independent live
admission—not missing happy-path orchestration.
