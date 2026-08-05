# Fleet operational reliability contract

This document defines the execution reliability boundary for physical fleet
evidence. A benchmark is not trustworthy merely because a model answered
requests. The controller, SSH trust, remote source, locks, storage, clocks,
transport, artifacts, signatures, and post-run state must also be trustworthy.

## Required execution sequence

```text
controller readiness and input validation
        |
        v
strict SSH trust and complete remote preflight
        |
        v
immutable exact-commit render
        |
        v
all-node rollout with bounded timeouts
        |
        v
atomic retried artifact collection
        |
        v
all-node postflight
        |
        v
archive/provenance release gate
        |
        v
atomic operator state + operator-manifest fingerprint
        |
        v
independent lms-fleet-operator verify
        |
        v
optional OpenSSH detached evidence attestation
```

No later phase may turn an earlier failure into success.

## Controller readiness

Before any SSH connection, `lms-fleet-operator` requires:

- regular configuration and private environment files;
- neither controller input may be a symbolic link;
- both files must be owned by the controller user on POSIX systems;
- SHA-256 identities for both inputs;
- `ssh` and `scp` installed;
- a real, nonsymlinked writable workspace;
- at least the configured controller free-space threshold;
- no group or world access to the private environment file by default;
- positive timeout and capacity limits;
- a run ID restricted to 1-128 ASCII letters, numbers, dots, underscores, and
  hyphens, with no surrounding whitespace or `..` sequence.

An insecure environment-file mode requires the explicit
`--allow-insecure-env-file` exception and is recorded as a warning. This flag
should be limited to a temporary migration, not routine operation.

The controller acquires its real execution lock before creating the run
directory. Therefore an overlapping or unrecoverable stale lock cannot leave a
new orphan run directory.

## SSH trust

The default is:

```text
BatchMode=yes
ConnectTimeout=10
ConnectionAttempts=1
ServerAliveInterval=15
ServerAliveCountMax=2
StrictHostKeyChecking=yes
LogLevel=ERROR
```

Host-key verification may never be disabled, and known-host files may not be
redirected to `/dev/null`.

First contact requires both:

```text
--accept-new-host-keys
StrictHostKeyChecking=accept-new
```

The operator passes both intentionally and records
`ssh_trust_mode=accept_new_explicit`. After bootstrap, rerun with strict known
hosts. A changed host key must stop execution.

Preflight retries only transport-level failures: SSH timeout or return code 255.
A deterministic remote-readiness failure is never retried automatically. The
full attempt history is recorded. Per-node controls are:

```text
preflight_attempts
preflight_retry_backoff_seconds
```

Git remote URLs are never copied into readiness artifacts. The operator records
only a SHA-256 fingerprint of the configured remote, preventing embedded Git
credentials from leaking into logs.

## Remote readiness

Every node must prove:

- `git`, `tar`, and `gzip` are available;
- configured Python exists and imports `requests`;
- the checkout is a Git repository and completely clean;
- branch and commit match exactly when code update is disabled;
- every model root exists and is readable/searchable;
- artifact and lock roots can be created and written;
- artifact filesystem free space exceeds the configured floor;
- the open-file soft limit exceeds the configured floor;
- the node clock is within the configured skew limit;
- no active or ambiguous remote rollout lock exists;
- the run ID has no pre-existing artifact directory or archive.

Optional `expected_hostnames` makes canonical hostname matching mandatory.

## Exact-commit updates

`--update-code` may not follow a moving branch with `git pull`. The remote
script instead:

1. proves the checkout is completely clean;
2. fetches the configured branch;
3. verifies `FETCH_HEAD` equals the configured 40-character commit exactly;
4. verifies the current local commit is an ancestor of that exact commit;
5. fast-forwards only to the exact fetched commit;
6. verifies final `HEAD` equals the configured commit.

A branch move, rollback, or divergent checkout fails before benchmark
execution. This prevents a failed provenance check from leaving a node changed
to an unintended branch tip.

## Lock behavior

### Controller lock

The controller lock records:

- host;
- process PID;
- boot ID;
- run ID;
- config SHA-256;
- start time.

An active lock always fails. `--recover-stale-lock` works only when the owner is
provably stale on the same controller: the boot changed or the owner PID no
longer exists. Foreign-host and corrupt locks fail closed.

### Remote lock

The remote lock records the Bash process actually holding the lock rather than
a short-lived helper process. It also records host, boot ID, run ID, and start
time.

When acquisition finds an existing lock:

- active, corrupt, or foreign-host ownership fails;
- a same-host lock from an earlier boot or dead PID is atomically renamed to a
  retained `.stale.*` directory;
- the new lock is acquired only after the stale rename succeeds.

The EXIT trap removes only the lock acquired by the current rollout.

## Timeouts and interruption

Every controller subprocess runs in its own process group with a phase-specific
timeout. Timeout handling sends `SIGTERM`, waits briefly, then sends `SIGKILL`
when required. The result records timeout, duration, command, log, and log
SHA-256.

The rollout timeout defaults to the sum of every selected node's configured
remote and SCP timeout plus controller overhead. It may be overridden, but must
never be unbounded.

An interruption records `failure_stage=interrupted`, writes final state and the
run manifest, releases the local lock, and exits with code 130.

## Artifact transport

The remote workload is never automatically rerun after it finishes. Only the
immutable compressed archive may be retried.

Each SCP attempt writes a unique file such as:

```text
.<node>.tar.gz.attempt-<n>.partial
```

A successful nonempty partial file is atomically promoted to the final archive
name and its parent directory is synchronized. Failed and timed-out partial
files are removed. The final archive path is never exposed to a partial write.

Per-node controls are:

```text
scp_attempts                     default 3
scp_retry_backoff_seconds        default 2.0
scp_timeout_seconds              default 300
```

Every attempt records timing, return code, timeout state, stderr, and partial
size. Retrying SCP does not rerun the model workload.

## Partial failure behavior

Rollout uses all-node `--continue-on-error` so one failed node does not hide the
state of the others. A nonzero rollout still proceeds to postflight and, when
`rollout_results.json` exists, the release gate. This preserves diagnostics
without converting the run into success.

Success requires all three:

```text
rollout return code == 0
all postflight checks == true
release gate return code == 0
```

## Postflight

Postflight reconnects to every node and proves:

- source is now the exact configured branch and commit;
- the checkout is still clean;
- no active or ambiguous remote lock remains;
- the expected artifact directory exists;
- the expected compressed archive exists and is nonempty;
- disk, file-limit, model-root, hostname, and clock checks still pass.

A valid archive with a dirty checkout or abandoned lock is not a successful
operator run.

## Artifact verification

The release gate independently checks:

- every required node appears exactly once;
- remote and SCP return codes and timeout states;
- local archive size and SHA-256;
- safe tar member paths and member types;
- bundle manifest file size and SHA-256 entries;
- bundle fingerprint, run ID, node ID, and source fingerprint;
- exact source provenance;
- observation fingerprints;
- selected-model full content SHA-256;
- reliability evidence for sweep mode.

The operator then writes `operator-manifest.json`, which fingerprints the local
control files and references every collected archive by its recorded size and
SHA-256. Archive references must resolve inside the immutable run directory,
and duplicate control or archive entries are rejected.

Verify later with:

```bash
lms-fleet-operator verify \
  --run-dir /path/to/workspace/<run-id> \
  --require-success
```

Any edited control file, missing archive, changed archive size, changed archive
hash, duplicate entry, or archive path outside the run directory invalidates the
run.

## Authenticated evidence attestation

The operator manifest's canonical hash detects accidental edits but, by itself,
does not authenticate an actor who can rewrite both content and hash.
Production evidence can be sealed with an OpenSSH detached signature:

```bash
lms-fleet-attest sign \
  --run-dir /path/to/workspace/<run-id> \
  --key ~/.ssh/lms-evidence-signing \
  --require-success
```

The signing key must be a regular nonsymlinked file, owned by the current user,
with no group or world permissions. Signing writes:

```text
operator-manifest.json.sig
operator-attestation.json
```

The attestation records the manifest hash, signature hash, namespace, signing
key fingerprint, run ID, success state, and its own canonical fingerprint.

Independent verification uses an OpenSSH allowed-signers file:

```bash
lms-fleet-attest verify \
  --run-dir /path/to/workspace/<run-id> \
  --allowed-signers /secure/policy/lms_allowed_signers \
  --identity fleet-operator-prod \
  --require-success
```

The allowed-signers file should be controlled separately from the run evidence.
Verification recomputes the operator manifest, attestation, manifest SHA,
signature SHA, run identity, and OpenSSH signature before returning success.
Signing and verification never grant runtime admission.

## No automatic resume

A run ID is immutable. Preflight rejects pre-existing remote artifacts, and the
controller rejects an existing local run directory. The operator does not
resume a partially completed run because doing so could combine evidence from
different source states, clocks, thermal states, or runtime processes.

After a failed run:

1. preserve the failed run directory;
2. verify and review its manifest and failure stage;
3. correct the underlying issue;
4. use a new run ID;
5. rerun complete preflight and postflight.

## Retention boundary

No automatic deletion is performed. Failed runs, stale lock directories,
collected archives, signatures, and negative failure-injection evidence are
retained until reviewed. Automated pruning must not be introduced without:

- a dry-run inventory;
- minimum age and minimum retained-run controls;
- protection for unsigned, unverified, failed, or referenced evidence;
- an operator-approved deletion manifest;
- separate remote and controller policies.

Disk readiness checks are therefore mandatory until a reviewed retention tool
exists.

## Failure matrix

| Failure | Detection | Result |
|---|---|---|
| Insecure, unowned, or symlinked controller input | Controller readiness | Stop before SSH |
| Unsafe or path-like run ID | Installed operator boundary | Stop before lock or directory creation |
| Unknown or changed SSH host key | SSH strict checking | Stop before remote script |
| Transient preflight transport failure | Bounded attempt history | Retry transport only |
| Deterministic preflight failure | Remote readiness result | Stop without retry |
| Controller overlap | Local lock owner check | Stop |
| Stale controller lock | Same-host boot/PID proof | Explicit recovery only |
| Active remote rollout | Remote lock owner check | Stop node and run |
| Stale remote lock | Same-host boot/PID proof | Archive stale lock, continue |
| Dirty or wrong source | Preflight/provenance/postflight | Stop or fail gate |
| Origin branch moved beyond configured commit | Exact fetched-commit check | Stop before update |
| Divergent local source | Fast-forward ancestry check | Stop before execution |
| Existing run artifacts | Remote preflight | Stop; require new run ID |
| Low controller or remote disk | Readiness checks | Stop |
| Low open-file limit | Remote readiness | Stop |
| Excess clock skew | Controller/remote midpoint comparison | Stop |
| Remote timeout | Bounded SSH execution | Record failure, continue diagnostics |
| Transient SCP failure | Atomic partial transfer | Retry archive only |
| SCP exhaustion or missing archive | Collection and gate | Fail |
| Corrupt or unsafe archive | Release gate | Fail |
| Node omitted or duplicated | Coverage and gate | Fail |
| Abandoned lock after run | Postflight | Fail |
| Source drift after run | Postflight | Fail |
| Controller interruption | Process-group termination and finalization | Exit 130, preserve evidence |
| Local evidence tampering | Operator manifest verification | Reject run |
| Manifest and self-hash rewritten together | OpenSSH attestation verification | Reject unauthenticated evidence |
| Archive reference escapes run directory | Hardened manifest verification | Reject run |

## Remaining physical checks

Code can enforce the contract but cannot prove physical behavior until executed.
The first production run should deliberately exercise at least these failure
injections on a noncritical node:

1. unknown host key;
2. active local lock;
3. provably stale local lock;
4. active remote lock;
5. stale remote lock;
6. insufficient artifact free-space threshold;
7. excessive simulated clock-skew threshold;
8. forced remote timeout;
9. transient SCP failure followed by recovery;
10. exhausted SCP retries;
11. archive tampering before verification;
12. source modification between rollout and postflight;
13. origin branch movement during `--update-code`;
14. controller interruption during remote execution;
15. wrong allowed-signers identity;
16. signature and attestation tampering.

Results should be retained as negative evidence before any runtime admission is
considered.
