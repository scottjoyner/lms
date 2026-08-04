# Fleet operational reliability contract

This document defines the execution reliability boundary for physical fleet
evidence. A benchmark is not trustworthy merely because a model answered
requests. The controller, SSH trust, remote source, locks, storage, clocks,
artifacts, and post-run state must also be trustworthy.

## Required execution sequence

```text
controller readiness
        |
        v
strict SSH trust and complete remote preflight
        |
        v
immutable render
        |
        v
all-node rollout with bounded timeouts
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
```

No later phase may turn an earlier failure into success.

## Controller readiness

Before any SSH connection, `lms-fleet-operator` requires:

- regular configuration and private environment files;
- SHA-256 identities for both inputs;
- `ssh` and `scp` installed;
- a writable workspace;
- at least the configured controller free-space threshold;
- no group or world access to the private environment file by default.

An insecure environment-file mode requires the explicit
`--allow-insecure-env-file` exception and is recorded as a warning. This flag
should be limited to a temporary migration, not routine operation.

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
SHA-256.

Verify later with:

```bash
lms-fleet-operator verify \
  --run-dir /path/to/workspace/<run-id> \
  --require-success
```

Any edited control file, missing archive, changed archive size, or changed
archive hash invalidates the run.

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

## Failure matrix

| Failure | Detection | Result |
|---|---|---|
| Insecure private env permissions | Controller readiness | Stop before SSH |
| Unknown or changed SSH host key | SSH strict checking | Stop before remote script |
| Controller overlap | Local lock owner check | Stop |
| Stale controller lock | Same-host boot/PID proof | Explicit recovery only |
| Active remote rollout | Remote lock owner check | Stop node and run |
| Stale remote lock | Same-host boot/PID proof | Archive stale lock, continue |
| Dirty or wrong source | Preflight/provenance/postflight | Stop or fail gate |
| Existing run artifacts | Remote preflight | Stop; require new run ID |
| Low controller or remote disk | Readiness checks | Stop |
| Low open-file limit | Remote readiness | Stop |
| Excess clock skew | Controller/remote midpoint comparison | Stop |
| Remote timeout | Bounded SSH execution | Record failure, continue diagnostics |
| SCP timeout or missing archive | Collection and gate | Fail |
| Corrupt or unsafe archive | Release gate | Fail |
| Node omitted or duplicated | Coverage and gate | Fail |
| Abandoned lock after run | Postflight | Fail |
| Source drift after run | Postflight | Fail |
| Controller interruption | Process-group termination and finalization | Exit 130, preserve evidence |
| Local evidence tampering | Operator manifest verification | Reject run |

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
9. forced SCP failure;
10. archive tampering before verification;
11. source modification between rollout and postflight;
12. controller interruption during remote execution.

Results should be retained as negative evidence before any runtime admission is
considered.
