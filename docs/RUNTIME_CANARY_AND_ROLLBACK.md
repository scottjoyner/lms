# Transactional runtime canary, soak, and rollback

`lms-runtime-canary` closes the operational gap between static qualification and
live admission. It executes one reviewed candidate lifecycle as a transaction,
requires a sustained burn-in, restores the previous runtime, verifies that
rollback, and writes tamper-evident evidence.

The command is deliberately non-admitting. A successful run ends with the prior
runtime restored; it never leaves the candidate running, changes routing, or
enables a profile.

## Execution sequence

```text
validate immutable plan and command binaries
        |
        v
acquire local canary lock
        |
        v
snapshot previous runtime
        |
        v
start reviewed candidate
        |
        v
verify candidate health
        |
        v
run exact-loadout qualification
        |
        v
run sustained soak probes
        |
        v
stop candidate
        |
        v
restore previous runtime
        |
        v
verify restored runtime health
        |
        v
seal run state and manifest
```

Any failure after candidate startup enters the recovery path immediately:

```text
best-effort candidate stop -> rollback -> rollback health verification
```

The run remains failed even when rollback succeeds. The evidence distinguishes
`success`, `rollback_attempted`, and `rollback_succeeded` so a failed experiment
cannot be mistaken for a qualified candidate.

## Plan safety

Start from:

```text
examples/runtime-canary.plan.example.json
```

The installed entrypoint requires:

- a nonsymlinked plan that is not group/world writable;
- an absolute, existing working directory;
- an exact lowercase SHA-256 loadout fingerprint;
- absolute executable paths for every lifecycle command;
- no `sh -c`, `bash -c`, PowerShell, or other shell interpreter;
- no API keys, tokens, passwords, or secrets in command arguments;
- secret values inherited only through reviewed `environment_names`;
- bounded timeouts for every command;
- one global canary lock with explicit stale-lock recovery.

Every command record includes the resolved executable path and full executable
SHA-256. Command stdout and stderr are stored as files and referenced by hash,
not copied into the run summary.

The lifecycle wrappers are operator-owned programs. They should use direct
`exec`/system APIs and fixed arguments rather than accepting arbitrary shell
fragments.

## Soak probe contract

`soak_probe` must print one JSON object as its final stdout line:

```json
{
  "ok": true,
  "latency_seconds": 0.42,
  "rss_bytes": 12884901888,
  "temperature_c": 72.5,
  "tps": 24.1,
  "ttft_seconds": 0.18
}
```

Only those metric fields are retained. Probe output is limited to one MiB.

The soak gate can enforce:

- minimum sample count;
- minimum success rate;
- maximum consecutive failures;
- maximum p95 latency;
- maximum resident-memory growth;
- minimum terminal-to-baseline TPS ratio;
- maximum observed temperature.

A missing latency sample fails the p95 gate. When a temperature ceiling is
configured, missing temperature evidence fails the temperature gate. This keeps
hardware telemetry requirements explicit rather than silently optional.

## Run and verify

```bash
chmod 600 /secure/plans/x1-370-canary.json

lms-runtime-canary validate \
  --plan /secure/plans/x1-370-canary.json

lms-runtime-canary run \
  --plan /secure/plans/x1-370-canary.json \
  --workspace /secure/lms-canary-runs \
  --run-id x1-370-first-canary

lms-runtime-canary verify \
  --run-dir /secure/lms-canary-runs/x1-370-first-canary \
  --require-success
```

A run directory contains:

```text
plan.normalized.json
runtime-canary-state.json
runtime-canary-manifest.json
soak-samples.jsonl
logs/
```

The manifest records the exact plan/loadout fingerprints and every artifact's
size and SHA-256. Verification rejects changed, missing, duplicate, unsafe,
escaping, or symlinked artifacts.

## Authenticate the evidence

```bash
lms-runtime-canary-attest sign \
  --run-dir /secure/lms-canary-runs/x1-370-first-canary \
  --key /secure/keys/lms-canary-signing \
  --require-success

lms-runtime-canary-attest verify \
  --run-dir /secure/lms-canary-runs/x1-370-first-canary \
  --allowed-signers /secure/policy/lms_allowed_signers \
  --identity runtime-canary-prod \
  --require-success
```

The OpenSSH signature binds the run ID, canary ID, loadout fingerprint, plan
fingerprint, success state, and verified rollback state to the exact manifest
bytes.

## Production admission boundary

A runtime should not become admission-eligible until an independent consumer has
verified all of these roots:

1. signed fleet observation evidence;
2. signed exact-loadout qualification evidence;
3. signed runtime-canary evidence with `rollback_succeeded=true`;
4. current live identity, reachability, capacity, freshness, and rollback checks.

This command proves the candidate can be introduced and removed safely on one
node under the reviewed lifecycle wrappers. It does not prove power-loss
recovery, kernel/GPU reset, network partition behavior, or multi-node rollout.
Those remain explicit physical failure-injection drills.
