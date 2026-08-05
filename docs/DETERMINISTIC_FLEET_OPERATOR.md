# Deterministic Fleet Operator

The supported physical fleet entrypoint is:

```bash
lms-fleet-operator
```

It is a fail-closed controller for complete-fleet observation evidence. It does
not invent candidates, enable live admission, change routing, or restore KV
state.

The complete operational contract—including SSH trust, exact-commit updates,
controller and remote locks, disk and clock checks, bounded retries, atomic
artifact transport, postflight, run-manifest verification, and OpenSSH evidence
attestation—is defined in:

```text
docs/FLEET_OPERATIONAL_RELIABILITY.md
```

## Required sequence

```bash
lms-fleet-operator preflight \
  --config /secure/fleet-rollout.json \
  --env-file /secure/fleet-rollout.env \
  --workspace /secure/lms-fleet-runs

lms-fleet-operator observe \
  --config /secure/fleet-rollout.json \
  --env-file /secure/fleet-rollout.env \
  --workspace /secure/lms-fleet-runs

lms-fleet-operator verify \
  --run-dir /secure/lms-fleet-runs/<run-id> \
  --require-success
```

Production evidence should then be authenticated with a key whose private half
is stored separately from the run directory:

```bash
lms-fleet-attest sign \
  --run-dir /secure/lms-fleet-runs/<run-id> \
  --key /secure/keys/lms-evidence-signing \
  --require-success

lms-fleet-attest verify \
  --run-dir /secure/lms-fleet-runs/<run-id> \
  --allowed-signers /secure/policy/lms_allowed_signers \
  --identity fleet-operator-prod \
  --require-success
```

## Host-key bootstrap

Normal operation requires an existing verified known-host entry. First contact
is explicit:

```bash
lms-fleet-operator preflight \
  --accept-new-host-keys \
  --config /secure/fleet-rollout.json \
  --env-file /secure/fleet-rollout.env \
  --workspace /secure/lms-fleet-runs
```

Review the resulting host key out of band, then return to strict known-host
operation. Host-key verification cannot be disabled.

## Failure behavior

- Any incomplete fleet coverage fails before rollout.
- Any deterministic preflight failure stops execution without retry.
- Transient SSH preflight failures may retry with recorded attempts.
- An active or ambiguous lock fails closed.
- A stale controller lock requires explicit safe recovery.
- A stale remote lock is archived only when same-host boot/PID evidence proves
  it stale.
- Remote code update is pinned to the configured exact commit.
- The remote workload is never automatically rerun.
- Immutable SCP collection may retry through partial files and atomic promotion.
- Postflight and the release gate run even after partial rollout failure when
  diagnostic artifacts are available.
- Success requires rollout, postflight, and release gate success.
- Interrupted and failed runs remain non-admitted and are retained for review.

## Current fleet

The rollout contract contains nine runnable nodes:

```text
destroyer
beelink-ryzen-7-mini-pc
deathstar-xps-8920
scott-lenovo-ideapad-330s-15ikb
scott-optiplex-9030-aio
scotts-macbook-air
scotts-macbook-pro-2
x1-370
xwing
```

`joyner` remains `benchmark_deferred` while powered off. It must not receive a
profile based on stale evidence.

## Candidate sweeps

Observation does not execute inference candidates. Candidate sweeps remain a
separate reviewed phase because candidate IDs must come from collected
`benchmark_plan.json` artifacts and be explicitly approved. The operator must
not invent or substitute candidate IDs.

## Physical boundary

Repository tests exercise failure behavior synthetically and with real local
OpenSSH signatures. They do not prove fleet network reachability, real host
keys, remote disk behavior, power loss, thermal stability, process cleanup, or
rollback on the physical nodes. Those failure injections must be performed on a
noncritical node before live admission.