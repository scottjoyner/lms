# Reliable exact-loadout qualification run

`lms-loadout-qualification-run` executes the complete local qualification
sequence for one explicitly approved loadout. It replaces manual stitching of
throughput, base Hermes, context-pressure Hermes, binding, and qualification
commands.

It does not discover a candidate, start or stop a runtime, change routing,
restore KV state, or grant admission. The reviewed runtime must already be
running on a loopback endpoint.

## Required inputs

The operator requires:

- one exact `model_loadout_manifest.v1` file;
- one inventory CSV row matching the loadout node, model, and loopback endpoint;
- one reviewed throughput case suite;
- the exact model artifact whose full SHA-256 matches the loadout;
- a clean LMS checkout on an exact branch and 40-character commit;
- a clean Hermes checkout on an exact branch and 40-character commit;
- an already running loopback OpenAI-compatible endpoint exposing the exact
  model ID exactly once;
- a new immutable run ID or an automatically generated one.

The installed package must be sourced from the reviewed LMS checkout. This
prevents an operator from reviewing one commit while executing another installed
wheel or checkout.

## Execute

```bash
lms-loadout-qualification-run run \
  --loadout /reviewed/loadout.json \
  --inventory-csv /reviewed/inventory.csv \
  --cases-file /reviewed/throughput-cases.json \
  --model-artifact /models/exact-model.gguf \
  --endpoint http://127.0.0.1:8080/v1 \
  --api-key-env LMSTUDIO_API_KEY \
  --lms-repo /srv/lms \
  --lms-branch full-auto-reconciliation-20260730 \
  --lms-commit <40-character-commit> \
  --hermes-repo /srv/hermes-agent \
  --hermes-branch <reviewed-branch> \
  --hermes-commit <40-character-commit> \
  --workspace /secure/lms-qualification-runs
```

The default sequence is:

```text
validate exact inputs and model bytes
        |
        v
verify clean pinned LMS and Hermes source
        |
        v
probe exact loopback model and completion
        |
        v
acquire qualification lock and create immutable run
        |
        v
three-trial reliability-first throughput benchmark
        |
        v
three-trial base Hermes suite
        |
        v
three-trial context-pressure Hermes suite
        |
        v
bind throughput to exact loadout
        |
        v
create and independently verify combined qualification
        |
        v
re-probe endpoint, source, and model bytes
        |
        v
write qualification-state and qualification-run-manifest
```

Every phase has a bounded process-group timeout. A failed phase stops later
phases, records the exact failure stage, writes the final state and manifest,
and releases the lock.

## Verify

```bash
lms-loadout-qualification-run verify \
  --run-dir /secure/lms-qualification-runs/<run-id> \
  --require-success
```

Verification recomputes the manifest fingerprint and every local artifact's
size and SHA-256. It rejects duplicate, missing, unsafe, changed, or escaping
artifact paths.

A successful run directory contains at least:

```text
inputs/
qualification-state.json
qualification-run-manifest.json
throughput/
throughput-sidecars/
hermes-base.json
hermes-context.json
hermes-base-work/
hermes-context-work/
throughput-evidence.json
loadout-qualification.json
logs/
```

The exact model file is not copied because it can be very large. Its absolute
path, size, and full SHA-256 are recorded, and its SHA-256 is checked before and
after the complete run.

## Reliability rules

- The endpoint must be loopback-local.
- Inventory must contain exactly one matching endpoint/model row.
- The model artifact must match the loadout before any run directory is created.
- Both repositories must be clean and pinned before and after execution.
- A global qualification lock prevents concurrent local qualification runs.
- A stale lock requires explicit `--recover-stale-lock` and same-host boot/PID
  proof.
- No automatic resume combines evidence from separate execution conditions.
- No workload phase is automatically rerun by the orchestrator.
- The throughput runner applies its own whole-trial retry policy and preserves
  retry rates in reliability evidence.
- Base and context Hermes reports must independently pass their suite gates.
- The combined qualification must independently verify against the original
  exact loadout.
- All outputs remain `admission.admitted=false`.

## Authentication boundary

The qualification-run manifest is tamper-evident but is not currently signed by
`lms-fleet-attest`, which is defined for fleet operator manifests. Until a
qualification-specific signing command is added, retain the run under access
control and include its manifest SHA-256 in the reviewed profile-import record.

## Remaining physical boundary

This command cannot prove runtime startup, shutdown, crash cleanup, GPU reset,
power loss, thermal throttling, or rollback because it intentionally does not
manage the runtime lifecycle. Those controls remain separate physical failure
injections before live admission.