# LMS Agent Benchmarking Toolkit

`lms-agent-bench` produces deterministic, non-admitting evidence for local and
fleet inference systems. It separates observation, repeated throughput,
intelligence, exact-loadout qualification, prompt-cache evidence, desired-state
profile import, and live admission.

> The package intentionally does not install a command named `lms`; that name belongs to LM Studio's official CLI.

## Installed commands

```text
lms-agent                          Agent-facing doctor/probe/profile/route CLI
lms-bench                          Reliability-first benchmark CLI
lms-fleet                          Hardware observation, planning, and selection
lms-fleet-bench                    Guarded loopback candidate execution
lms-fleet-models                   Model inventory and selected-model hashing
lms-fleet-rollout                  Low-level census-validated SSH rollout
lms-fleet-operator                 Hardened fleet observation workflow
lms-fleet-attest                   OpenSSH sign/verify fleet evidence
lms-fleet-gate                     Collected-archive release gate
lms-loadout-matrix                 Exact model/runtime loadout matrices
lms-loadout-compare                Separate quality and throughput comparisons
lms-hermes-bench                   Secret-safe Hermes MCP intelligence suites
lms-loadout-qualify                Combined exact-loadout evidence gate
lms-loadout-qualification-run      One-run throughput and Hermes orchestration
lms-loadout-qualification-attest   OpenSSH sign/verify qualification evidence
lms-prompt-cache                   Record-only prompt-prefix/KV registry
```

## Install

```bash
python3 -m pip install -e '.[test]'
```

## Current fleet scope

The operator-confirmed fleet contains ten nodes. Nine are runnable now:

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

`joyner` remains in the census as `benchmark_deferred` because it is powered
off. It must return to `benchmark_required` when it comes online. Raspberry Pi
and iPhone devices are not fleet inference nodes.

## Private setup

```bash
mkdir -p ~/.config/lms-fleet ~/lms-fleet-runs
cp examples/fleet-rollout.full-fleet.template.json \
  ~/.config/lms-fleet/full-fleet.json
cp examples/fleet-benchmark-census.v1.json \
  ~/.config/lms-fleet/fleet-benchmark-census.v1.json
cp examples/fleet-rollout.full-fleet.env.example \
  ~/.config/lms-fleet/full-fleet.env
chmod 600 ~/.config/lms-fleet/full-fleet.env
$EDITOR ~/.config/lms-fleet/full-fleet.env
```

## Reliable fleet observation

```bash
lms-fleet-operator preflight \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs

lms-fleet-operator observe \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --workspace ~/lms-fleet-runs

lms-fleet-operator verify \
  --run-dir ~/lms-fleet-runs/<run-id> \
  --require-success
```

The operator enforces strict SSH host-key verification, complete fleet coverage,
safe controller inputs, exact-commit source, boot/PID locks, disk and clock
readiness, transient-only retries, bounded process groups, atomic archive
collection, all-node postflight, release gating, and verifiable run manifests.

Authenticate production evidence:

```bash
lms-fleet-attest sign \
  --run-dir ~/lms-fleet-runs/<run-id> \
  --key /secure/keys/lms-evidence-signing \
  --require-success

lms-fleet-attest verify \
  --run-dir ~/lms-fleet-runs/<run-id> \
  --allowed-signers /secure/policy/lms_allowed_signers \
  --identity fleet-operator-prod \
  --require-success
```

See:

```text
docs/DETERMINISTIC_FLEET_OPERATOR.md
docs/FLEET_OPERATIONAL_RELIABILITY.md
docs/PHYSICAL_FLEET_ROLLOUT.md
```

## Reliable exact-loadout qualification

A final qualification requires one immutable loadout fingerprint across
repeated throughput, base Hermes intelligence, and context-pressure Hermes
intelligence. The supported execution path is one locked run:

```bash
lms-loadout-qualification-run run \
  --loadout /reviewed/loadout.json \
  --inventory-csv /reviewed/inventory.csv \
  --cases-file /reviewed/throughput-cases.json \
  --model-artifact /models/exact-model.gguf \
  --endpoint http://127.0.0.1:8080/v1 \
  --api-key-env LMSTUDIO_API_KEY \
  --lms-repo /srv/lms \
  --lms-branch <reviewed-branch> \
  --lms-commit <40-character-commit> \
  --hermes-repo /srv/hermes-agent \
  --hermes-branch <reviewed-branch> \
  --hermes-commit <40-character-commit> \
  --workspace /secure/lms-qualification-runs

lms-loadout-qualification-run verify \
  --run-dir /secure/lms-qualification-runs/<run-id> \
  --require-success
```

API-key values are inherited from the named environment variable and are not
placed in Hermes process arguments or logs.

Authenticate the qualification evidence:

```bash
lms-loadout-qualification-attest sign \
  --run-dir /secure/lms-qualification-runs/<run-id> \
  --key /secure/keys/lms-qualification-signing \
  --require-success

lms-loadout-qualification-attest verify \
  --run-dir /secure/lms-qualification-runs/<run-id> \
  --allowed-signers /secure/policy/lms_allowed_signers \
  --identity qualification-operator-prod \
  --require-success
```

See `docs/LOADOUT_QUALIFICATION_OPERATOR.md` and
`docs/EXACT_LOADOUT_QUALIFICATION.md`.

## Prompt-cache evidence

`lms-prompt-cache` stores metadata in SQLite and opaque engine-native payloads
in an atomic content-addressed local store. It is record-only: candidate hits do
not restore KV state, skip tokens, change routing, or claim measured savings.

## Safety boundary

All generated evidence remains non-admitted. Repository tests do not prove
physical host keys, network paths, disk failure, power interruption, thermal
behavior, runtime process cleanup, or rollback. Physical failure injection and
independent live admission remain required.