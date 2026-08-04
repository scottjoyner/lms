# LMS Agent Benchmarking Toolkit

`lms-agent-bench` produces deterministic, non-admitting evidence for local and
fleet inference systems. It separates observation, repeated throughput,
intelligence, exact-loadout qualification, prompt-cache evidence, desired-state
profile import, and live admission.

> The package intentionally does not install a command named `lms`; that name belongs to LM Studio's official CLI.

## Installed commands

```text
lms-agent              Agent-facing doctor/probe/profile/quick/route CLI
lms-bench              Reliability-first benchmark CLI
lms-fleet              Hardware observation, planning, and selection
lms-fleet-bench        Guarded loopback-only candidate execution
lms-fleet-models       Model inventory and selected-model hashing
lms-fleet-rollout      Low-level census-validated SSH rollout
lms-fleet-operator     Hardened preflight/render/observe/postflight/gate workflow
lms-fleet-attest       OpenSSH sign/verify operator evidence
lms-fleet-gate         Collected-archive release gate
lms-loadout-matrix     Exact model/runtime loadout matrices
lms-loadout-compare    Separate quality and throughput comparisons
lms-hermes-bench       Real Hermes MCP agent-loop intelligence suites
lms-loadout-qualify    Combined exact-loadout evidence gate
lms-prompt-cache       Record-only exact prompt-prefix/KV registry
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

Create reviewed private configuration outside the repository:

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

The operator enforces:

- strict SSH host-key verification;
- complete fleet coverage;
- safe controller inputs and run IDs;
- clean exact-commit source;
- controller and remote boot/PID locks;
- disk, file-limit, hostname, model-root, and clock readiness;
- transient-only preflight retries;
- bounded process-group timeouts;
- atomic retried archive collection without rerunning the workload;
- all-node postflight;
- release gating;
- atomic local state and archive-contained run manifests.

Production evidence can be authenticated separately from its storage location:

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

## Exact-loadout qualification

A final qualification requires one immutable loadout fingerprint across
repeated throughput, base Hermes intelligence, and context-pressure Hermes
intelligence:

```bash
lms-loadout-qualify bind-throughput \
  --loadout loadout.json \
  --reliability reliability.json \
  --out throughput-evidence.json

lms-loadout-qualify qualify \
  --loadout loadout.json \
  --throughput throughput-evidence.json \
  --base-hermes hermes-base.json \
  --context-hermes hermes-context.json \
  --out loadout-qualification.json
```

Qualification remains `admission.admitted=false`.

## Prompt-cache evidence

`lms-prompt-cache` stores metadata in SQLite and opaque engine-native payloads
in an atomic content-addressed local store. It is record-only: candidate hits do
not restore KV state, skip tokens, change routing, or claim measured savings.

## Safety boundary

All generated evidence remains non-admitted. Repository tests do not prove
physical host keys, network paths, disk failure, power interruption, thermal
behavior, model-process cleanup, or rollback. Physical failure injection and
independent live admission remain required.