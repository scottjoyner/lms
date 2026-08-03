# Physical fleet rollout

This runbook moves the loadout system from repository validation to physical-node evidence without enabling persistent services or live routing.

## Fleet scope

The canonical controller files are:

```text
examples/fleet-benchmark-census.v1.json
examples/fleet-rollout.full-fleet.template.json
examples/fleet-rollout.full-fleet.env.example
```

The census accounts for all 11 current devices.

Ten devices use the existing SSH/filesystem benchmark runner:

```text
destroyer
raspberrypi
beelink-ryzen-7-mini-pc
deathstar-xps-8920
scott-lenovo-ideapad-330s-15ikb
scott-optiplex-9030-aio
scotts-macbook-air
scotts-macbook-pro-2
x1-370
xwing
```

`iphone-12-pro-max` remains in benchmark scope as `adapter_required`. The current controller cannot drive an iOS-local inference runtime through SSH and filesystem evidence collection. Issue #9 tracks the signed mobile benchmark agent or equivalent adapter required before mobile qualification can finish.

The three-node Tier-1 template is explicitly `coverage_mode=partial`. It is useful for proving the workflow, but it is not the fleet benchmark definition.

## Coverage semantics

`lms-fleet-rollout` checks census coverage before rendering or contacting a node.

Full mode rejects:

- a missing `benchmark_required` node;
- a rollout node absent from the census;
- an adapter-required or unsupported device configured as an SSH rollout node;
- duplicate census identities;
- invalid policies;
- adapter-required or unsupported policies without reasons.

A resolved current report includes:

```text
coverage.ready=true
coverage.coverage_complete=true
coverage.benchmark_interface_complete=false
coverage.fleet_device_count=11
coverage.benchmark_required_count=10
coverage.adapter_required_count=1
coverage.configured_benchmark_count=10
coverage.accounted_device_count=11
coverage.adapter_required_node_ids=[iphone-12-pro-max]
```

`coverage_complete=true` means the SSH rollout fully matches the census and every non-SSH node is explicitly accounted for. It does not claim that every device has completed a physical benchmark. `benchmark_interface_complete=false` preserves the mobile adapter blocker.

A complete configuration may still execute one node at a time with `--node`; coverage is checked against the complete configuration before node selection.

## Safety boundary

The rollout defaults to observation, model inventory, fair candidate planning, and dry-run launch rendering. It does not install packages, admit endpoints, modify routers, or enable persistent services.

Real inference requires an exact reviewed `NODE_ID=CANDIDATE_ID`. Ephemeral candidates and mapped endpoints must remain loopback-local. Every remote exit attempts to package diagnostic evidence, but bundles with nonzero `remote_exit_code` are not promotable.

## 1. Prepare private configuration

```bash
mkdir -p ~/.config/lms-fleet rollout
cp examples/fleet-rollout.full-fleet.template.json \
  ~/.config/lms-fleet/full-fleet.json
cp examples/fleet-rollout.full-fleet.env.example \
  ~/.config/lms-fleet/full-fleet.env
cp examples/fleet-benchmark-census.v1.json \
  ~/.config/lms-fleet/fleet-benchmark-census.v1.json
chmod 600 ~/.config/lms-fleet/full-fleet.env
```

Keep the copied census beside the copied rollout JSON or update `census_file` to its private path.

Fill every SSH target, repository path, Python executable, and model root. Repository, Python, and model paths must be absolute on the remote machine.

Do not commit private usernames, hostnames, Tailscale addresses, device IDs, account emails, filesystem paths, or credentials.

Pin all remote checkouts to the same reviewed commit:

```bash
LMS_EXPECTED_COMMIT=THE_REVIEWED_40_CHARACTER_COMMIT
```

Each checkout must use the configured branch and exact commit with no tracked, staged, or untracked changes.

## 2. Validate before SSH

```bash
lms-fleet-rollout validate \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --out rollout/full-fleet-validation.json
```

Require:

```text
ready_for_observation=true
coverage.ready=true
coverage.coverage_complete=true
coverage.fleet_device_count=11
coverage.benchmark_required_count=10
coverage.configured_benchmark_count=10
coverage.accounted_device_count=11
admission.admitted=false
```

Also confirm that `benchmark_interface_complete=false` is explained only by `iphone-12-pro-max` in `adapter_required_node_ids`.

The supplied fleet export reports key-expiry dates in the past for `destroyer` and `raspberrypi`. Renew or re-authenticate those identities, or independently prove current connectivity, before physical rollout. Do not remove them from the census to bypass the problem.

## 3. Render and inspect

```bash
lms-fleet-rollout render \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --dry-run-limit 4 \
  --output-dir rollout/full-fleet-render
```

Inspect representative scripts from each hardware class and verify exact source provenance, loopback execution, bounded commands, locking, and failure-safe packaging.

## 4. Collect observations

Start one node at a time while retaining the complete configuration:

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node destroyer \
  --output-dir rollout/observe-destroyer
```

Repeat for all ten remote-runner nodes. Once controller and network behavior are proven, an all-node observation may use `--continue-on-error`.

Observation mode performs no candidate inference. It collects:

- `source_control.json`;
- `machine_observation.json`;
- quick model inventory;
- fair benchmark plan;
- rendered candidate intent;
- cryptographically linked bundle metadata.

A machine with no viable model or runtime must produce explicit diagnostic or remediation evidence. It must not disappear from the census.

## 5. Gate observations

For a combined full-fleet observation, require every remote-runner node with repeated `--required-node` arguments:

```bash
lms-fleet-gate \
  --mode observe \
  --rollout-results rollout/full-fleet-observe/rollout_results.json \
  --required-node destroyer \
  --required-node raspberrypi \
  --required-node beelink-ryzen-7-mini-pc \
  --required-node deathstar-xps-8920 \
  --required-node scott-lenovo-ideapad-330s-15ikb \
  --required-node scott-optiplex-9030-aio \
  --required-node scotts-macbook-air \
  --required-node scotts-macbook-pro-2 \
  --required-node x1-370 \
  --required-node xwing \
  --out rollout/full-fleet-observe/release-gate.json
```

The gate verifies archive integrity, source provenance, node identity, observations, inventory, plans, bundle identity, and non-admission.

## 6. Review candidates by capability

Do not force identical models or contexts across heterogeneous machines.

- `x1-370`: Vulkan, CPU, available accelerator, and XDNA2 paths.
- `xwing`: available accelerated backend and CPU comparison.
- MacBook Air and MacBook Pro: Metal and CPU as observations permit.
- Deathstar: supported GPU backend versus CPU.
- Destroyer and Beelink: accelerated backend when observed, otherwise CPU.
- Lenovo and OptiPlex: small-model and utility workloads.
- Raspberry Pi: conservative small-model, context, and concurrency qualification.
- iPhone: mobile-adapter qualification under issue #9; never substitute simulator evidence.

## 7. Run exact reliability-qualified sweeps

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node raspberrypi \
  --execute-candidate raspberrypi=THE_REVIEWED_CANDIDATE_ID \
  --output-dir rollout/raspberrypi-sweep-1
```

Each candidate must pass strict protocol validation, complete sample accounting, raw-output integrity, three valid trials, confidence and dispersion gates, bounded retries, memory headroom, concurrency, stability, and post-trial exact-model health.

## 8. Gate each sweep

```bash
lms-fleet-gate \
  --mode sweep \
  --rollout-results rollout/raspberrypi-sweep-1/rollout_results.json \
  --required-node raspberrypi \
  --out rollout/raspberrypi-sweep-1/release-gate.json
```

Sweep mode requires the execution manifest, selected loadout, selected-model full hash, standalone reliability artifact, matching reliability fields, loopback isolation, and all hard selection gates.

## 9. Completion criteria

Each remote-runner node must produce either:

1. a passing reliability-qualified sweep and release gate; or
2. a reviewed remediation record explaining why no viable candidate exists yet.

The iPhone must produce a passing physical mobile-adapter reliability artifact. A remediation record or adapter blocker is diagnostic state, not routable capacity.

## 10. Import desired state

Use `tools/import_lms_bundle_reliable.py` from `fleet-llm-profiles` with source, observation, plan, execution, selection, selected inventory, bundle manifest, and selected reliability artifacts.

The importer recomputes artifact fingerprints, verifies bundle files, reapplies reliability thresholds, and always writes `admission.enabled=false`.

## 11. Live promotion remains external

A profile remains unadmitted until an external authority independently proves current runtime/model identity, model content fingerprint, approved path behavior, shared capacity, evidence freshness, sustained stability, and rollback.
