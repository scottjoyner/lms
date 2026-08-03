# Physical fleet rollout

This runbook moves the loadout system from repository validation to physical-node evidence without enabling persistent services or live routing.

## Fleet scope

The canonical controller configuration is now:

```text
examples/fleet-rollout.full-fleet.template.json
examples/fleet-rollout.full-fleet.env.example
examples/fleet-benchmark-census.v1.json
```

The census contains 11 devices. Ten are required benchmark nodes:

1. `destroyer`
2. `raspberrypi`
3. `beelink-ryzen-7-mini-pc`
4. `deathstar-xps-8920`
5. `scott-lenovo-ideapad-330s-15ikb`
6. `scott-optiplex-9030-aio`
7. `scotts-macbook-air`
8. `scotts-macbook-pro-2`
9. `x1-370`
10. `xwing`

`iphone-12-pro-max` remains in the census with an explicit `unsupported` policy. The current runner requires a remotely executable OpenAI-compatible inference runtime plus filesystem evidence collection, which the iOS device does not expose. It is accounted for rather than silently omitted.

The older Tier-1 template is marked `coverage_mode=partial`. It remains useful for proving the physical workflow on `x1-370`, `xwing`, and `scotts-macbook-air`, but it is not the complete fleet benchmark definition.

## Coverage enforcement

`lms-fleet-rollout` validates the rollout against the census before rendering or contacting any node.

For `coverage_mode=full`, validation fails when:

- a `benchmark_required` census node is absent from the configuration;
- a configured rollout node is absent from the census;
- an `unsupported` device is incorrectly configured as a benchmark node;
- the census has duplicate nodes or an invalid policy;
- an unsupported device has no recorded reason.

The validation report contains:

```text
coverage.fleet_device_count
coverage.benchmark_required_count
coverage.configured_benchmark_count
coverage.accounted_device_count
coverage.missing_required_node_ids
coverage.unsupported_node_ids
coverage.coverage_complete
coverage.ready
```

A complete current configuration reports 11 fleet devices, 10 required benchmark nodes, 10 configured benchmark nodes, 11 accounted devices, and one unsupported device.

A full configuration may still be executed one node at a time with `--node`. Coverage is evaluated against the complete configuration before node selection, so staged execution does not weaken the fleet source of truth.

## Safety boundary

`lms-fleet-rollout` defaults to observation, model inventory, fair candidate planning, and dry-run launch rendering. It does not install packages, switch branches unless explicitly requested, load persistent services, modify routers, or admit endpoints.

A real inference sweep requires an exact `NODE_ID=CANDIDATE_ID` argument. Ephemeral llama.cpp candidates and mapped existing endpoints must be loopback-local. Candidate process groups are stopped after evidence capture.

Every remote exit attempts to package evidence, including failed benchmarks and failed selections. The archive manifest records the original `remote_exit_code`; failed archives are diagnostic evidence and cannot be imported as desired state.

`lms-fleet-gate` verifies collected rollout results and archives before candidate review or profile import. It never deploys or admits a runtime.

## 1. Prepare the full-fleet controller configuration

Copy the checked-in templates to private locations:

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

Because the template references the census by a relative path, keep the copied census beside the copied rollout JSON or update `census_file` to the correct private path.

Fill every environment value with the exact SSH target, repository path, Python executable, and model root on each machine. Repository and Python paths must be absolute. Do not commit the populated environment file, private filesystem paths, credentials, private hostnames, Tailscale IP addresses, tailnet DNS names, or device identifiers.

Pin every node to the same reviewed source commit:

```bash
LMS_EXPECTED_COMMIT=THE_REVIEWED_40_CHARACTER_COMMIT
```

Every remote checkout must be on `full-auto-reconciliation-20260730`, at that exact commit, with no tracked, staged, or untracked changes.

## 2. Resolve and validate the complete fleet before SSH

```bash
lms-fleet-rollout validate \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --out rollout/full-fleet-validation.json
```

A successful report must include:

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

The command also rejects unresolved variables, duplicate model roots, relative repository paths, invalid Python paths, whitespace in SSH targets, invalid timeouts, and non-loopback endpoint mappings.

Validation may target one node while retaining full coverage enforcement:

```bash
lms-fleet-rollout validate \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node raspberrypi \
  --out rollout/raspberrypi-validation.json
```

## 3. Render and inspect remote scripts

Render all scripts without contacting any node:

```bash
lms-fleet-rollout render \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --dry-run-limit 4 \
  --output-dir rollout/full-fleet-render
```

Inspect representative scripts from every hardware class:

```bash
less rollout/full-fleet-render/scripts/x1-370.sh
less rollout/full-fleet-render/scripts/deathstar-xps-8920.sh
less rollout/full-fleet-render/scripts/raspberrypi.sh
less rollout/full-fleet-render/scripts/scotts-macbook-pro-2.sh
```

Without `--update-code`, each generated script refuses to run unless the remote repository is already on the configured branch and exact commit. The provenance step also rejects any dirty working tree.

## 4. Collect observation evidence from every benchmark node

The safest first physical pass is one node at a time, preserving the full configuration:

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node destroyer \
  --output-dir rollout/observe-destroyer
```

Repeat for all ten benchmark-required nodes. Once controller and network behavior are proven, all-node observation can be run with failure isolation:

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --continue-on-error \
  --dry-run-limit 4 \
  --output-dir rollout/full-fleet-observe
```

This stage performs no candidate inference. Each node collects:

- `source_control.json`;
- `machine_observation.json`;
- quick `model_inventory.json`;
- a fair model/backend `benchmark_plan.json`;
- rendered candidate intent;
- `bundle_manifest.json` with source, run, file, and bundle fingerprints.

A node with no available models or no usable runtime is expected to produce explicit diagnostic evidence rather than disappear from coverage.

## 5. Gate complete observation coverage

For the combined observation run, require all ten benchmark nodes:

```bash
lms-fleet-gate \
  --rollout-results rollout/full-fleet-observe/rollout_results.json \
  --mode observe \
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

The observation gate verifies remote and collection success, archive integrity, bundle identity, source provenance, node identity, machine observation, model inventory, candidate planning, and non-admission.

A node that cannot yet produce a benchmark plan remains a tracked remediation item. Do not remove it from the census or full configuration to make the gate green.

## 6. Review candidate matrices by hardware class

Use observation evidence to select conservative initial candidates:

- `x1-370`: Vulkan/CPU/available accelerator and XDNA2 candidates;
- `xwing`: accelerated and CPU development-worker candidates;
- `scotts-macbook-air`: Metal and CPU candidates;
- `scotts-macbook-pro-2`: hardware-observation-driven Metal or CPU candidates;
- `deathstar-xps-8920`: supported GPU backend versus CPU;
- `destroyer` and `beelink-ryzen-7-mini-pc`: accelerated backend when observed, otherwise CPU;
- Lenovo and OptiPlex: small-model utility candidates;
- Raspberry Pi: only models and contexts that the observation and planner prove feasible.

Do not force identical model sizes or contexts onto every machine. Fleet completeness means every machine is measured within its viable capability, not that every machine receives the same workload.

For mapped existing endpoints, add only reviewed candidate IDs to a private `endpoint_map`. URLs must remain loopback-local.

## 7. Run reliability-qualified sweeps

Run an exact reviewed candidate for one node:

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/full-fleet.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node raspberrypi \
  --execute-candidate raspberrypi=THE_REVIEWED_CANDIDATE_ID \
  --output-dir rollout/raspberrypi-sweep-1
```

Each selected candidate must pass the reliability contract documented in `docs/BENCHMARK_RELIABILITY.md`, including three complete valid trials, exact sample accounting, raw output integrity, confidence and dispersion gates, bounded retries, and post-trial exact-model health.

A nonzero remote exit still triggers diagnostic packaging. Preserve the archive but do not import it.

## 8. Gate every successful sweep

```bash
lms-fleet-gate \
  --rollout-results rollout/raspberrypi-sweep-1/rollout_results.json \
  --mode sweep \
  --required-node raspberrypi \
  --out rollout/raspberrypi-sweep-1/release-gate.json
```

Sweep mode requires the execution manifest, selected loadout, full selected-model hash, standalone reliability artifact, recomputed reliability fingerprint, matching reliability metrics, loopback isolation, and all hard selection gates.

Full fleet qualification is complete only when every one of the ten required benchmark nodes has either:

- a passing reliability-qualified candidate and sweep gate; or
- an explicit reviewed remediation record explaining why no viable candidate exists yet.

A remediation record is not an admission profile and must not be converted into routable capacity.

## 9. Import reliability-preserving desired state

Use `tools/import_lms_bundle_reliable.py` from `fleet-llm-profiles` with:

- `bundle_manifest.json`;
- `source_control.json`;
- machine observation;
- benchmark plan;
- execution manifest;
- selected loadout;
- `model_inventory.selected.json`;
- the selected candidate's `suite/reliability.json`;
- explicit physical-instance, access-path, service-manager, and rollback values.

The importer recomputes every artifact fingerprint, verifies every bundle file hash, independently reapplies reliability thresholds, and always writes `admission.enabled=false`.

## 10. Promotion gates

A profile remains unadmitted until all of these are recorded independently:

- physical runtime and loaded-model identity;
- full model content fingerprint;
- LAN and Tailscale path behavior without double-counting capacity;
- declared slot count and shared admission proof;
- sustained thermal and memory stability;
- evidence freshness;
- rollback canary.

Only the external live authority may admit or route a runtime. Cross-repository rollout progress is tracked in LMS issue #7.
