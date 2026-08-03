# LMS Agent Benchmarking Toolkit

LMS is an agent-facing toolkit for profiling and benchmarking local or Tailscale-reachable LM Studio and OpenAI-compatible inference nodes. It produces deterministic benchmark evidence, hardware-aware model planning, guarded physical candidate execution, census-complete fleet coverage, and non-admitted runtime recommendations.

> **Command ownership:** this package intentionally does **not** install a command named `lms`. That command belongs to LM Studio's official CLI and is used for Link-aware observation. The agent-facing command is `lms-agent`.

## Installed commands

```text
lms-agent             Main agent-facing doctor/probe/profile/quick/route CLI
lms-bench             Manifest-driven benchmark CLI
lmsbench               Alias for lms-bench
lms-bench-endpoints    Endpoint registry and discovery
lmstudio-bridge        LM Studio CLI bridge
lms-fleet              Hardware observation, fair planning, and selection
lms-fleet-models       Local model inventory and selected-model hashing
lms-fleet-bench        Guarded loopback-only candidate execution
lms-fleet-rollout      Census-validated SSH rollout rendering and execution
lms-fleet-gate         Collected-archive observation and sweep release gate
```

The canonical implementation lives in `src/lms_agent_bench/`. Root compatibility modules re-export package implementations where historical imports still require them.

## Install

```bash
python3 -m pip install -e '.[test]'
```

Confirm the command surface:

```bash
lms-agent --help
lms-fleet --help
lms-fleet-bench --help
lms-fleet-models --help
lms-fleet-rollout --help
lms-fleet-gate --help
```

## Complete fleet benchmark scope

The canonical controller files are:

```text
examples/fleet-benchmark-census.v1.json
examples/fleet-rollout.full-fleet.template.json
examples/fleet-rollout.full-fleet.env.example
docs/PHYSICAL_FLEET_ROLLOUT.md
```

The current census accounts for 11 devices. Ten are required benchmark nodes:

- `destroyer`
- `raspberrypi`
- `beelink-ryzen-7-mini-pc`
- `deathstar-xps-8920`
- `scott-lenovo-ideapad-330s-15ikb`
- `scott-optiplex-9030-aio`
- `scotts-macbook-air`
- `scotts-macbook-pro-2`
- `x1-370`
- `xwing`

`iphone-12-pro-max` remains census-accounted with an explicit unsupported policy because the current runner requires a remotely executable OpenAI-compatible runtime and filesystem evidence collection.

The older Tier-1 template contains only `x1-370`, `xwing`, and `scotts-macbook-air`. It is marked `coverage_mode=partial` and must not be treated as complete fleet qualification.

## Validate full fleet coverage

Prepare a private environment file outside the repository, then run:

```bash
lms-fleet-rollout validate \
  --config examples/fleet-rollout.full-fleet.template.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --out rollout/full-fleet-validation.json
```

A complete report contains:

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

Full coverage validation fails when a required benchmark node disappears, an unknown node is added, or an unsupported device is incorrectly configured as a benchmark target.

A complete configuration may still be executed one node at a time with `--node`; coverage is evaluated before node selection.

## Physical observation and sweep flow

Render without contacting a node:

```bash
lms-fleet-rollout render \
  --config examples/fleet-rollout.full-fleet.template.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --all-nodes \
  --output-dir rollout/full-fleet-render
```

Collect observation evidence from one node:

```bash
lms-fleet-rollout run \
  --config examples/fleet-rollout.full-fleet.template.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node raspberrypi \
  --output-dir rollout/observe-raspberrypi
```

Observation mode performs no candidate inference. It collects source provenance, machine observation, model inventory, a fair benchmark plan, rendered candidate intent, and a cryptographically linked archive.

Real inference requires an exact reviewed candidate ID:

```bash
lms-fleet-rollout run \
  --config examples/fleet-rollout.full-fleet.template.json \
  --env-file ~/.config/lms-fleet/full-fleet.env \
  --node raspberrypi \
  --execute-candidate raspberrypi=REVIEWED_CANDIDATE_ID \
  --output-dir rollout/raspberrypi-sweep-1
```

Physical endpoint mappings must remain loopback-local.

## Reliability-first benchmark contract

A selectable physical candidate must prove:

- exact planned model identity;
- cold-load and stabilized warmup canaries;
- strict OpenAI-compatible streaming completion;
- complete endpoint/case/repeat sample accounting;
- one unique non-empty raw output per successful sample;
- three valid complete trials by default;
- whole-trial retries only;
- request success of at least 98%;
- evaluator success of at least 90%;
- a 95% Wilson request-success lower bound of at least 0.80;
- TPS trial coefficient of variation no greater than 0.20;
- TTFT trial coefficient of variation no greater than 0.35;
- TPS and TTFT relative MAD no greater than 0.25;
- retry rate no greater than 0.25;
- post-trial exact-model health;
- at least 10% system-memory headroom;
- no observed crash.

Valid trial manifests retain aggregate artifact hashes, raw sample size/SHA-256 evidence, and a canonical manifest fingerprint. Resume verifies all retained evidence and only reuses a clean first-attempt success, preventing hidden retry history.

See `docs/BENCHMARK_RELIABILITY.md`.

## Release gates

Observation archives:

```bash
lms-fleet-gate \
  --mode observe \
  --rollout-results rollout/observe-raspberrypi/rollout_results.json \
  --required-node raspberrypi \
  --out rollout/observe-raspberrypi/release-gate.json
```

Sweep archives:

```bash
lms-fleet-gate \
  --mode sweep \
  --rollout-results rollout/raspberrypi-sweep-1/rollout_results.json \
  --required-node raspberrypi \
  --out rollout/raspberrypi-sweep-1/release-gate.json
```

Sweep mode verifies source provenance, archive and bundle integrity, loopback-only execution, selected-model full SHA-256, the standalone reliability fingerprint, matching selected reliability metrics, and every hard selection gate.

## Agent benchmark quick start

```bash
lms-agent doctor
lms-agent probe
lms-agent quick
lms-agent show latest --task coding
lms-agent route latest --task structured_output
```

The default endpoint is `http://127.0.0.1:1234/v1`, unless `LMS_BASE_URL` or `LMSTUDIO_BASE_URL` is set.

A normal agent benchmark creates a self-contained run directory with endpoint probes, machine profile, inventory, raw benchmark results, summaries, recommendations, routing rules, and sidecars.

## Product objective

The toolkit should answer:

1. Which devices are in the fleet, and which are benchmark-eligible?
2. What hardware and acceleration backends are currently available on every eligible machine?
3. What local models and quantizations are present?
4. Which model/loadout combinations fit safely?
5. What can each combination do reliably at measured latency and throughput?
6. Where are its context, memory, stability, and instruction-following limits?
7. Which reviewed desired loadout should proceed to external identity, path, capacity, freshness, and rollback gates?

## Safety and authority rules

- Do not silently remove weak, offline, model-less, or misconfigured machines from fleet coverage.
- Record unsupported devices and remediation states explicitly.
- Do not require cloud services or internet access during benchmark execution.
- Do not leak prompts, credentials, model output, private paths, device IDs, or private network identities outside retained evidence.
- Keep physical candidate endpoints loopback-local.
- Treat LAN, Tailscale, loopback, and LM Studio Link URLs as access paths to one physical capacity record.
- Execute only explicitly reviewed candidate IDs.
- Preserve failed archives for diagnosis, but never import or admit them.
- Treat deterministic evaluation as authoritative for hard gates.
- Make every recommendation traceable to immutable benchmark evidence.
- Keep profile output and selection artifacts non-admitted until the external live authority verifies runtime identity, model SHA-256, shared capacity, path behavior, sustained stability, freshness, and rollback.
