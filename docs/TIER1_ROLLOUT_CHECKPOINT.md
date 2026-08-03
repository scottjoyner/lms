# Tier-1 partial rollout checkpoint

This checkpoint covers only `x1-370`, `xwing`, and `scotts-macbook-air`. It is a deliberately partial proving tranche for the physical evidence workflow, not the fleet benchmark definition.

The canonical complete-fleet files are:

```text
examples/fleet-benchmark-census.v1.json
examples/fleet-rollout.full-fleet.template.json
examples/fleet-rollout.full-fleet.env.example
docs/PHYSICAL_FLEET_ROLLOUT.md
```

The Tier-1 template is marked `coverage_mode=partial`. Validation reports `coverage_complete=false` and lists the seven required remote-runner nodes deferred from this tranche. A successful Tier-1 run must never be represented as complete fleet qualification.

## Tier-1 roles

| Node | Initial role | First-pass scope |
|---|---|---|
| `x1-370` | heavy local reasoning | Linux CPU/Vulkan/available accelerator candidates, LM Studio, and XDNA2 endpoint observation |
| `xwing` | development worker | Linux CPU and available accelerated candidates |
| `scotts-macbook-air` | fast small-model worker | Metal and CPU candidates with latency-sensitive workloads |

## Prepare the controller checkout

```bash
git checkout full-auto-reconciliation-20260730
git pull --ff-only origin full-auto-reconciliation-20260730
python3 -m pip install -e '.[test]'
```

Pin every remote node to the exact reviewed 40-character commit and require a completely clean working tree.

## Create the private Tier-1 environment

```bash
mkdir -p ~/.config/lms-fleet
cp examples/fleet-rollout.tier1.env.example ~/.config/lms-fleet/tier1.env
chmod 600 ~/.config/lms-fleet/tier1.env
$EDITOR ~/.config/lms-fleet/tier1.env
```

Do not commit private SSH targets, hostnames, Tailscale addresses, repository paths, Python environments, model directories, or credentials.

## Validate the partial tranche

```bash
lms-fleet-rollout validate \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --out rollout/tier1-validation.json
```

A resolved Tier-1 report may set `ready_for_observation=true`, but it must also show:

```text
coverage.coverage_mode=partial
coverage.coverage_complete=false
coverage.configured_benchmark_count=3
coverage.benchmark_required_count=10
coverage.adapter_required_count=1
coverage.adapter_required_node_ids=[iphone-12-pro-max]
```

This prevents the three-node tranche from being confused with full configuration coverage or full benchmark qualification.

## Render and inspect scripts

```bash
lms-fleet-rollout render \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --dry-run-limit 6 \
  --output-dir rollout/tier1-render
```

Inspect all three scripts before physical execution. They must preserve exact source provenance, loopback-only candidate execution, per-node locking, bounded execution, and failure-safe artifact packaging.

## Collect observation evidence

```bash
lms-fleet-rollout run \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --continue-on-error \
  --dry-run-limit 6 \
  --output-dir rollout/tier1-observe
```

This performs observation, quick model inventory, candidate planning, and dry-run rendering. It does not perform candidate inference.

Gate the three collected archives:

```bash
lms-fleet-gate \
  --mode observe \
  --rollout-results rollout/tier1-observe/rollout_results.json \
  --required-node x1-370 \
  --required-node xwing \
  --required-node scotts-macbook-air \
  --out rollout/tier1-observe/release-gate.json
```

## Execute exact reviewed candidates

Run one conservative candidate at a time:

```bash
lms-fleet-rollout run \
  --config /path/to/private-tier1-sweep.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --node x1-370 \
  --execute-candidate x1-370=REVIEWED_CANDIDATE_ID \
  --output-dir rollout/x1-370-sweep-1
```

Every candidate remains subject to the complete reliability contract: strict protocol, exact sample accounting, raw-output integrity, three valid trials, confidence and dispersion gates, bounded retries, and post-trial exact-model health.

## Exit from Tier-1 into full-fleet qualification

Tier-1 is complete only when the three-node evidence flow is understood and reproducible. The next action is not merge or admission. It is to populate and validate the full-fleet configuration, then collect evidence for:

- `destroyer`
- `raspberrypi`
- `beelink-ryzen-7-mini-pc`
- `deathstar-xps-8920`
- `scott-lenovo-ideapad-330s-15ikb`
- `scott-optiplex-9030-aio`
- `scotts-macbook-pro-2`

The iPhone remains census-accounted as `adapter_required`; issue #9 must produce a physical mobile reliability artifact before `benchmark_interface_complete` can become true.

No runtime may be admitted or routed merely because the Tier-1 tranche passed. Full remote-runner coverage, mobile adapter qualification, profile import, live identity, path behavior, capacity, freshness, and rollback remain separate gates.
