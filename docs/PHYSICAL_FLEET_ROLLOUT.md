# Physical fleet rollout

This runbook moves the loadout system from repository validation to physical-node evidence without enabling persistent services or live routing.

## Safety boundary

`lms-fleet-rollout` defaults to observation, model inventory, plan generation, and dry-run launch rendering. It does not install packages, switch branches, load persistent services, modify routers, or admit endpoints.

A real inference sweep requires an exact `NODE_ID=CANDIDATE_ID` argument. The remote candidate remains ephemeral and loopback-only, and the process group is stopped after evidence capture.

## 1. Prepare a node configuration

Copy `examples/fleet-rollout.v1.example.json` and replace every placeholder with the exact repository and model paths on that machine. Keep each physical machine under its full Tailscale device name.

Required first-wave nodes:

1. `x1-370`
2. `xwing`
3. `scotts-macbook-air`

Do not assume that model paths, Python environments, or repository locations are identical across those machines.

## 2. Render and inspect the remote script

```bash
lms-fleet-rollout render \
  --config fleet-rollout.json \
  --node x1-370 \
  --output-dir rollout/x1-render

less rollout/x1-render/scripts/x1-370.sh
```

Without `--update-code`, the generated script refuses to run unless the remote repository is already on the configured branch. This avoids silently changing a machine during discovery.

## 3. Run observation and dry-run planning

```bash
lms-fleet-rollout run \
  --config fleet-rollout.json \
  --node x1-370 \
  --output-dir rollout/x1-observe
```

The collected archive contains:

- `machine_observation.json`
- `model_inventory.json` with quick planning fingerprints
- `benchmark_plan.json`
- rendered launch commands for a limited candidate sample
- `bundle_manifest.json` with per-file SHA-256 values

No candidate performs inference in this stage.

## 4. Review candidate IDs

Inspect the plan and choose a deliberately small first sweep. Start with one model and conservative context/slot combinations.

```bash
python - <<'PY'
import json
p = json.load(open('benchmark_plan.json'))
for item in p['candidates']:
    print(item['candidate_id'], item['backend'], item['model']['id'], item['context_tokens'], item['parallel_slots'])
PY
```

For an existing endpoint such as the XDNA2 NPU server, add the chosen candidate ID to that node's `endpoint_map` in the rollout configuration. Existing endpoint URLs are never guessed.

## 5. Execute exact candidates

```bash
lms-fleet-rollout run \
  --config fleet-rollout.json \
  --node x1-370 \
  --execute-candidate x1-370=THE_REVIEWED_CANDIDATE_ID \
  --output-dir rollout/x1-sweep-1
```

This stage runs readiness, streaming, concurrency, optional cancellation policy, deterministic task quality, throughput, latency, process memory, system headroom, and crash checks. It then selects an eligible loadout and computes a full content SHA-256 only for the selected model artifact.

## 6. Import desired state

Use `tools/import_lms_bundle.py` from `fleet-llm-profiles` with:

- the observation;
- benchmark plan;
- execution manifest;
- selected loadout;
- `model_inventory.selected.json`;
- explicit physical-instance, access-path, service-manager, and rollback values.

The importer verifies the artifact chain and always writes `admission.enabled=false`.

## 7. Promotion gates

A profile remains draft until all of these are recorded independently:

- physical runtime and loaded-model identity;
- full model content fingerprint;
- LAN and Tailscale path behavior without double-counting capacity;
- declared slot count and shared admission proof;
- sustained thermal and memory stability;
- rollback canary;
- observation and benchmark evidence that has not expired.

Only the external live authority may admit or route the runtime.
