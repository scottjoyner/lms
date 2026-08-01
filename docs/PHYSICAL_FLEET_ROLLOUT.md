# Physical fleet rollout

This runbook moves the loadout system from repository validation to physical-node evidence without enabling persistent services or live routing.

## Safety boundary

`lms-fleet-rollout` defaults to observation, model inventory, fair candidate planning, and dry-run launch rendering. It does not install packages, switch branches, load persistent services, modify routers, or admit endpoints.

A real inference sweep requires an exact `NODE_ID=CANDIDATE_ID` argument. Ephemeral llama.cpp candidates and mapped existing endpoints must be loopback-local. Candidate process groups are stopped after evidence capture.

Every remote exit attempts to package evidence, including failed benchmarks and failed selections. The archive manifest records the original `remote_exit_code`; failed archives are diagnostic evidence and cannot be imported as desired state.

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

- `machine_observation.json`;
- `model_inventory.json` with quick planning fingerprints;
- `benchmark_plan.json` with round-robin coverage across model/backend groups;
- rendered launch intent for a limited candidate sample;
- `bundle_manifest.json` with the remote exit code and per-file SHA-256 values.

No candidate performs inference in this stage. A missing llama-server binary is recorded in the rendered intent instead of failing a dry run. Non-llama candidates are rendered with the exact endpoint mapping they will require.

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

For an existing endpoint such as the XDNA2 NPU server, add the chosen candidate ID to that node's `endpoint_map`. The URL must be loopback-local, such as `http://127.0.0.1:1236/v1`; LAN and Tailscale mappings are rejected by the physical executor so the execution manifest can truthfully assert loopback isolation.

## 5. Execute exact candidates

```bash
lms-fleet-rollout run \
  --config fleet-rollout.json \
  --node x1-370 \
  --execute-candidate x1-370=THE_REVIEWED_CANDIDATE_ID \
  --output-dir rollout/x1-sweep-1
```

This stage runs readiness, streaming, concurrency, optional cancellation policy, deterministic task quality, throughput, latency, process memory, system headroom, and crash checks. It then selects an eligible loadout and computes a full content SHA-256 only for the selected model artifact.

A nonzero remote exit still triggers artifact packaging and collection. Preserve that archive for diagnosis, but do not attempt profile import from it.

## 6. Import desired state

Use `tools/import_lms_bundle.py` from `fleet-llm-profiles` with:

- `bundle_manifest.json`;
- the observation;
- benchmark plan;
- execution manifest;
- selected loadout;
- `model_inventory.selected.json`;
- explicit physical-instance, access-path, service-manager, and rollback values.

The importer recomputes every artifact fingerprint, verifies every bundle file hash, requires a successful remote exit and all hard benchmark gates, and always writes `admission.enabled=false`.

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
