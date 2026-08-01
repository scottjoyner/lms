# Physical fleet rollout

This runbook moves the loadout system from repository validation to physical-node evidence without enabling persistent services or live routing.

## Safety boundary

`lms-fleet-rollout` defaults to observation, model inventory, fair candidate planning, and dry-run launch rendering. It does not install packages, switch branches, load persistent services, modify routers, or admit endpoints.

A real inference sweep requires an exact `NODE_ID=CANDIDATE_ID` argument. Ephemeral llama.cpp candidates and mapped existing endpoints must be loopback-local. Candidate process groups are stopped after evidence capture.

Every remote exit attempts to package evidence, including failed benchmarks and failed selections. The archive manifest records the original `remote_exit_code`; failed archives are diagnostic evidence and cannot be imported as desired state.

`lms-fleet-gate` verifies collected rollout results and archives before candidate review or profile import. It never deploys or admits a runtime.

## 1. Prepare the Tier-1 controller configuration

Copy the checked-in templates to private locations:

```bash
mkdir -p ~/.config/lms-fleet rollout
cp examples/fleet-rollout.tier1.template.json ~/.config/lms-fleet/tier1.json
cp examples/fleet-rollout.tier1.env.example ~/.config/lms-fleet/tier1.env
chmod 600 ~/.config/lms-fleet/tier1.env
```

Fill every value in `tier1.env` with the exact SSH target, repository path, Python executable, and model root on each machine. Repository and Python paths must be absolute. Keep each physical machine under its full canonical node ID:

1. `x1-370`
2. `xwing`
3. `scotts-macbook-air`

Do not commit the populated environment file, private filesystem paths, credentials, private hostnames, or unredacted Tailscale exports.

## 2. Resolve and validate configuration before SSH

```bash
lms-fleet-rollout validate \
  --config ~/.config/lms-fleet/tier1.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --out rollout/tier1-validation.json
```

The command rejects unresolved variables, duplicate model roots, relative repository/Python paths, whitespace in SSH targets, and non-loopback endpoint mappings. A successful report sets `ready_for_observation=true` and always retains `admission.admitted=false`.

Validate one node while correcting paths:

```bash
lms-fleet-rollout validate \
  --config ~/.config/lms-fleet/tier1.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --node x1-370 \
  --out rollout/x1-validation.json
```

## 3. Render and inspect the remote script

```bash
lms-fleet-rollout render \
  --config ~/.config/lms-fleet/tier1.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --node x1-370 \
  --output-dir rollout/x1-render

less rollout/x1-render/scripts/x1-370.sh
```

Without `--update-code`, the generated script refuses to run unless the remote repository is already on the configured branch. This avoids silently changing a machine during discovery.

Confirm that the script invokes the hardened `fleet_loadout_entrypoint` and `fleet_bench_entrypoint` modules, binds candidates only to loopback, and contains no persistent service or routing changes.

## 4. Run observation and dry-run planning

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/tier1.json \
  --env-file ~/.config/lms-fleet/tier1.env \
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

Gate the observation archive before reviewing candidates:

```bash
lms-fleet-gate \
  --rollout-results rollout/x1-observe/rollout_results.json \
  --mode observe \
  --required-node x1-370 \
  --out rollout/x1-observe-gate.json
```

The gate verifies the remote and collection return codes, archive paths, bundle manifest, per-file sizes and hashes, node identity, required observation artifacts, model inventory, and plan references.

## 5. Review candidate IDs

Inspect the collected `benchmark_plan.json` and choose a deliberately small first sweep. Start with one model and conservative context/slot combinations.

```bash
python - <<'PY'
import json
p = json.load(open('benchmark_plan.json'))
for item in p['candidates']:
    print(item['candidate_id'], item['backend'], item['model']['id'], item['context_tokens'], item['parallel_slots'])
PY
```

For an existing endpoint such as the XDNA2 NPU server, add the chosen candidate ID to that node's private `endpoint_map`. The URL must be loopback-local, such as `http://127.0.0.1:1236/v1`; LAN and Tailscale mappings are rejected so the execution manifest can truthfully assert loopback isolation.

## 6. Execute exact candidates

```bash
lms-fleet-rollout run \
  --config ~/.config/lms-fleet/tier1.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --node x1-370 \
  --execute-candidate x1-370=THE_REVIEWED_CANDIDATE_ID \
  --output-dir rollout/x1-sweep-1
```

This stage runs readiness, streaming, concurrency, optional cancellation policy, deterministic task quality, throughput, latency, process memory, system headroom, and crash checks. It then selects an eligible loadout and computes a full content SHA-256 only for the selected model artifact.

A nonzero remote exit still triggers artifact packaging and collection. Preserve that archive for diagnosis, but do not attempt profile import from it.

Gate the successful sweep before profile import:

```bash
lms-fleet-gate \
  --rollout-results rollout/x1-sweep-1/rollout_results.json \
  --mode sweep \
  --required-node x1-370 \
  --out rollout/x1-sweep-1-gate.json
```

Sweep mode additionally requires the execution manifest, selected loadout, all hard selection gates, loopback isolation, an actually executed candidate, and exactly one fully fingerprinted selected model. The report remains non-admitted.

## 7. Import desired state

Use `tools/import_lms_bundle.py` from `fleet-llm-profiles` with:

- `bundle_manifest.json`;
- the observation;
- benchmark plan;
- execution manifest;
- selected loadout;
- `model_inventory.selected.json`;
- explicit physical-instance, access-path, service-manager, and rollback values.

The importer recomputes every artifact fingerprint, verifies every bundle file hash, requires a successful remote exit and all hard benchmark gates, and always writes `admission.enabled=false`.

## 8. Promotion gates

A profile remains unadmitted until all of these are recorded independently:

- physical runtime and loaded-model identity;
- full model content fingerprint;
- LAN and Tailscale path behavior without double-counting capacity;
- declared slot count and shared admission proof;
- sustained thermal and memory stability;
- rollback canary;
- observation and benchmark evidence that has not expired.

Only the external live authority may admit or route the runtime. Cross-repository rollout progress is tracked in LMS issue #7.
