# Tier-1 physical rollout checkpoint

This checkpoint advances `x1-370`, `xwing`, and `scotts-macbook-air` from repository validation to physical evidence. It remains observation-first, candidate-explicit, loopback-only, and non-admitted.

## Tier-1 roles

| Node | Tailscale identity | Initial role | First-pass scope |
|---|---|---|---|
| `x1-370` | `x1-370.tailcb8954.ts.net` / `100.64.43.123` | heavy local reasoning | Linux CPU/Vulkan/available accelerator candidates, LM Studio, and XDNA2 endpoint observation |
| `xwing` | `xwing.tailcb8954.ts.net` / `100.108.99.47` | default development worker | Linux CPU/GPU candidates and the currently available large local models |
| `scotts-macbook-air` | `scotts-macbook-air.tailcb8954.ts.net` / `100.85.64.117` | fast small-model scout | Metal and CPU candidates for short low-latency work; do not schedule heavy fleet jobs initially |

The template does not guess SSH usernames, repository locations, Python environments, or model directories. Those values must be supplied explicitly.

## 1. Prepare the control checkout

```bash
git checkout full-auto-reconciliation-20260730
git pull --ff-only origin full-auto-reconciliation-20260730
python3 -m pip install -e '.[test]'
```

The installed rollout command is `lms-fleet-rollout`. Generated remote scripts invoke the hardened `fleet_loadout_entrypoint` and `fleet_bench_entrypoint` modules; they do not bypass fair planning, dry-run behavior, or loopback enforcement.

## 2. Create the private environment file

```bash
mkdir -p ~/.config/lms-fleet
cp examples/fleet-rollout.tier1.env.example \
  ~/.config/lms-fleet/tier1.env
chmod 600 ~/.config/lms-fleet/tier1.env
$EDITOR ~/.config/lms-fleet/tier1.env
```

Fill every blank value with the exact SSH target, `lms` checkout path, and model root for that node. Keep the completed file outside the repository.

## 3. Resolve and validate the complete Tier-1 configuration

```bash
mkdir -p rollout/tier1-validate
lms-fleet-rollout validate \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --out rollout/tier1-validate/report.json
```

Validation fails before SSH when a variable is blank, a path is not remote-absolute/home-relative, a model root is duplicated, or an endpoint mapping is not loopback-local.

## 4. Render and inspect the observation scripts

```bash
lms-fleet-rollout render \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --update-code \
  --dry-run-limit 6 \
  --output-dir rollout/tier1-render

less rollout/tier1-render/scripts/x1-370.sh
less rollout/tier1-render/scripts/xwing.sh
less rollout/tier1-render/scripts/scotts-macbook-air.sh
```

`--update-code` only changes the generated remote behavior at this stage. During `run`, it fetches and fast-forwards the configured branch before observation. Remove the flag when a machine has already been pinned manually to the exact branch head.

## 5. Run observation and dry-run planning

```bash
lms-fleet-rollout run \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --update-code \
  --continue-on-error \
  --dry-run-limit 6 \
  --output-dir rollout/tier1-observe
```

This stage performs no inference. Each node collects hardware observation, endpoint observation, quick model inventory, a fair model/backend candidate plan, and rendered candidate intent. Remote failures still attempt to package diagnostic evidence, but failed bundles are not promotable.

## 6. Verify the collected observation archives

```bash
lms-fleet-gate \
  --mode observe \
  --rollout-results rollout/tier1-observe/rollout_results.json \
  --required-node x1-370 \
  --required-node xwing \
  --required-node scotts-macbook-air \
  --out rollout/tier1-observe/release-gate.json
```

The gate verifies:

- every required node has one successful remote result;
- collection succeeded and the archive exists;
- archive paths are safe and non-duplicated;
- every manifest size and SHA-256 matches the tar member;
- no unlisted files exist in the archive;
- the bundle records `remote_exit_code=0`;
- observation, inventory, and a non-empty candidate plan are present;
- the plan references the collected observation;
- admission remains false.

A zero exit means the archives are ready for candidate review, not deployment.

## 7. Review candidates one node at a time

Extract only after the gate passes:

```bash
mkdir -p rollout/tier1-observe/extracted/x1-370
mkdir -p rollout/tier1-observe/extracted/xwing
mkdir -p rollout/tier1-observe/extracted/scotts-macbook-air

tar -xzf rollout/tier1-observe/artifacts/x1-370.tar.gz \
  -C rollout/tier1-observe/extracted/x1-370
tar -xzf rollout/tier1-observe/artifacts/xwing.tar.gz \
  -C rollout/tier1-observe/extracted/xwing
tar -xzf rollout/tier1-observe/artifacts/scotts-macbook-air.tar.gz \
  -C rollout/tier1-observe/extracted/scotts-macbook-air
```

Print the candidate matrix:

```bash
python3 - <<'PY'
import json
from pathlib import Path
for node in ('x1-370', 'xwing', 'scotts-macbook-air'):
    path = Path('rollout/tier1-observe/extracted') / node / 'benchmark_plan.json'
    plan = json.loads(path.read_text())
    print(f'\n[{node}]')
    for item in plan['candidates']:
        print(
            item['candidate_id'],
            item['backend'],
            item['model']['id'],
            item['context_tokens'],
            item['parallel_slots'],
        )
PY
```

Begin with one conservative candidate on one node. Do not launch all candidate combinations together.

For an existing XDNA2 NPU service, copy the Tier-1 JSON to a private resolved configuration and add only the reviewed candidate ID to `x1-370.endpoint_map`, using `http://127.0.0.1:1236/v1`. LAN or Tailscale endpoint mappings are rejected.

## 8. Execute an exact reviewed candidate

Example for `x1-370`:

```bash
lms-fleet-rollout run \
  --config /path/to/private-tier1-sweep.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --node x1-370 \
  --execute-candidate x1-370=REVIEWED_CANDIDATE_ID \
  --output-dir rollout/x1-370-sweep-1
```

Then verify the sweep archive:

```bash
lms-fleet-gate \
  --mode sweep \
  --rollout-results rollout/x1-370-sweep-1/rollout_results.json \
  --required-node x1-370 \
  --out rollout/x1-370-sweep-1/release-gate.json
```

Sweep mode additionally requires loopback-only execution, an eligible selected candidate, every hard benchmark gate, and one full selected-model SHA-256. A passing result is ready for reviewed profile import; it still does not admit or route the runtime.

## 9. Promotion order

1. `x1-370`: observation, one conservative llama.cpp/LM Studio candidate, then XDNA2 preflight and exact NPU candidate.
2. `xwing`: observation and one development-worker candidate after the `x1-370` archive flow is proven.
3. `scotts-macbook-air`: observation and one small Metal candidate after the Linux flow is stable.
4. Import successful sweep evidence into `fleet-llm-profiles`.
5. Independently verify runtime identity, private-path behavior, shared capacity, sustained stability, and rollback before live admission.

Keep all four pull requests draft until these physical gates are recorded.
