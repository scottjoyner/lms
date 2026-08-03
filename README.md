# LMS Agent Benchmarking Toolkit

LMS is an agent-facing toolkit for profiling and benchmarking local or Tailscale-reachable LM Studio and OpenAI-compatible inference nodes. It provides deterministic benchmark evidence, hardware-aware model planning, guarded physical candidate execution, and non-admitted runtime recommendations.

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
lms-fleet-rollout      Validated SSH rollout rendering and execution
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

## Fast start for agent benchmarks

```bash
# Check local scripts and the default LM Studio endpoint.
lms-agent doctor

# List models from local LM Studio server mode.
lms-agent probe

# Run profile + manifest benchmark + recommendations.
lms-agent quick

# Benchmark an explicitly reachable endpoint.
lms-agent quick --endpoint http://100.64.0.10:1234/v1

# Limit a quick run to known model IDs.
lms-agent quick --models 'qwen/qwen3-coder-30b,openai/gpt-oss-20b'

# Inspect and route from the latest evidence.
lms-agent show latest --task coding
lms-agent route latest --task structured_output
lms-agent route latest --task long_context --json
```

The default endpoint is `http://127.0.0.1:1234/v1`, unless `LMS_BASE_URL` or `LMSTUDIO_BASE_URL` is set.

## Tier-1 physical rollout

The first controlled physical rollout targets:

- `x1-370`
- `xwing`
- `scotts-macbook-air`

Use these retained artifacts:

```text
examples/fleet-rollout.tier1.template.json
examples/fleet-rollout.tier1.env.example
docs/TIER1_ROLLOUT_CHECKPOINT.md
```

Prepare the private environment file outside the repository, validate it before SSH, render and inspect the generated scripts, run observation-only collection, and verify the archives:

```bash
lms-fleet-rollout validate \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --out rollout/tier1-validate/report.json

lms-fleet-rollout run \
  --config examples/fleet-rollout.tier1.template.json \
  --env-file ~/.config/lms-fleet/tier1.env \
  --all-nodes \
  --update-code \
  --continue-on-error \
  --dry-run-limit 6 \
  --output-dir rollout/tier1-observe

lms-fleet-gate \
  --mode observe \
  --rollout-results rollout/tier1-observe/rollout_results.json \
  --required-node x1-370 \
  --required-node xwing \
  --required-node scotts-macbook-air \
  --out rollout/tier1-observe/release-gate.json
```

Observation mode performs no inference. Real inference requires exact reviewed `NODE_ID=CANDIDATE_ID` values. Physical benchmark endpoint mappings must be loopback-local.

Cross-repository rollout progress is tracked in GitHub issue #7.

## Agent CLI commands

### `lms-agent doctor`

Checks local scripts and probes endpoint reachability.

```bash
lms-agent doctor
lms-agent doctor --endpoint http://127.0.0.1:1234/v1
```

### `lms-agent probe`

Lists models available from one or more OpenAI-compatible endpoints.

```bash
lms-agent probe
lms-agent probe --endpoint http://100.64.0.10:1234/v1
lms-agent probe --json
```

For tailnet-backed registries:

```bash
lms-bench-endpoints discover-tailscale
lms-bench quick --from-registry --discover-tailscale
LMS_DISCOVER_TAILSCALE=1 lms-bench quick --from-registry
```

### `lms-agent inventory`

Creates the benchmark inventory CSV.

```bash
lms-agent inventory --out lmstudio_inventory.csv
lms-agent inventory --endpoint http://100.64.0.10:1234/v1 --max-models 3
```

Columns:

```csv
host_name,host_ip,endpoint_id,base_url,reachable,model_id,model_key
```

### `lms-agent profile`

Collects machine and endpoint information without a full benchmark.

```bash
lms-agent profile --output-dir runs/profile-local
lms-agent profile --endpoint http://100.64.0.10:1234/v1 --output-dir runs/profile-xwing
```

### `lms-agent quick`

Runs the default agent workflow:

1. Probe endpoints.
2. Write the inventory.
3. Collect machine profile artifacts.
4. Run the packaged agent skill suite.
5. Apply deterministic evaluators.
6. Produce run and task summaries.
7. Produce capability and routing recommendations.

```bash
lms-agent quick
lms-agent quick --endpoint http://100.64.0.10:1234/v1 --max-models 2 --repeats 1
lms-agent quick --max-context-tokens 16384
lms-agent quick --profile-only
```

### Evidence inspection and routing

```bash
lms-agent runs
lms-agent show latest --task coding
lms-agent route latest --task structured_output
lms-agent route latest --task repo_work --write
lms-agent recommend latest
```

### Deterministic evaluation

```bash
echo '{"status":"ok"}' | \
  lms-agent eval --evaluators-json '[{"type":"json_parse"}]' --pretty
```

## Output contract

A normal agent benchmark creates a self-contained run directory:

```text
runs/<run_id>/
  lms_run_config.json
  endpoint_probes.json
  machine_profile.json
  machine_synopsis.md
  lmstudio_inventory.csv
  config.json
  run_results.csv
  run_summary.csv
  task_summary.csv
  capability_matrix.csv
  agent_recommendations.md
  routing_rules.yaml
  routing_rules.json
  agent_skill_suite.v1.json
  sidecars/
```

The fleet rollout path additionally creates immutable node archives containing observations, model inventory, candidate plans, execution manifests, selected loadouts, selected-model fingerprints, and a per-file SHA-256 bundle manifest.

## Deterministic benchmark evidence

Important aggregate fields include:

```csv
run_id,host_name,host_ip,base_url,model_key,load_s,ttft_med,tps_med,ok_rate,eval_ok_rate,eval_score_avg,cases
```

Fleet selection additionally requires:

- request success of at least 98%;
- streaming success;
- declared concurrency success;
- no observed crash;
- at least 10% system-memory headroom;
- sustained-stability success;
- loopback-only physical execution evidence.

A selected loadout remains desired state only. It does not admit or route the runtime.

## Benchmark suite

The packaged suite is:

```text
src/lms_agent_bench/benchmarks/agent_skill_suite.v1.json
```

Task families include operational health, structured output, coding, agent planning, long-context behavior, and repository-work simulation. Evaluators are deterministic and include JSON parsing, required keys, contains checks, regex checks, length checks, and markdown-fence restrictions.

## Product objective

The toolkit should answer:

1. What hardware and acceleration backends are currently available?
2. What local models and quantizations are present?
3. Which model/loadout combinations fit safely?
4. What can each combination do reliably at measured latency and throughput?
5. Where are its context, memory, stability, and instruction-following limits?
6. Which reviewed desired loadout should proceed to external identity, path, capacity, and rollback gates?

## Safety and authority rules

- Do not require cloud services for local benchmarking.
- Do not require internet access during benchmark execution.
- Do not leak prompts, credentials, model output, or private paths outside retained evidence.
- Keep physical candidate endpoints loopback-local.
- Treat LAN, Tailscale, loopback, and LM Studio Link URLs as access paths to one physical capacity record.
- Execute only explicitly reviewed candidate IDs.
- Preserve failed archives for diagnosis, but never import or admit them.
- Treat LLM-as-judge as optional; deterministic evaluation is authoritative for hard gates.
- Make every recommendation traceable to immutable benchmark evidence.
- Keep profile output and selection artifacts non-admitted until the external live authority verifies runtime identity, model SHA-256, shared capacity, path behavior, sustained stability, freshness, and rollback.
