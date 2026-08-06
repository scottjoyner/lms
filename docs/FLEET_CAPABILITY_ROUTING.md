# Tailnet Fleet Capability and Routing Matrix

## Purpose

The fleet is larger than the set of machines that can host a complete Hermes or OpenCode runtime. Every Tailscale node should be visible, but visibility must not be confused with permission to execute an agent, load a model, or run code.

`lms-fleet-routing-matrix` creates one non-admitting artifact that joins:

- the complete `tailscale status --json` census;
- an operator-owned node role policy;
- exact-loadout benchmark comparisons from `lms-loadout-compare`;
- optional auto-router value-matrix exports.

The artifact is consumed by AssistX for durable inventory and allocation evidence. Auto-router receives only approved runtime/model portions through the signed AssistX runtime projection.

## Node states

Every tailnet node appears in the matrix with `tailnet_discovered: true`. A node then has one worker mode:

| Worker mode | Meaning |
| --- | --- |
| `observer_only` | Visible in inventory and topology. Never receives work. |
| `benchmark_only` | May be inventoried or benchmarked under an operator workflow. Not a production worker. |
| `auxiliary` | May receive bounded model tasks such as summarization, compression, or extraction. Does not run a full agent runtime. |
| `agent` | May run the explicitly allowed Hermes/OpenCode execution contract. |

Unknown Tailscale peers default to `observer_only`. This intentionally keeps phones, tablets, Raspberry Pis, routers, and unreviewed machines visible without making them routable.

## Role model

Roles grant narrow capabilities:

- `full_agent`: complete local agent runtime and its approved tools;
- `code_agent`: bounded code tasks and approved code execution;
- `tool_agent`: structured tool-use tasks;
- `reasoning`: reasoning workloads without granting an agent shell;
- `long_context`: context-heavy reading and synthesis;
- `auxiliary_llm`: summarization, compression, and extraction;
- `summarization`, `compression`, `extraction`: single-purpose auxiliary roles;
- `benchmark_only`: observation and benchmark evidence only;
- `observer`: inventory only.

Roles do not admit a model. Runtime admission still requires fresh loaded-model identity, capacity, private access paths, and operator approval in AssistX.

## Build the matrix

Install the current checkout:

```bash
python3 -m pip install -e '.[test]'
```

Copy and review the role policy:

```bash
mkdir -p ~/.config/lms-fleet ~/lms-fleet-runs
cp examples/fleet-role-policy.v1.json ~/.config/lms-fleet/fleet-role-policy.v1.json
$EDITOR ~/.config/lms-fleet/fleet-role-policy.v1.json
```

Build directly from the local Tailscale daemon and one or more comparison artifacts:

```bash
lms-fleet-routing-matrix \
  --policy ~/.config/lms-fleet/fleet-role-policy.v1.json \
  --comparison ~/lms-fleet-runs/compare/current-loadouts.json \
  --comparison ~/lms-fleet-runs/compare/router-value-matrix.json \
  --out ~/lms-fleet-runs/fleet-routing-matrix.json
```

For deterministic tests or an offline controller, capture Tailscale status first:

```bash
tailscale status --json > ~/lms-fleet-runs/tailscale-status.json
lms-fleet-routing-matrix \
  --tailscale-json ~/lms-fleet-runs/tailscale-status.json \
  --policy ~/.config/lms-fleet/fleet-role-policy.v1.json \
  --comparison ~/lms-fleet-runs/compare/current-loadouts.json \
  --out ~/lms-fleet-runs/fleet-routing-matrix.json
```

## Publish to AssistX

The CLI can publish the completed artifact without placing credentials in process arguments:

```bash
export BASIC_AUTH_USER=admin
export BASIC_AUTH_PASS='read-from-a-secret-store'

lms-fleet-routing-matrix \
  --policy ~/.config/lms-fleet/fleet-role-policy.v1.json \
  --comparison ~/lms-fleet-runs/compare/current-loadouts.json \
  --out ~/lms-fleet-runs/fleet-routing-matrix.json \
  --assistx-url http://127.0.0.1:8000
```

AssistX stores discovery and benchmark evidence in Neo4j. It does not automatically admit a runtime or enable an agent.

## Ranking policy

Quality and speed are not averaged indiscriminately.

1. Select nodes whose operator role permits the task family.
2. Exclude offline and observer-only nodes.
3. Apply a task-specific quality floor when sufficient quality evidence exists.
4. Rank remaining candidates using task-specific weights for quality, speed, reliability, and evidence confidence.

Coding, reasoning, and tool use have high quality floors. Summarization and compression place more weight on throughput after meeting a lower quality floor. A very fast model that fails the quality floor is not selected merely because it produces more tokens per second.

## Recommended deployment roles

The committed example is a starting point, not automatic admission:

- `x1-370`, `xwing`: full agent and high-quality work after exact-loadout qualification;
- Apple Silicon nodes: full or auxiliary roles based on current benchmark evidence;
- legacy Linux nodes: auxiliary summarization/compression/extraction unless benchmarks justify broader work;
- powered-off or unresolved nodes: benchmark-only or observer-only;
- unlisted tailnet devices: observer-only.

Update roles after every meaningful hardware, model, quantization, runtime, context, or KV-cache change. Benchmark evidence is tied to the exact loadout fingerprint, not the model name alone.

## Safety boundaries

The matrix is evidence only:

- it does not load or unload models;
- it does not grant SSH or shell access;
- it does not claim tasks;
- it does not bypass AssistX admission or claim fencing;
- it does not make phones or miscellaneous tailnet peers production workers;
- it does not treat `/v1/models` visibility as physical loaded-model identity.
