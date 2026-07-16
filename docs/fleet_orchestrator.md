# Fleet Orchestrator — dynamic model loadout harness

## Why this exists

The router (`auto-router`) routes requests across the fleet, but it can only route
to models that are **actually loaded** on a node. Right now the fleet is mostly
*unloaded* — nodes are online over tailscale but their LM Studio `:1234` endpoints
report zero models, so concurrent requests 504 even though the hardware exists.

Routing fail-fast (per-attempt timeout, circuit breakers, liveness gating) is
necessary plumbing, but it does not *create* capacity. **Capacity is created by
mounting the right models on the right machines.** That is this harness's job.

## The model

```
 tailscale status ──▶ discover_nodes() ──▶ list of fleet nodes (hostname, ip, online)
        │
        ▼
 probe_node(ip:1234) ──▶ live state per node: loaded models, busy?, specs reachable?
        │
        ▼
 capability_store ◀── runs/<node>/{model_fit.csv, run_summary.csv}
   (per node×model: RAM/VRAM fit, tps_med tokens/sec, ttft, quality score)
        │
        ▼
 plan_loadouts(nodes, capability, demand)
   • spec-aware: never mount a model that doesn't fit the node's RAM/VRAM budget
   • tps-aware: for realtime demand, prefer the model with the best tokens/sec
   • coverage-aware: ensure every routing alias has ≥1 healthy mounted candidate
   • dynamic: bigger/smaller/mixed loadouts depending on the live demand profile
   • safe: never unmount a model that is currently busy (active generation)
        │
        ▼
 apply_loadouts(plan)  ──▶ LM Studio load/unload API on each node (default dry-run)
```

## Key concepts

### Discovery (programmatic, not hard-coded)
`discover_nodes()` shells `tailscale status --json`, enumerates every peer that is
`Online`, and adds the local host (Self). For each it builds the LM Studio base
URL `http://<ip>:1234/v1` and the native models URL `http://<ip>:1234/api/v1/models`.
This replaces the hand-maintained `NODES` dict in `bench_fleet.py` — the fleet is
whatever tailscale sees right now.

### Busy detection
A node is "busy" when it currently has active generations. The authoritative source
is the router's per-owner in-flight counts (`fleet.json` / admin endpoint). The
orchestrator reads that and **refuses to unmount a model that is actively serving**,
so loadout changes never interrupt a live request.

### Spec awareness (from `lms_model_fit.py`)
`model_fit.csv` per node records `estimated_model_memory_gib`, `system_ram_gib`,
`available_ram_gib`, `largest_nvidia_vram_gib`, and a `fit_grade` (good/…). The
planner uses this to keep total mounted model memory under the node's budget, so a
node "can run a few models, all within the specs it has."

### Performance awareness (from `run_summary.csv`)
`run_summary.csv` per node×model records `tps_med` (median tokens/sec),
`ttft_med` (time-to-first-token), and quality scores. This is the measured
"how each machine performs with different models and variants" data. For a
**realtime** request the planner ranks candidates by `tps_med`/`ttft`; for a
**quality** request it ranks by quality score.

### Demand-driven, dynamic loadouts
`demand` is a profile, e.g.:
- `realtime`: maximize tokens/sec, small/fast models, low ttft.
- `quality`: maximize eval score, larger models.
- `balanced`: a mix.

The planner produces a target loadout per node that satisfies the demand while
respecting specs, then diffs against the live `probe_node` state to emit only the
`load`/`unload` actions needed. Because it is diff-based and demand-driven, the
fleet can shift from "mostly small fast models" to "a couple of big quality models
+ fast ones" as demand changes — without human intervention.

## Safety
- `apply_loadouts` defaults to `dry_run=True`. Nothing is mounted/unmounted unless
  `--apply` is passed.
- Never unmount a model that is busy.
- Mounts are best-effort; the router's circuit breakers + liveness gating handle
  any node that fails after a model is loaded.

## Relation to the rest of the stack
- `bench_fleet.py` / `lms_model_fit.py` **produce** the capability data this reads.
- `auto-router` **consumes** the resulting mounted models (and tells us what is busy).
- This harness is the **control loop** between them: it watches demand + capacity
  and keeps the fleet's mounted models aligned with what the router needs.

## See also

- `docs/deploy_model.md` — operator runbook for adding one model to the fleet
  (acquire → place → load → verify → benchmark), including the two-layer decision
  model (placement vs. routing) and troubleshooting.

## Measured loadout (`loadout` command)
In addition to the capability-driven `plan`/`apply` flow, the orchestrator can
consume the *measured* loadout produced by `fleet.py plan`:

```
python3 fleet_orchestrator.py loadout            # dry-run convergence from fleet_loadout.json
python3 fleet_orchestrator.py loadout --apply    # actually mount/unmount
python3 fleet_orchestrator.py loadout --only x1-370
```

`cmd_loadout` reads `fleet_loadout.json` (per-node mount lists + best-node-per-model
routing, all derived from real benchmark tps and the measured concurrency tiers),
probes each node for its currently-loaded and busy models, and reuses the existing
`apply_loadouts` actuator to emit only the `load`/`unload`/`keep_busy` actions needed.
This is the payoff for the benchmark pipeline: it bypasses the RAM-only fit grades
that can label a 35B model "fits" on a small node, and instead mounts what actually
ran at useful throughput. Nodes absent from `fleet_loadout.json` are skipped.

## Relation to the rest of the stack
- `bench_fleet.py` / `lms_model_fit.py` **produce** the capability data this reads.
- `fleet.py plan` produces `fleet_loadout.json` (measured mount lists) which the
  `loadout` command here consumes — closing the benchmark→orchestrate loop.
- `auto-router` **consumes** the resulting mounted models (and tells us what is busy).
- This harness is the **control loop** between them: it watches demand + capacity
  and keeps the fleet's mounted models aligned with what the router needs.
- `fleet.py watch` keeps `fleet_state.json` / `routing_rules.json` live in the
  background (default every 900s) so the measured artifacts never go stale.
