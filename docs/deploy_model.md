# Deploying a model to the fleet

This is the repeatable runbook for adding **one model** to the swarm so it is
mounted on the right nodes, discovered by the router, benchmarked, and then
*leveraged* by `auto` routing. It is the operator-facing counterpart to
`fleet_orchestrator.md` (the control loop) and the router's routing policy.

The swarm runs as a **two-layer decision system**:

- **Placement decisions** (capacity level, seconds–minutes): *where* to mount a
  model. Driven by `fleet_orchestrator.py` from the measured capability matrix
  (`runs/<node>/*.csv`). Spec-fit + tokens/sec + coverage + demand profile.
- **Routing decisions** (request level, milliseconds): *which mounted copy* serves
  each incoming request. Driven by `auto-router` policy — capability tier → latency
  EMA → plan-time reservations → liveness/circuit-breaker gating.

A model is only useful once **both** layers know about it. Deployment is the act of
getting it mounted (layer 1) and verified routable (layer 2), then feeding the
value layer (benchmarks) so both layers keep making good decisions.

---

## 0. Prerequisites

- `auto-router` is running and reachable (default `http://localhost:8088` or
  `http://x1-370.tailcb8954.ts.net:8088` over tailscale).
- The target node(s) are online over tailscale with LM Studio `:1234` up.
- The **node reporter** is deployed on the target node so the orchestrator can read
  its downloaded library and the router gets a real-time fleet view:

  ```bash
  # from the lms/ dir, on the control host (x1-370)
  ./deploy_reporter.sh <user@host>            # router-url defaults to x1-370
  ./deploy_reporter.sh <user@host> http://<router-tailscale-ip>:8088
  ```

  > Without the reporter, `bootstrap` cannot guess a node's exact model identifiers
  > and will refuse to mount ("no reported library"). macOS nodes need a launchd
  > plist instead of the systemd unit in `deploy_reporter.sh` (see §7).

- The fleet device inventory exists at `fleet_baseline.csv` (device_id, hostname,
  os, tailscale_ip). Discovery is baseline-driven; tailscale is only a fallback.

---

## 1. Decide *where* to place it (placement decision)

Do this **before** loading. The orchestrator never mounts a model a node cannot
fit, and it prefers nodes where the model is fast.

### 1a. Check fit + performance from the capability matrix

Per node, `runs/<node>/model_fit.csv` and `runs/<node>/run_summary.csv` hold the
measured data:

- `fit_grade` (`good` / `ok` / `tight` / `poor`) and `estimated_model_memory_gib`
  vs `available_ram_gib` → a *rough* RAM-only fit hint (it marks almost everything
  "good", including 27–35B models that actually fail to load).
- `ok_rate` (from `run_summary.csv`) → **the real gate.** The planner only mounts a
  model on a node when it benchmarked successfully there (`ok_rate >= 0.5`). A model
  that never loaded/ran is excluded regardless of `fit_grade`.
- `reliability_grade` (from `capability_matrix.csv`) → per-task reliability; low
  reliability demotes a model in the ranking even if it has decent tps.
- `tps_med` (tokens/sec), `ttft_med` (time-to-first-token) → realtime suitability.
- `eval_score_avg` → quality suitability.

> Per-node RAM/VRAM in `model_fit.csv` is captured from the benchmark *runner* host,
> not the remote node, so it is not reliable for nodes that differ from the runner
> (e.g. the optiplex/lenovo/Macs). Trust `ok_rate` + `tps_med` — the measured signal —
> over the RAM estimate when deciding placement.

```bash
# all nodes that have benchmarked this model
grep -H "<model-key>" runs/*/model_fit.csv runs/*/run_summary.csv
```

### 1b. Apply the demand profile

Placement is demand-driven (`fleet_orchestrator.py plan/apply --demand`):

| profile    | prefers                                  |
|------------|------------------------------------------|
| `realtime` | best `tps_med`, low `ttft_med`, must fit |
| `quality`  | best `eval_score_avg`, must fit          |
| `balanced` | blend of tps and quality                 |

### 1c. Coverage + node classes

- **BIG_NODES** (`xwing`, `x1-370`, `deathstar`, `beelink-ryzen-7-mini-pc`) may get
  a large model mounted in addition to small/fast ones.
- The planner keeps total mounted memory under ~90% of `available_ram_gib` and mounts
  at most ~4 models per node ("each machine can run a few models").
- Ensure **≥1 healthy mounted copy** of every routing alias you care about
  (coverage-aware) so `auto` always has a candidate.

```bash
python3 fleet_orchestrator.py status --only <node>     # what is loaded + busy now
python3 fleet_orchestrator.py plan  --demand realtime  # see the proposed mount set
```

---

## 2. Acquire the model on the target node(s)

A model must be **downloaded into that node's LM Studio library** before it can be
mounted. Do this on the node itself (or copy the GGUF):

```bash
# on the target node
lms get <model-key>                 # e.g. lms get google/gemma-4-12b-qat
# or via the LM Studio UI / by placing the .gguf in the LM Studio models folder
```

Verify it is in the library (the reporter publishes this; you can also ask directly):

```bash
ssh <user@host> 'lms ls --json' | python3 -m json.tool | grep -i "<model-key>"
```

---

## 3. Load (mount) the model

### 3a. Automated, conservative first load — `bootstrap`

For a model's **first ever** mount on a node there is usually no capability data
yet, so use `bootstrap`, which reads the node's *live reported library* and mounts a
safe representative set (embedding + tiny + mid + [big node] large). Run dry first:

```bash
python3 fleet_orchestrator.py bootstrap --only <node>            # dry-run
python3 fleet_orchestrator.py bootstrap --only <node> --apply    # actually load
```

To mount **only your specific model** without the conservative mix, load it directly
(see 3b). `bootstrap` is best for "make sure every node has *something* useful."

### 3b. Targeted load — `apply` (after benchmarks exist) or `lms load`

Once the model is benchmarked on a node (§6), `plan`/`apply` place it by the demand
profile instead of the conservative mix:

```bash
python3 fleet_orchestrator.py plan  --demand realtime --only <node>
python3 fleet_orchestrator.py apply --demand realtime --only <node> --apply
```

Or load it by hand on the node (no orchestrator needed):

```bash
ssh <user@host> 'lms load <model-key>'
# equivalent raw LM Studio API used by the orchestrator:
curl -s -X POST http://<node-ip>:1234/api/v1/models/load \
  -H 'content-type: application/json' -d '{"model":"<model-key>"}'
```

> `apply`/`bootstrap` default to **dry-run**. Nothing mounts/unloads without
> `--apply`. The orchestrator refuses to unmount a model that is currently busy
> (active generation), so loadout changes never interrupt a live request.

---

## 4. Publish & discover

Mounted models are discovered two ways; both feed the router:

1. **Node reporter → router pubsub.** The reporter posts library + loaded models to
   `POST /api/fleet/node-report` every 30s. The router aggregates at
   `GET /api/fleet/nodes`.
2. **Router model-registry scan.** The router periodically probes each node's
   `/api/v1/models` (see `refresh_fleet_health_task`) and records what is loaded.

So within ~30s of loading, the model should appear in the router's catalog:

```bash
curl -s http://localhost:8088/v1/models | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d['data'] if '<model-key>' in m['id']])"

curl -s http://localhost:8088/api/fleet/nodes | python3 -c \
  "import sys,json; d=json.load(sys.stdin); [print(n['hostname'], [m for m in n.get('loaded',[]) if '<model-key>' in m]) for n in d['nodes']]"
```

If it does not appear, confirm the node reporter is posting (`deploy_reporter.sh`
status) and that LM Studio on the node actually loaded it (`lms ls` / `:1234/v1/models`).

---

## 5. Verify it is routed & leveraged

Send a real `auto` request and confirm the new model is selected and the load is
spread, not piled onto one node:

```bash
curl -s -X POST http://localhost:8088/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"ping"}],"max_tokens":16}'
```

Check which node served recent traffic:

```bash
curl -s "http://localhost:8088/admin/usage?limit=30" | python3 -c \
  "import sys,json,collections as c; d=json.load(sys.stdin);
   print(c.Counter(r['provider_id'] for r in d['recent']))"
```

Routing is **capability-tier** aware: an exact `model` request goes to that model;
an `auto` request is balanced across the tier (text:S <3B / text:M 3–14B /
text:L >14B / vision / embed) by learned latency, so a newly mounted copy will be
picked up once it has a latency sample.

---

## 6. Benchmark it (feeds the value layer)

Benchmarks are what make placement and routing *value-aware* instead of guesswork.
Run the benchmark for the node(s) you mounted on; artifacts land in `runs/<node>/`
where the orchestrator and router read them:

```bash
python3 bench_fleet.py --only <node>            # one pass
python3 bench_fleet.py --loop --sleep 3600      # periodic, background
```

`bench_fleet.py` runs the agent skill suite per model, then `lms_model_fit.py`
writes `model_fit.csv` + `run_summary.csv`. Until this exists for a node×model, the
planner reports "no model-fit data" and `apply` will not place that model there — so
**benchmark before relying on automated placement**.

---

## 7. Register so bootstrap/plan include it

- The orchestrator reads the node's **live library** (from the reporter), so a
  downloaded model is automatically a candidate for `bootstrap` — no static list to
  edit.
- `fleet_baseline.csv` is the device inventory (add a node there to enroll it in
  discovery). Model identifiers themselves are discovered dynamically, not listed.
- For macOS nodes, `deploy_reporter.sh` installs a **systemd** user unit, which does
  not exist on macOS. Use a `launchd` plist instead:

  ```bash
  # on the macOS node, as the user (no sudo)
  mkdir -p ~/Library/LaunchAgents
  cat > ~/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist <<'EOF'
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0"><dict>
    <key>Label</key><string>com.hermes.fleet-node-reporter</string>
    <key>ProgramArguments</key>
    <array><string>/usr/bin/python3</string><string>$HOME/fleet_node_reporter.py</string>
          <string>--router-url</string><string>http://100.64.43.123:8088</string>
          <string>--interval</string><string>30</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>WorkingDirectory</key><string>$HOME</string>
    <key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  </dict></plist>
  EOF
  launchctl load ~/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist
  ```

  (Copy `fleet_node_reporter.py` to `~` on the Mac first, same as the systemd path.)

---

## 8. Update / unload safely

```bash
# unmount on a node (orchestrator skips it if busy)
python3 fleet_orchestrator.py apply --demand realtime --only <node> --apply
# or directly:
ssh <user@host> 'lms unload <model-key>'
curl -s -X POST http://<node-ip>:1234/api/v1/models/unload \
  -H 'content-type: application/json' -d '{"model":"<model-key>"}'
```

The router's circuit breakers + liveness gating handle a node that drops a model
mid-flight; in-flight requests fail over to the next candidate.

---

## 9. Decision model — how placement + routing compose

This is the "efficient model of decision making and distributed workloads" the
swarm is built around:

```
                 demand profile (realtime/quality/balanced)
                          │
            ┌─────────────┴─────────────┐
            ▼                            ▼
   PLACEMENT (fleet_orchestrator)   ROUTING (auto-router policy)
   slow, capacity-level:            fast, request-level:
   • spec-fit (RAM/VRAM budget)     • capability tier (text S/M/L, vision, embed)
   • tps / quality from benchmarks   • latency EMA (learned per node)
   • coverage (≥1 copy/alias)        • plan-time reservations (concurrent spread)
   • demand-driven loadout          • liveness + circuit-breaker gating
            │                            │
            ▼                            ▼
   mounted models on nodes ──discovered──▶ router candidate set
                                          │
                                          ▼
                                each request → best mounted copy
```

- **Placement** answers "which machines should hold this weight?" using measured
  capacity, so VRAM/RAM is never overcommitted and fast nodes get the realtime load.
- **Routing** answers "which copy should serve *this* request *now*?" using live
  latency and health, spreading concurrent work and failing over on bad nodes.

Together they keep the fleet's *mounted* models aligned with what is actually
needed, and steer every request to the cheapest healthy copy — the distributed
workload scheduler. Benchmarks (§6) are the feedback signal that makes both layers
improve over time.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Model never appears in `/v1/models` | Node reporter not posting, or LM Studio didn't load it | check `deploy_reporter.sh` status on node; `lms ls` on node |
| `bootstrap` says "no reported library" | Reporter not deployed on that node | run `deploy_reporter.sh <user@host>` |
| `plan` says "no model-fit data" | Not benchmarked on that node yet | run `bench_fleet.py --only <node>` (§6) |
| `auto` ignores the new model | No latency sample yet / tier mismatch | send a few requests; confirm tier in policy; check `/admin/usage` |
| Requests 503/504 under load | Fleet capacity < concurrency (overload) | mount more copies / bigger nodes; this is correct back-pressure, not a hang |
| Router health `000` / loop wedged | Sync I/O on event loop (fixed) — if seen again, check `asyncio.to_thread` coverage | ensure hot-path redis/sqlite are off the loop |
