# Fleet Benchmark — Methodology, Tooling & Results

> Master record of the LM Studio fleet benchmarking effort (2026-07-15).
> The live, regenerable results live in [`fleet_analysis.md`](./fleet_analysis.md); this doc explains **how it was produced**, **what the numbers mean**, and **what was learned**.

## 1. Goal

Benchmark every loaded model on all 9 fleet LM Studio nodes and produce a
per-machine fleet writeup with real RAM/CPU/VRAM capacity analysis for agents
and the orchestrator — so we know which models fit where and how many
concurrent sessions each node can safely serve.

## 2. Fleet topology

Nine nodes, all on Tailscale, all confirmed UP with models loaded:

| Node | Role | Tailscale IP |
|---|---|---|
| x1-370 | Runner / orchestrator (Ryzen AI 9 HX 370, 96 GiB) | 192.168.1.241 (local) |
| xwing | Ryzen AI MAX PRO 390, 23 GiB | 100.108.99.47 |
| destroyer | i7-10610U, 31 GiB | 100.81.57.77 |
| deathstar | i7, 24 GiB, RX480 8 GiB VRAM | 100.78.106.121 |
| beelink | Ryzen 7 (6-8c), 16 GiB, no GPU | 100.85.72.121 |
| joyner | Ryzen 5 iGPU (max 5 GiB VRAM), 16 GiB | 100.83.215.83 |
| scott-optiplex-9030-aio | i5-4590, 7.7 GiB | — |
| lenovo-ideapad-330s-15ikb | i3-8130U, 11.6 GiB | — |
| scotts-macbook-air | Apple Silicon, 8 GiB (struggles >3-4 GiB) | 100.85.64.117 |

## 3. Tooling

| Script | Purpose |
|---|---|
| `bench_fleet.py` | Crash-doc harness: benchmarks all loaded chat models on all nodes, single-stream, `--concurrency N`. |
| `bench_concurrency_probe.py` | Per-model 1-vs-2 concurrent streaming probe. Struggle nodes (optiplex, lenovo, destroyer, deathstar) capped at 1. Records ttft/tps/speed-hit/status. `--max-concurrent 2`. |
| `run_bench_then_probe.sh` | Wrapper: runs `bench_fleet.py`, then auto-chains the concurrency probe when done. |
| `collect_node_profile.py` | Stdlib-only, platform-aware (Linux `/proc`, macOS `sysctl`) hardware collector. Emits `host_profile.json` (real RAM/CPU/VRAM). |
| `fleet_analysis.py` | Reads all `runs/<node>/` dirs + `host_profile.json` → `docs/fleet_analysis.md` (overview, per-node deep dive, capacity model, recommendations). Regenerable. |
| `fleet_writeup.py` | Basic per-machine writeup → `docs/fleet_writeup.md`. |
| `bootstrap_keys.sh` | One-time SSH key installer for nodes reached via RustDesk (foothold before SSH keys exist). |

SSH pubkeys installed on nodes (runner → node):
- `hermes-agent@x1-370` (ed25519 …KbH)
- `hermes@kipnerter` (ed25519 …kSy)

## 4. Concurrency policy (user-specified)

- Test ceiling = **2 concurrent**; hard cap **4** (user-gated).
- Small models (<~4B) benefit from multiplexing; big models (Hermes/30B+) degrade at 2 sessions.
- **Cap-1 nodes** (optiplex, lenovo, destroyer, deathstar): never parallel; one model, single-stream.

## 5. Data interpretation (critical)

- `ok_rate` / `eval_ok_rate` = **quality** (share of eval cases passed), NOT availability.
- Availability = model produced output = `tps_med > 0`.
- Track completion by **`run_summary.csv`**, not `capability_matrix.csv`
  (the latter is stale from crashed runs and reports false "done").

## 6. Incidents & gotchas

1. **Disk full (2026-07-15):** Docker filled the root volume (313 GiB volumes,
   128 GiB images) mid-run, interrupting node benchmarks and leaving zero-tps
   rows. User purged → 808 GiB free. High "Failed" tallies in this pass are
   **artifacts**, not real model breakage.
2. **x1-370 self-contention:** it was benchmarked *while orchestrating* the
   other 8 nodes, penalizing its own numbers. Requires a clean solo re-run for
   final figures.
3. **Stale `macbook-air` dir:** a pre-SSH crash leftover with no
   `host_profile.json` shadowed the real `scotts-macbook-air` data (showing a
   bogus 91.9 GiB runner fallback). Fixed by aliasing `macbook-air →
   scotts-macbook-air` in `fleet_analysis.py`.
4. **Missing SSH keys:** 4 nodes initially rejected the runner key
   (`Permission denied`); resolved by capturing hardware manually from
   owner-provided specs instead of SSH (no RustDesk step needed).

## 7. Hardware capture status

All 9 nodes are now **V** (real/verified hardware):
- 5 captured live via `collect_node_profile.py`: x1-370, xwing, destroyer,
  lenovo, optiplex.
- 4 captured from owner-provided specs (written to `host_profile.json`):
  deathstar (24 GiB / i7 / RX480 8 GiB), beelink (Ryzen 7 16 GiB / no GPU),
  scotts-macbook-air (8 GiB, struggles >3-4 GiB), joyner (16 GiB / Ryzen 5
  iGPU max 5 GiB).

## 8. Capacity model

Effective limit = known VRAM where it is the binding constraint, else system
RAM minus ~4 GiB OS headroom. From `fleet_analysis.md`:

- **deathstar / joyner** are VRAM-bound (8 GiB / 5 GiB) → small models only.
- **beelink / macbook-air / optiplex** are RAM-bound and small.
- **x1-370 / destroyer / xwing** can host the large (30B+) models.
- 35B-class models are too big for every node except x1-370 (and only barely).

## 9. How to regenerate

```sh
python3 fleet_analysis.py      # docs/fleet_analysis.md
# hardware:
python3 collect_node_profile.py | ssh <node> 'cat > runs/<node>/host_profile.json'
# full re-bench:
bash run_bench_then_probe.sh
```

## 10. Open items (resolved)

- ~~`lenovo-ideapad-330s-15ikb` REDO pending~~ — its `run_summary.csv` was
  produced by the redo run; `fleet_state.json` now indexes all 9 nodes.
- ~~Clean solo re-runs for contended nodes~~ — `fleet.py bench --only <node>`
  makes a clean solo re-run a one-liner. The stale crash-era bench pipeline
  (pids 799920 / 3630323) was killed so it could no longer clobber the
  post-disk-cleanup data with contended partial runs.
- ~~Per-node VRAM unknown on GPU-less/Apple nodes~~ — `collect_node_profile.py`
  now reports Apple-Silicon unified memory (`apple-unified`) and Linux
  integrated GPUs (`integrated`, VRAM 0) instead of returning `{}`; the
  owner-provided host profiles for `scotts-macbook-air` / `beelink` were
  updated to the new semantics.

## 11. Next-level enhancement: `fleet.py` control plane

The point-in-time scripts above were unified into a single control plane that
fixes the structural gaps found during the first pass.

### New files
- **`fleet_discover.py`** — single source of truth for node discovery. Reads
  `fleet.toml` (explicit, wins) then augments from `tailscale status --json`.
  Provides `live_nodes()` with **retry + exponential backoff** (the old tools
  used a single probe, so a transient Tailscale blip marked a node permanently
  down) and `all_aliases()` (resolves `macbook-air` ↔ `scotts-macbook-air`
  in one place, replacing the opposite-direction `ALIAS` hacks in the analysis
  scripts).
- **`fleet.toml`** — the authoritative node registry (name, url, aliases,
  notes). Kills the duplicated `NODES` dicts in `bench_fleet.py` /
  `bench_concurrency_probe.py`.
- **`fleet.py`** — subcommand CLI:
  - `discover` — list configured/live nodes.
  - `status` — quick live health snapshot (UP/DOWN per node).
  - `state` — assemble **`fleet_state.json`**, a machine-readable fleet view:
    per-node health, verified hardware, per-model availability + measured tps +
    **derived concurrency tier**, and a per-node `stale` flag (artifacts older
    than 6h are flagged so a crash artifact is never read as live). Also
    indexes the best node per model.
  - `routes` — emit **`routing_rules.json`** from measured state (not the
    hand-curated `NOTES` dicts): per-node max concurrency, preferred
    low-latency models, and a primary+fallback node per model.
  - `bench` — runs `bench_fleet.py` then `bench_concurrency_probe.py`, then
    regenerates state/routes/report.
  - `report` — regenerate the Markdown docs + state + routes.

### What "measured concurrency" replaces
The old `--struggle-nodes` / `NOTES` caps were set by fiat. `fleet.py` instead
reads each model's concurrency-probe `summary` row (its `speed_hit` / `gain`
and `OK/DEGRADED/POOR/FAIL` status) and derives a safe tier per (node, model):
`FAIL`/`POOR` → 1, `DEGRADED` → 2, `OK` (non-negative gain) → 2, otherwise 1.
Models with no probe are conservatively tier-1.

### Usage

  - `plan` — emit **`fleet_loadout.json`**, the orchestrator-consumable
    artifact: per-node mount lists (top-N by demand: balanced/realtime/quality)
    + max-concurrency, excluding stale nodes, plus the full model→best-node
    routing map. This is what `fleet_orchestrator.py` should consume instead of
    hand-derived `NOTES`.
  - `watch` — continuously refresh `fleet_state.json` + `routing_rules.json`
    (default every 900s); pair with `nohup`/`cron` so the machine-readable
    artifacts stay live and the `stale` flags remain meaningful.

### Hardening applied in this pass
- **Single discovery source:** `bench_fleet.py` / `bench_concurrency_probe.py`
  now import `NODES` from `fleet_discover` (no more duplicated dicts) and gate
  targets on `live_nodes()` with retry/backoff.
- **Retry/backoff in the benchmark stage:** `fleet_discover.retry()` wraps the
  node-bench subprocess and the probe's `loaded_models()` call, so a transient
  LM Studio hiccup no longer fails a whole node.
- **Live hardware co-capture:** `bench_fleet.bench_node()` now snapshots real
  hardware *during* the bench pass (local via `collect_node_profile.py`,
  remote best-effort over SSH) into `runs/<node>/host_profile.json`, so capacity
  numbers reflect the contended run rather than a stale manual snapshot.
- **`bootstrap_keys.sh`** now takes keys from `$RUNNER_KEYS` / `$RUNNER_KEY2`
  (defaults still the runner's published *public* ed25519 keys) instead of
  hardcoding them inline.

### Usage
```sh
python3 fleet.py discover
python3 fleet.py status
python3 fleet.py state      # -> fleet_state.json
python3 fleet.py routes     # -> routing_rules.json (from state)
python3 fleet.py plan --demand realtime   # -> fleet_loadout.json
python3 fleet.py bench --only x1-370 --max-concurrent 2
python3 fleet.py report
nohup python3 fleet.py watch --sleep 900 &   # keep state/routes live
```

