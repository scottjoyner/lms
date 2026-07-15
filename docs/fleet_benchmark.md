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

## 10. Open items

- `lenovo-ideapad-330s-15ikb` still "REDO pending" — its `run_summary.csv`
  regenerates from the in-flight redo (pid 948743).
- Clean solo re-runs for x1-370 and beelink (0 ran in current pass) for final
  non-contended figures.
- Per-node VRAM still unknown on GPU-less nodes (n/a) and only owner-provided
  on deathstar/joyner.
