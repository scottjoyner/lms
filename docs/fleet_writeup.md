# Fleet Writeup - per-machine capabilities

> Generated: 2026-07-15 21:51 UTC  
> Status: **preliminary** - crash-doc pass and concurrency probe may still be running. Regenerate with `python3 fleet_writeup.py`.

## Fleet overview

- **Machines profiled:** 10
- **Model benchmark rows so far:** 91
- **Concurrency-capable (tested at 2):** beelink-ryzen-7-mini-pc, joyner, macbook-air, scotts-macbook-air, x1-370, xwing
- **Concurrency-limited (cap 1):** deathstar, destroyer, lenovo-ideapad-330s-15ikb, scott-optiplex-9030-aio

### Concurrency principles (from fleet observation)

- Small models generally *benefit* from concurrency: more simultaneous sessions with acceptable latency, better aggregate throughput.
- Large models (Hermes-class, 30B+) degrade sharply at 2 concurrent sessions - response times balloon or requests fail. Keep big models single-stream.
- A handful of nodes (optiplex, lenovo, destroyer, deathstar) choke under any parallel load and are capped at 1 concurrent request in testing.
- deathstar additionally cannot run models >7 GiB reliably (CPU maxed by other jobs).
- x1-370 is the strongest concurrency node (96 GiB RAM) but co-resident services keep per-stream throughput low (~2-8 tok/s under concurrent load) - fine for multiplexing.

## Per-machine

### beelink-ryzen-7-mini-pc

(24 logical)

- **Models benchmarked:** 9 (ok: 8, errors: 1)
- **Median tps (non-embedding, ok):** 7.6
- **Fastest:** liquid/lfm2.5-1.2b (18.6 tok/s), vibethinker-3b-heretic_decensored (13.6 tok/s), vibethinker-3b-hermes (10.4 tok/s)
- **Slowest:** google/gemma-4-12b-qat (0.9 tok/s), orinth-1.0-9b (0.7 tok/s), qwen3.6-28b-reap20-a3b (0.2 tok/s)
- **Errors/crashes:** qwen3.6-14b-a3b-vibeforged-v2
- **Fit grades:** {'good': 8, 'unknown': 1}

**Concurrency posture:** Test at 2 concurrent.

_Sources:_ `runs/beelink-ryzen-7-mini-pc/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### deathstar

(24 logical)

- **Models benchmarked:** 20 (ok: 2, errors: 18)
- **Median tps (non-embedding, ok):** 42.0
- **Fastest:** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (44.2 tok/s), refinedtoolcallv5-3b (39.8 tok/s)
- **Slowest:** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (44.2 tok/s), refinedtoolcallv5-3b (39.8 tok/s)
- **Errors/crashes:** text-embedding-nomic-embed-text-v1.5; ornith-1.0-9b; ornith-1.0-35b; vibethinker-3b-hermes; vibethinker-3b-i1; qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex
- **Fit grades:** {'unknown': 2, 'good': 18}

**Concurrency posture:** Force concurrency 1 in the probe. CPU cores are maxed by unrelated work, so it cannot absorb parallel load.
**Constraints:** Models >7 GiB struggle / may stall or crash. Keep large models off this node or accept very slow, unreliable throughput.

_Sources:_ `runs/deathstar/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### destroyer

(24 logical)

- **Models benchmarked:** 11 (ok: 1, errors: 10)
- **Median tps (non-embedding, ok):** 4.7
- **Fastest:** liquid/lfm2-24b-a2b (4.7 tok/s)
- **Slowest:** liquid/lfm2-24b-a2b (4.7 tok/s)
- **Errors/crashes:** mradermacher/vibethinker-3b-hermes; refinedneuro/vibethinker-3b-hermes; vibethinker-3b-i1; qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp; qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex; google/gemma-4-12b-qat
- **Fit grades:** {'good': 10, 'unknown': 1}

**Concurrency posture:** Chokes under concurrency. Cap at 1.
**Constraints:** Single-stream small models only.

_Sources:_ `runs/destroyer/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### joyner

(24 logical)

- **Models benchmarked:** 4 (ok: 1, errors: 3)
- **Median tps (non-embedding, ok):** 6.2
- **Fastest:** refinedtoolcallv5-3b (6.2 tok/s)
- **Slowest:** refinedtoolcallv5-3b (6.2 tok/s)
- **Errors/crashes:** ornith-1.0-9b; google/gemma-4-12b-qat; text-embedding-nomic-embed-text-v1.5
- **Fit grades:** {'good': 3, 'unknown': 1}

**Concurrency posture:** Test at 2 concurrent.

_Sources:_ `runs/joyner/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### lenovo-ideapad-330s-15ikb

(24 logical)

- _benchmark not yet complete for this node_

**Concurrency posture:** Chokes under concurrency. Cap at 1.
**Constraints:** Single-stream small models only.

_Sources:_ `runs/lenovo-ideapad-330s-15ikb/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### macbook-air

(24 logical)

- **Models benchmarked:** 4 (ok: 3, errors: 1)
- **Median tps (non-embedding, ok):** 42.5
- **Fastest:** liquid/lfm2.5-1.2b (57.2 tok/s), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (27.7 tok/s)
- **Slowest:** liquid/lfm2.5-1.2b (57.2 tok/s), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (27.7 tok/s)
- **Errors/crashes:** refinedtoolcallv5-3b

**Concurrency posture:** Test at 2 concurrent; mind unified-memory pressure.

_Sources:_ `runs/macbook-air/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### scott-optiplex-9030-aio

(24 logical)

- **Models benchmarked:** 11 (ok: 1, errors: 10)
- **Median tps (non-embedding, ok):** 8.8
- **Fastest:** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (8.8 tok/s)
- **Slowest:** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (8.8 tok/s)
- **Errors/crashes:** refinedtoolcallv5-3b; vibethinker-3b-i1; qwen3.5-2b-claude-4.6-opus-reasoning-distilled; lfm2.5-8b-a1b; qwen3.5-4b; qwen3.5-2b
- **Fit grades:** {'good': 9, 'unknown': 2}

**Concurrency posture:** Chokes under concurrency. Cap at 1.
**Constraints:** Don't multiplex models here; single-stream small models only.

_Sources:_ `runs/scott-optiplex-9030-aio/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### scotts-macbook-air

(24 logical)

- **Models benchmarked:** 4 (ok: 4, errors: 0)
- **Median tps (non-embedding, ok):** 29.1
- **Fastest:** liquid/lfm2.5-1.2b (56.7 tok/s), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (29.0 tok/s), refinedtoolcallv5-3b (1.7 tok/s)
- **Slowest:** liquid/lfm2.5-1.2b (56.7 tok/s), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (29.0 tok/s), refinedtoolcallv5-3b (1.7 tok/s)
- **Fit grades:** {'good': 3, 'unknown': 1}

**Concurrency posture:** Test at 2 concurrent; mind unified-memory pressure.

_Sources:_ `runs/scotts-macbook-air/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### x1-370

(24 logical)

- **Models benchmarked:** 22 (ok: 1, errors: 21)
- **Median tps (non-embedding, ok):** 5.2
- **Fastest:** minicpm5-1b-agentic-tooluse (5.2 tok/s)
- **Slowest:** minicpm5-1b-agentic-tooluse (5.2 tok/s)
- **Errors/crashes:** text-embedding-nomic-embed-text-v1.5; refinedtoolcallv5-3b; vibethinker-3b-hermes; orinth-1.0-9b; qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled; ornith-1.0-35b
- **Fit grades:** {'good': 20, 'unknown': 2}

**Concurrency posture:** Handles several models at once with decent concurrency. refinedtoolcallv5-3b, orinth-1.0-35b, and qwen3.5-0.8b run concurrently decently well; throughput drops to ~2-8 tok/s under concurrent load. Good for multiplexing small models.
**Constraints:** Numbers taken while it was also the orchestrator are depressed; re-run solo for clean figures. Tons of co-resident services compete for RAM/CPU.

_Sources:_ `runs/x1-370/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`

### xwing

(24 logical)

- **Models benchmarked:** 6 (ok: 1, errors: 5)
- **Median tps (non-embedding, ok):** 8.7
- **Fastest:** vibethinker-3b-hermes (8.7 tok/s)
- **Slowest:** vibethinker-3b-hermes (8.7 tok/s)
- **Errors/crashes:** orinth-1.0-9b; ornith-1.0-35b; qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled; qwen3.6-14b-a3b-vibeforged-v2; text-embedding-nomic-embed-text-v1.5
- **Fit grades:** {'good': 5, 'unknown': 1}

**Concurrency posture:** Small models benefit from concurrency (more sessions, acceptable latency). Large models (e.g. Hermes-class) degrade badly at 2 concurrent sessions (response times blow up / fail). Test at 2, prefer 1 for big models.
**Constraints:** Big-model concurrency is the danger zone here.

_Sources:_ `runs/xwing/machine_synopsis.md`, `run_summary.csv`, `model_fit.csv`
