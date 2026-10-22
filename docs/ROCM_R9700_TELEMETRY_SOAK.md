# ROCm R9700 Telemetry and Soak Layer

This layer exists to prevent misleading optimization conclusions. A configuration is not considered superior merely because it reports a higher short-run token rate.

## What is recorded continuously

During every `llama-server` case the harness samples once per second by default:

- complete `amd-smi metric --json` payload;
- `llama-server` process RSS;
- `llama-server` virtual memory size;
- UTC timestamp and PID.

The raw JSONL is preserved in `telemetry/`. A derived summary records RSS start/end/delta/max and aggregates every numeric AMD-SMI metric with mean/p50/p95/max. When a soak case supplies output token count, the analyzer also attempts a tokens-per-joule estimate when a usable power metric is exposed by the local `amd-smi` schema.

Do not delete the raw telemetry when the analyzer cannot recognize a power field; the raw payload remains authoritative.

## Cold versus warm prefix

Every normal server configuration is probed twice:

- `warm`: identical prefix across requests, suitable for repeated agent/system-prompt cache behavior;
- `cold`: unique nonce appended per request, forcing a distinct prompt prefix.

Compare TTFT p50/p95/p99 and end-to-end wall latency between these modes. This is not a claim of engine-level persistent KV reuse; it is an empirical cache-state/control comparison under the exact runtime.

## Tail latency

The server probe records p50/p95/p99 plus max for:

- TTFT;
- total wall time;
- per-request decode t/s.

At `np=2` and `np=4`, aggregate throughput is not sufficient for promotion. Reject configurations whose p95/p99 latency or fairness is operationally unacceptable even if aggregate tokens/s is high.

## Soak execution

Normal experiment:

```bash
bash scripts/run_rocm_r9700_maxout.sh
```

Add long-running stability workload:

```bash
SOAK=1 bash scripts/run_rocm_r9700_maxout.sh
```

Deep + soak includes the four-slot 64K case:

```bash
SOAK=1 DEEP=1 bash scripts/run_rocm_r9700_maxout.sh
```

The default soak request multiplier is 20 requests per slot and can be increased:

```bash
SOAK=1 SOAK_REQUEST_MULTIPLIER=60 bash scripts/run_rocm_r9700_maxout.sh
```

Longer soak runs are preferred for final promotion. The execution agent should target enough duration to reach thermal steady state rather than assuming a fixed wall-clock duration across models with very different speeds.

## Memory-growth check

For each soak case calculate at minimum:

- RSS at start;
- RSS at end;
- RSS delta;
- max RSS;
- VRAM/memory metrics from AMD-SMI if exposed;
- request count and successful completion count.

A persistent monotonic host-RSS increase should be flagged for manual review. Re-run the same loadout after process restart to distinguish expected allocator high-water behavior from unbounded growth.

## Thermal and power interpretation

A short burst may run above sustainable clocks. For promoted profiles, compare early and late telemetry and look for:

- rising edge/hotspot temperature;
- falling clocks;
- power-limit behavior;
- throughput decay;
- fan/thermal saturation when reported.

Do not compare two power-tuned states under the same loadout ID. Stock, undervolted, overclocked, or changed power-limit states must be separate identities.

## Speculation sweep

When `DRAFT_MODEL` is supplied the runner tests draft lengths 2, 4, 8, and 16. When MTP is enabled and the registry allows it, the same draft-length sweep is attempted for MTP.

For each speculative case retain:

- identical non-speculative baseline;
- server metrics and logs;
- output throughput;
- TTFT/tail latency;
- VRAM/power telemetry;
- any available draft/accepted token counters.

The winning speculative profile is the configuration with the strongest net end-to-end benefit after memory/context and power costs, not the largest acceptance percentage in isolation.

## Filled-context follow-up

The main context sweep currently configures the server at 8K/32K/64K/128K/256K and uses deterministic long prompts. During physical execution, the agent should additionally retain actual prompt-token counts from probe usage so analysis can distinguish configured context from filled context. If a target tier is materially under-filled, perform an explicit filled-context run before calling decode degradation at that tier measured.

## Promotion gates

A recommended production profile should have evidence for:

1. steady-state generation throughput;
2. cold and warm TTFT;
3. p95/p99 latency;
4. RSS/VRAM stability;
5. thermal stability;
6. power efficiency where measurable;
7. context headroom;
8. quality gate;
9. speculation benefit if enabled.

Keep separate winners for interactive, long-context, multi-agent throughput, quality-max, speculative and efficiency profiles.
