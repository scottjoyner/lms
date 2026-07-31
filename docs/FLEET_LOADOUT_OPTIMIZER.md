# Fleet loadout optimizer

`lms-fleet` converts machine observations into benchmark candidates and benchmark evidence into a desired loadout. It is intentionally not a deployment or admission tool.

## State boundary

1. **Observation** records what Linux or macOS reports now, including CPU, memory, GPU, Vulkan, CUDA, ROCm, Metal, and AMD XDNA/NPU probes.
2. **Plan** produces loopback-only candidates with explicit context, slots, backend, memory estimate, and candidate ID.
3. **Evidence** is a CSV row tied to that candidate ID.
4. **Selection** ranks eligible candidates but always emits `admission.admitted=false`.

The live router or reconciliation authority must independently verify physical runtime identity, model fingerprint, access paths, shared slot capacity, freshness, and rollback before routing traffic.

## Commands

```bash
lms-fleet discover \
  --endpoint http://127.0.0.1:1234/v1 \
  --endpoint http://127.0.0.1:1236/v1 \
  --out runs/node/machine_observation.json

lms-fleet plan \
  --observation runs/node/machine_observation.json \
  --models models.json \
  --contexts 4096,8192,16384,32768 \
  --out runs/node/benchmark_plan.json

lms-fleet select \
  --plan runs/node/benchmark_plan.json \
  --results-csv runs/node/loadout_results.csv \
  --out runs/node/selected_loadout.json

lms-fleet inventory \
  --tailscale-csv tailscale-devices.csv \
  --out inventory/tailscale-nodes.json
```

Example model inventory:

```json
{
  "models": [
    {
      "id": "qwen3.5-9b-q4_k_m",
      "path": "/models/qwen3.5-9b-q4_k_m.gguf",
      "parameter_billions": 9,
      "quantization": "Q4_K_M",
      "max_context": 32768
    }
  ]
}
```

## Benchmark result contract

The loadout result CSV must include `candidate_id` plus:

```csv
candidate_id,ok_rate,eval_score_avg,eval_ok_rate,tps_med,ttft_med,memory_peak_bytes,memory_headroom_ratio,concurrency_ok,streaming_ok,cancellation_ok,crash_count
```

A candidate is ineligible when it crashes, cannot respect its concurrency limit, drops below 98% request success, fails streaming, or leaves less than 10% memory headroom. Raw speed cannot override those gates.

## Recommended fleet workflow

Run `discover` locally on every relevant Linux host and the MacBook Air. Keep the observations outside desired-state profiles until reviewed. Run `plan` on the same physical machine, execute each candidate through the existing LMS task suite, then use `select` to generate the profile input consumed by `fleet-llm-profiles`. Rendering launchers belongs downstream; live admission remains external.
