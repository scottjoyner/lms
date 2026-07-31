# Fleet loadout optimizer

`lms-fleet` converts machine observations into benchmark candidates and benchmark evidence into a desired loadout. `lms-fleet-bench` safely executes selected candidates through the existing LMS deterministic task suite. Neither command deploys or admits a production runtime.

## State boundary

1. **Observation** records what Linux or macOS reports now, including CPU, memory, GPU, Vulkan, CUDA, ROCm, Metal, and AMD XDNA/NPU probes.
2. **Plan** produces loopback-only candidates with explicit context, slots, backend, memory estimate, and candidate ID.
3. **Execution** starts one selected candidate as an ephemeral loopback-only process, or targets an explicitly mapped existing endpoint, then runs readiness, streaming, concurrency, optional cancellation, resource-headroom, crash, and LMS task-suite checks.
4. **Evidence** is written to `loadout_results.csv`, keyed by candidate ID.
5. **Selection** ranks eligible candidates but always emits `admission.admitted=false`.

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

# Preview one candidate's llama-server command without starting it.
lms-fleet-bench \
  --plan runs/node/benchmark_plan.json \
  --candidate <candidate-id> \
  --suite-file benchmarks/agent_skill_suite.v1.json \
  --output-dir runs/node/execution \
  --llama-server-bin /path/to/llama-server \
  --dry-run

# Execute selected candidate IDs. The command intentionally requires --candidate
# or --all so an expensive plan cannot start accidentally.
lms-fleet-bench \
  --plan runs/node/benchmark_plan.json \
  --candidate <candidate-id> \
  --suite-file benchmarks/agent_skill_suite.v1.json \
  --output-dir runs/node/execution \
  --llama-server-bin /path/to/llama-server \
  --test-cancellation

# Benchmark an already-running NPU or other adapter endpoint.
lms-fleet-bench \
  --plan runs/node/benchmark_plan.json \
  --candidate <npu-candidate-id> \
  --endpoint-map <npu-candidate-id>=http://127.0.0.1:1236/v1 \
  --suite-file benchmarks/agent_skill_suite.v1.json \
  --output-dir runs/node/execution

lms-fleet select \
  --plan runs/node/benchmark_plan.json \
  --results-csv runs/node/execution/loadout_results.csv \
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

## Execution behavior

For a llama.cpp candidate, `lms-fleet-bench`:

- inspects `llama-server --help` and renders only supported command-line flags;
- binds the ephemeral runtime to `127.0.0.1` on the candidate benchmark port;
- waits for `/v1/models` and records readiness evidence;
- checks OpenAI-compatible streaming;
- sends `parallel_slots + 1` simultaneous canaries to detect crashes or unsafe concurrency behavior;
- optionally closes a streaming request early and runs a post-cancellation canary;
- samples process RSS and minimum system-memory headroom on Linux or macOS;
- executes `agent_skill_suite.v1.json` through the existing LMS benchmark runner;
- records server logs, suite logs, raw task outputs, gate evidence, and one aggregate result row;
- terminates the complete process group and cools down before the next candidate.

Non-llama runtimes require an explicit `--endpoint-map CANDIDATE_ID=URL`; the executor never guesses or exposes a network endpoint.

## Benchmark result contract

The loadout result CSV includes:

```csv
candidate_id,engine,backend,model_id,base_url,ok_rate,eval_ok_rate,eval_score_avg,tps_med,ttft_med,memory_peak_bytes,memory_headroom_ratio,concurrency_ok,streaming_ok,cancellation_ok,crash_count,benchmark_exit_code,error,candidate_dir
```

A candidate is ineligible when it crashes, cannot respect its concurrency limit, drops below 98% request success, fails streaming, or leaves less than 10% system-memory headroom. Raw speed cannot override those gates. Cancellation is recorded separately because some helper runtimes may be intentionally non-cancellable but must remain non-admitted until policy explicitly allows that role.

## Recommended fleet workflow

Run `discover` locally on every relevant Linux host and the MacBook Air. Keep observations outside desired-state profiles until reviewed. Generate a constrained plan, dry-run the intended candidates, execute them one at a time, select from the resulting evidence, and import the selected artifact into `fleet-llm-profiles`. Rendering persistent launchers belongs downstream; live admission remains external.
