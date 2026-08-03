# Hermes intelligence and exact-loadout benchmark

A model is not qualified by its name, parameter count, or raw tokens per second. Qualification belongs to one exact model/runtime loadout and requires both repeated inference evidence and repeated Hermes agent-loop evidence.

## Exact loadout identity

Every benchmarked loadout must have a `model_loadout_manifest.v1` artifact containing:

- model ID, format, size, revision, and full content SHA-256;
- architecture kind: dense, MoE, hybrid MoE, recurrent, or other;
- total parameters and active parameters per token;
- MoE expert count, active experts per token, shared experts, and router type;
- weight quantization scheme, nominal/effective bits, group size, mixed precision, calibration, and importance-matrix use;
- inference engine, build/version, backend, GPU offload, tensor split, threads, batch, ubatch, flash attention, mmap, mlock, and extra arguments;
- configured and native context, RoPE scaling, sliding window, and prompt-pressure target;
- KV-cache key/value dtype and bits, placement, capacity, bytes per token, sharing, prefix reuse, persistence, key scope, and eviction policy;
- parallel slots and continuous batching;
- speculative decoding and exact draft-model identity when enabled.

The canonical fingerprint covers all of these settings. Benchmark results from one fingerprint cannot qualify another fingerprint. `candidate_id` is a human-readable review label and is deliberately excluded from the fingerprint, so relabeling a candidate does not create a false runtime identity.

## Weight quantization versus KV-cache quantization

Weight quantization and KV-cache quantization are independent variables.

Weight quantization can affect:

- factual and reasoning accuracy;
- structured-output reliability;
- tool selection;
- argument generation;
- error recovery;
- coding and patch quality.

KV-cache quantization can affect:

- long-context recall;
- preservation of tool state over many turns;
- correctness late in an agent loop;
- prompt-prefix reuse behavior;
- memory capacity and concurrency;
- TTFT and sustained throughput.

A result must therefore identify both the weight quant and the K/V cache dtypes. A label such as `Q4` is insufficient.

## Context matrix

Each viable model variant should be evaluated at controlled configured-context tiers such as:

```text
4K
8K
16K
32K
64K when supported and explicitly configured
```

For every tier, run both:

1. Raw inference measurements: cold TTFT, warm TTFT, prompt-processing rate, generation rate, memory, stability, and concurrency.
2. Hermes tasks under context pressure: graph retrieval, multistep tool use, file mutation, test execution, recovery, and a retained control fact.

Do not infer 32K agent reliability from a passing 4K run. Do not infer usable context from a model-card maximum.

Context beyond the model-native limit requires an explicit RoPE-scaling configuration in the loadout manifest. The scaled loadout is a distinct fingerprint.

## Cache measurements

Cache behavior requires separate trials:

- cold model and empty KV cache;
- warm model with empty request KV cache;
- exact-prefix reuse hit;
- near-prefix miss;
- concurrent slots with independent caches;
- shared/global cache when supported;
- eviction and post-eviction behavior;
- persistent cache restart behavior when enabled.

Record:

- cache hit/miss classification;
- prompt tokens avoided or reused;
- TTFT and prompt-processing deltas;
- resident-memory change;
- cross-request contamination checks;
- cache-key scope;
- whether model, quant, tokenizer, system prompt, tools, and runtime build are included in the cache identity.

A shared cache must never be reused across incompatible model hashes, tokenizer identities, quantizations, KV dtypes, tool schemas, or system prompts.

## Dense versus MoE

Absolute task success remains the qualification gate for both architectures.

Dense models report:

```text
total parameters = active parameters per token
```

MoE models report separately:

```text
total parameters
active parameters per token
active / total parameter ratio
total experts
active experts per token
shared experts
router type
```

Comparison output should include:

- successful Hermes tasks per hour;
- successful effect checkpoints per minute;
- generation and prompt-processing throughput;
- memory consumed;
- tasks per hour per active billion parameters;
- tasks per hour per total model GiB;
- task pass rate and P0/recovery pass rate.

Normalized metrics explain efficiency but never compensate for failed tasks, unsafe calls, or malformed tool arguments.

## Hermes intelligence suite

The deterministic suite scores observable effects rather than one golden chain of thought or one exact tool sequence. It covers:

- Neo4j/MCP retrieval;
- multihop dependency reasoning;
- file read/edit/verify workflows;
- graph-to-file workflows;
- transient MCP failure recovery;
- distractor-tool avoidance;
- code repair followed by tests;
- read-only graph boundaries.

Each case runs at least three valid trials. Qualification requires:

```text
overall task pass rate >= 80%
effect checkpoint rate >= 90%
MCP argument validity >= 95%
prohibited tool calls = 0
timeout or crash rate = 0%
all P0 cases repeatably pass
all recovery cases repeatably pass
```

The output remains non-admitted.

## Controlled matrix generation

Architecture and weight-quant variants require separate base manifests with different exact model hashes. They cannot be created by changing a label on one artifact.

Runtime axes may then vary in a controlled matrix:

```text
context.configured_tokens
kv_cache.capacity_tokens
kv_cache.key_dtype
kv_cache.value_dtype
concurrency.parallel_slots
runtime.flash_attention
runtime.batch_size
runtime.ubatch_size
runtime.gpu_layers
```

Generate the matrix with:

```bash
lms-loadout-matrix matrix \
  --bases examples/model-loadouts.v1.example.json \
  --axes examples/model-loadout-matrix.axes.example.json \
  --out rollout/model-loadout-matrix.json
```

Each generated candidate has a unique loadout fingerprint.

## Physical Hermes run

After selecting one exact loadout and starting its loopback endpoint:

```bash
lms-hermes-bench run \
  --loadout /path/to/exact-loadout.json \
  --hermes-repo /path/to/hermes-agent \
  --hermes-python /path/to/hermes-agent/.venv/bin/python \
  --endpoint http://127.0.0.1:1234/v1 \
  --workspace ~/lms-hermes-runs \
  --trials 3 \
  --out ~/lms-hermes-runs/result.json
```

Run the dedicated context-pressure suite against the same exact loadout:

```bash
lms-hermes-bench run \
  --loadout /path/to/exact-loadout.json \
  --hermes-repo /path/to/hermes-agent \
  --hermes-python /path/to/hermes-agent/.venv/bin/python \
  --endpoint http://127.0.0.1:1234/v1 \
  --suite src/lms_agent_bench/benchmarks/hermes_agent_context_suite.v1.json \
  --workspace ~/lms-hermes-context-runs \
  --trials 3 \
  --out ~/lms-hermes-context-runs/result.json
```

The runner creates an isolated Hermes home, a deterministic stdio MCP server, and a fresh workspace for every trial. It records the final response, message/tool trace, fixture calls, file effects, errors, latency, usage when exposed, and useful-work throughput.

Validate each retained report with:

```bash
lms-hermes-bench gate \
  --report ~/lms-hermes-runs/result.json \
  --node-id NODE \
  --candidate-id CANDIDATE \
  --model MODEL_ID \
  --model-content-sha256 SHA256 \
  --out ~/lms-hermes-runs/intelligence-gate.json
```

## Comparing loadouts

Compare two or more completed reports with:

```bash
lms-loadout-compare \
  --report /path/to/report-a.json \
  --report /path/to/report-b.json \
  --out rollout/loadout-comparison.json
```

Only a same-model, same-node, same-suite comparison with one changed dimension is marked as a controlled single-axis comparison. Dense-versus-MoE and cross-quant comparisons are observational because they necessarily change the model artifact. Quality and speed remain separate fields; the comparison command does not average them into one score.

## Promotion rule

A loadout is eligible for profile-import review only when the same loadout fingerprint has:

1. a passing repeated throughput/reliability artifact;
2. a passing base Hermes intelligence artifact;
3. a passing context-pressure Hermes intelligence artifact;
4. matching node, model ID, full model hash, runtime configuration, context configuration, KV-cache configuration, quantization, and architecture metadata.

Neither artifact admits or routes the runtime. Live identity, health, shared capacity, path behavior, freshness, and rollback remain external gates.
