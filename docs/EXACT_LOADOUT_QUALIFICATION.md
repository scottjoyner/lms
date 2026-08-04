# Exact-loadout qualification

`lms-loadout-qualify` closes the evidence-linking gap between repeated
throughput/reliability measurement and the two Hermes intelligence suites.

A loadout is qualified only when all three evidence classes belong to the same
immutable `loadout_fingerprint`:

1. repeated throughput and reliability;
2. base Hermes MCP agent intelligence;
3. Hermes intelligence under configured-context pressure.

The result remains non-admitted and does not change routing.

## 1. Bind reliable throughput to the exact loadout

The reliable benchmark report predates the exact-loadout manifest. The binder
therefore verifies and wraps it rather than trusting an operator to associate
files by name.

```bash
lms-loadout-qualify bind-throughput \
  --loadout loadout.json \
  --reliability reliable-output/reliability.json \
  --out throughput-evidence.json
```

The binder rejects the report unless:

- `reliable_benchmark.v1` and `benchmark_reliability` match;
- the reliability fingerprint recomputes exactly;
- the report and its only summary passed;
- at least three valid trials exist;
- the summary node equals the loadout node;
- the summary model equals the exact loadout model ID;
- the benchmark endpoint is loopback-local;
- the report remains non-admitted.

The resulting `loadout_throughput_evidence.v1` artifact embeds the complete
validated loadout and records the reliability artifact SHA-256.

## 2. Produce the combined qualification

Run the base and context-pressure Hermes suites against the same loadout, then:

```bash
lms-loadout-qualify qualify \
  --loadout loadout.json \
  --throughput throughput-evidence.json \
  --base-hermes hermes-base.json \
  --context-hermes hermes-context.json \
  --out loadout-qualification.json
```

The command independently verifies both Hermes benchmark fingerprints, suite
identities, passed gates, non-dry-run execution, loopback endpoint, model,
candidate, node, embedded loadout, and exact loadout fingerprint.

It rejects substitution of the base suite for the context suite and rejects any
change to context, KV quantization, weight quantization, runtime build,
concurrency, model hash, or another fingerprinted loadout field.

## 3. Verify before import

```bash
lms-loadout-qualify verify \
  --qualification loadout-qualification.json \
  --loadout loadout.json
```

A successful `loadout_qualification.v1` contains cryptographic references to:

- the throughput-evidence fingerprint;
- the underlying reliability fingerprint;
- the base Hermes benchmark and suite fingerprints;
- the context-pressure Hermes benchmark and suite fingerprints;
- the SHA-256 of each input artifact used by the qualification command.

Every gate is explicit, and the artifact always contains:

```json
{
  "qualified": true,
  "admission": {"admitted": false}
}
```

## Remaining physical boundary

This command can validate only evidence that exists. It does not execute a
physical model, infer runtime settings, restore KV state, or promote a profile.
The fleet rollout must still produce the reliable and Hermes reports from the
reviewed exact loadout on the real node.
