# R9700 / Ternary Bonsai 2 physical qualification

This lane turns the already accepted R9700/Bonsai runtime into one exact
`loadout_qualification.v1` evidence bundle. It does **not** start, stop, reload,
admit, or reroute a model.

## Accepted physical baseline

The preflight is intentionally tied to the physical evidence already retained
for the R9700 lane:

- node: `x1-370`
- GPU architecture: `gfx1201`
- runtime: `PrismML-Eng/llama.cpp@9a9394a895b96003ca842a6041cb28ac49a108f7`
- runtime tag: `prism-b10709-9a9394a`
- exact llama-server SHA-256:
  `e5c4211999de5b789b980626ad4b35f17d9b75f9a45c6d8b1801ff41f9162c88`
- model: `Ternary-Bonsai-2-27B-PQ2_0`
- model revision: `6ed5e12bf84b7a63069882c91dd9e9218647d17b`
- exact model SHA-256:
  `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`

The preflight also requires the live process to use the accepted full-GPU
`--gpu-layers 999` and `--flash-attn on` loadout on a loopback endpoint.

## One-command lane

Run this from the clean reviewed LMS checkout on x1-370. Substitute the exact
LMS PR head and exact clean Hermes head being reviewed:

```bash
PYTHONPATH="$PWD/src" python -m lms_agent_bench.r9700_bonsai2_qualification \
  --lms-repo "$PWD" \
  --lms-branch agent/qualification-decision-metrics \
  --lms-commit '<EXACT_PR_HEAD>' \
  --hermes-repo "$HOME/git/hermes-agent" \
  --hermes-branch main \
  --hermes-commit '<EXACT_HERMES_HEAD>' \
  --output-dir "$HOME/lms-qualification-prep/r9700-bonsai2-001" \
  --workspace "$HOME/lms-qualification-runs" \
  --run-id r9700-bonsai2-001 \
  --execute
```

Defaults match the retained acceptance layout:

- model:
  `~/.local/share/local-studio/experimental/bonsai2-r9700/models/Ternary-Bonsai-2-27B-PQ2_0.gguf`
- endpoint: `http://127.0.0.1:8000/v1`

Use `--model-artifact` or `--endpoint` only when the exact same accepted
artifact/runtime is exposed elsewhere. The SHA/build checks still apply.

## What preflight proves

Before the benchmark operator creates a run directory, the prep command rejects:

- the wrong host or missing `gfx1201`;
- model-byte drift;
- runtime-binary drift;
- a dirty or wrong LMS/Hermes source checkout;
- a non-loopback runtime;
- a different live model path/model ID;
- a different PrismML build;
- partial GPU offload;
- flash attention off;
- context disagreement between process argv and `/props`;
- slot disagreement between process argv and `/props`;
- speculative decoding.

It then writes:

```text
loadout.json
inventory.csv
throughput-cases.json
physical-preflight.json
```

The loadout fingerprint includes the observed runtime argv hash, current context,
current slot count, model bytes, runtime build, and physical baseline metadata.

## Execute/verify boundary

With `--execute`, the prep immediately calls the existing
`loadout_qualification_operator`. That operator remains responsible for:

1. source/model/loadout re-verification;
2. the three-trial reliability benchmark;
3. three-trial base Hermes qualification;
4. three-trial context-pressure qualification;
5. exact evidence binding;
6. postflight source/model/endpoint checks;
7. immutable run-manifest creation.

After a successful run the prep invokes manifest verification with
`--require-success`.

No step changes `admission.admitted=false`.

## Expected decision-metrics boundary

A successful PR #14-era bundle should contain `loadout_decision_metrics.v1`.
Native prompt-processing throughput, peak device memory, and peak host RSS remain
null until a later compatible telemetry collector is implemented. Do not fill
those values from the older standalone R9700 benchmark.
