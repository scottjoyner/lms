# Additional Competitors and Quantization Axes

The benchmark should remain focused enough to finish, but the current open-weight agent/coding landscape warrants a few additional controls when verified local artifacts are available.

## Competitor priority

### Tier 1: already core

- Ornith 1.0/1.5 variants
- Qwen3.5-35B-A3B and larger Qwen MoE controls
- Laguna S 2.1
- GLM heavy-quant variants
- DeepSeek-V4-Flash heavy quants

### Tier 2: high-value additions

- Kimi coding/agent family: useful large-MoE quality control against DeepSeek/GLM
- MiniMax coding/agent family: useful throughput/long-context control
- Qwen Coder/Next efficient variant: smaller capability-per-byte control
- GLM Flash-class efficient variant: latency/throughput control

Only add a row when the exact local artifact, model identity, quant and source are verified.

## Quantization ladder

Do not test only one quant per family. For at least the highest-value families, aim for a three-point ladder:

```text
quality anchor     Q5/Q6 or higher where feasible
balanced           Q4 / IQ4 / MXFP4-style
extreme-fit        IQ2/IQ3 or comparable aggressive quant
```

For MoE models, include expert-specific mixed quant recipes when available. Record average BPW and whether attention/non-expert tensors use a different precision from experts.

## What quantization changes

Treat quantization as more than model-file size. It can alter:

- GPU kernel choice;
- memory bandwidth demand;
- expert routing precision/quality;
- PP/TG ratio;
- ability to keep additional layers in dGPU VRAM;
- KV/context headroom;
- OCuLink transfer volume;
- startup/load time;
- coding/tool-use quality.

Therefore compare quants under matched runtime and memory-placement conditions before attributing differences to the model family.

## KV quantization is a separate axis

Model quantization and KV quantization must not be conflated. For a selected model quant, separately test:

```text
model Q4 + KV f16
model Q4 + KV q8_0
model Q4 + KV q4_0
```

This reveals whether context expansion is coming from model compression or cache compression and whether either causes quality/latency regressions.

## CPU-MoE and partial-offload interaction

For large MoE models, repeat the best model quants through `--n-cpu-moe` / `--cpu-moe`. An aggressive quant may allow more experts/layers to remain on the dGPU, changing the optimal host-placement point.

The final analysis should show quantization × placement, not quantization alone.

## Context-normalized quality

A smaller quant with 128K stable context may be more useful than a higher-quality quant that can only sustain 16K. Report quality and task success at matched actual filled context, not only at short prompts.

## Candidate selection rule

Promote at least one candidate in each category:

- best absolute quality;
- best quality-adjusted TG;
- best quality-adjusted PP;
- best max-context profile;
- best multi-agent throughput;
- best tokens/joule;
- best capability per GiB of resident model footprint.
