# Offline Batch Workflow for Benchmark Preparation

The benchmark campaign should use idle compute across the fleet for preparation and post-processing while the HX-370/eGPU host is reserved for controlled inference runs.

## Goal

Move CPU-heavy, I/O-heavy, transform-heavy and analysis-heavy work away from the benchmark host whenever doing so does not change the model artifact or invalidate reproducibility.

Useful offline jobs include:

- model inventory and SHA-256 hashing;
- GGUF/HF metadata extraction;
- conversion from HF/safetensors to GGUF;
- quantization candidate generation;
- calibration-corpus assembly;
- activation/router capture when a relief host can fit the source model;
- pruning candidate ranking and mask generation;
- quality-evaluation preparation;
- result packaging, compression and hashing;
- normalized result aggregation and Pareto-table preparation.

## Queue manifest

`benchmarks/offline_batch_jobs.tsv` is the canonical initial queue. Each row records stage, job type, preferred host, resource class, dependency and output class.

The queue runner is resumable through marker files under `STATE_ROOT`.

Dry-run:

```bash
HOST=$(hostname -s) DRY_RUN=1 \
  bash scripts/run_offline_batch_queue.sh
```

Execute jobs supported directly by the runner:

```bash
HOST=$(hostname -s) DRY_RUN=0 \
NAS_ROOT=/actual/nas/models \
LLAMA_BUILD=$HOME/src/llama.cpp/build-rocm-maxout \
  bash scripts/run_offline_batch_queue.sh
```

## Host-role guidance

The exact host names and capacities must be captured live; use these roles rather than assuming static capacity.

### Benchmark host

Reserve for:

- ROCm/eGPU kernel/runtime testing;
- controlled llama.cpp PP/TG/server runs;
- hybrid UMA/OCuLink memory-placement sweeps;
- GPU-specific activation capture that cannot run elsewhere;
- final validation of transformed/quantized artifacts.

Avoid running quantization, hashing, tar/compression, large NAS scans or result aggregation while collecting performance numbers.

### High-memory/modern relief host

Good for:

- model conversion/quantization when RAM allows;
- activation capture for models that fit;
- calibration and quality jobs;
- artifact verification;
- stateless services evacuated from the benchmark host.

### General-purpose/older relief host

Good for:

- hashing and metadata extraction;
- NAS model inventory;
- compression/result packaging;
- result analysis;
- calibration-corpus construction;
- low-throughput services moved off the benchmark host.

## Quantization jobs

Quantization is a natural background pipeline because it can be prepared before the eGPU benchmark slot. For each source checkpoint produce only deliberately selected candidates rather than every possible quant.

Suggested ladder:

```text
quality anchor
balanced
extreme fit
activation-guided mixed quant
```

Every transform job must preserve:

- source SHA;
- conversion tool SHA;
- quantizer/tool SHA;
- full command;
- output SHA;
- output size and average BPW where available;
- logs/warnings.

Output should land on durable NAS storage, not consume permanent SSD benchmark space. SSD staging remains one artifact at a time.

## Activation and pruning jobs

Activation capture can run independently of llama.cpp performance benchmarking. It should use the pinned calibration corpus and source checkpoint, write summaries/activation artifacts to NAS, then feed pruning/mixed-precision candidate generation.

Do not perform destructive checkpoint mutation from raw activation ranking alone. Produce an ablation/precision manifest first.

## Work stealing / scheduling

A simple operating model is:

```text
NAS queue
  -> relief host picks CPU/I/O/transform work
  -> durable result written back to NAS
  -> benchmark host stages one ready artifact to SSD
  -> GPU benchmark runs in controlled state
  -> evidence copied/packaged
  -> SSD artifact cleaned
  -> next ready artifact staged
```

The benchmark host should never wait for a SHA, conversion or quantization job that could already be running elsewhere.

## Resource-aware admission

Before a batch job begins, record enough capacity to avoid destabilizing the relief host:

- `MemAvailable`;
- swap pressure;
- free disk space at input/output paths;
- current Docker/service pressure;
- expected source/output artifact sizes.

Do not schedule a 90 GiB conversion onto a 32 GiB host unless the transform is explicitly streaming/out-of-core and known to be safe.

## Nice/ionice controls

Preparation jobs that share a relief host with services should run at reduced priority where appropriate:

```bash
nice -n 10 ionice -c2 -n7 <command>
```

This is especially useful for hashing, compression, NAS scans and quantization that can saturate storage or CPU for long periods.

## Reproducibility

Batch preparation is allowed to happen concurrently, but performance measurements must not. Only artifacts whose SHA and transform provenance are complete are admitted to the controlled benchmark matrix.

## Future extension

The queue format is intentionally simple so it can later be driven by an agent or converted to a Redis/Celery, systemd-run, Slurm-like, or existing fleet workload scheduler without changing the experiment data model.
