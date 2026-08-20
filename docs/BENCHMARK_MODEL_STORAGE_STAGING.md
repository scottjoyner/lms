# Benchmark Model Storage and SSD Staging

Large model artifacts should have one canonical storage location and one disposable benchmark staging location.

## Policy

- NAS: durable source of truth for GGUF/safetensors/model metadata.
- Local SSD: temporary performance staging area for the artifact currently under test.
- Do not accumulate every 60-100+ GiB quant on SSD.
- Never delete the NAS source as part of benchmark cleanup.
- Every staged artifact is content-addressed by SHA-256 prefix and verified before use.

## Preflight

Before downloading, copying or benchmarking large weights:

```bash
NAS_ROOT=/actual/nas/model/root \
SSD_ROOT=/actual/ssd/benchmark-stage \
  bash scripts/benchmark_storage_preflight.sh
```

The evidence captures `df -hT`, byte-accurate `df`, mounts, block devices, the source/stage filesystems and large model files already occupying the SSD staging directory.

## Stage one model

```bash
SOURCE=/actual/nas/path/model.gguf \
STAGE_ROOT=/actual/ssd/benchmark-stage \
MIN_FREE_GIB=40 \
  bash scripts/stage_model_for_benchmark.sh
```

The staging script requires:

`available SSD bytes >= model bytes + reserve bytes`

The reserve defaults to 40 GiB and should be raised for systems where benchmark results, temporary files, swap, container layers or other workloads share the filesystem.

The script:

1. captures `df` for source and destination;
2. measures source file size;
3. calculates required headroom before copying;
4. hashes the NAS source;
5. copies with resumable `rsync` when available;
6. verifies staged SHA-256;
7. records copy duration/filesystem information;
8. emits the exact staged path;
9. generates a cleanup script unless `KEEP_AFTER=1`.

## Why benchmark from SSD

NAS-backed model loading can confound model/runtime comparison with network/filesystem throughput, especially during cold load, mmap page faults and repeated startup. The main inference benchmark should therefore use the SSD-staged artifact when space permits.

NAS direct-load remains a useful separate control:

- cold startup from NAS;
- cold startup from SSD;
- warm/page-cached startup;
- `mmap` vs `--load-mode dio`;
- steady-state TG/PP after load.

Do not attribute a faster load path to model inference performance.

## One-at-a-time staging order

For a constrained SSD, stage candidates in experiment order, remove them after their evidence package is complete, and stage the next artifact. Suggested large-model sequence:

1. GLM-4.5-Air heavy quant;
2. Laguna S 2.1;
3. DeepSeek-V4-Flash IQ2;
4. alternate GLM quant;
5. boundary DeepSeek quant;
6. other large research candidates.

Smaller 20-40 GiB models can remain cached only if the live `df` reserve remains above policy.

## Storage admission rule

No benchmark runner should automatically copy/download a model merely because its nominal size fits. Admission requires all of:

- NAS/source file exists and is readable;
- SSD mount exists and is the expected filesystem/device;
- post-stage reserve target remains satisfied;
- source SHA is known;
- staging target is not a production-data path;
- benchmark output directory also has adequate space.

If any condition fails, return `deferred-storage` rather than filling the disk.

## Evidence retention

Raw benchmark results are small relative to weights and should remain on SSD until packaged/committed or copied to durable storage. The large staged GGUF itself is disposable because the canonical source remains on NAS.

Record model SHA in every evidence package so a later rerun can prove it used the same bytes even after the staging copy was removed.
