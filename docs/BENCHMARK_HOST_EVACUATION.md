# Benchmark Host Evacuation and Restore

Large-model benchmarking on the HX-370 host may require temporarily reclaiming RAM, swap headroom, CPU time, storage I/O, ports, and GPU resources consumed by normal services. This is part of the benchmark protocol, not an ad-hoc maintenance task.

## Principle

Never compare a maximum-fit model run against a normal-load run without recording the difference in host pressure. Every benchmark evidence set should state whether the host was:

- `normal-load` — normal production/service workload remains active;
- `quiesced` — safe local services stopped but not migrated;
- `evacuated` — selected services moved to another host and source copies stopped after validation;
- `bare-benchmark` — only required OS/network/storage/benchmark services remain.

These are separate benchmark loadouts.

## Candidate relief hosts

The experiment anticipates other machines may temporarily carry workloads. Do not encode capacity assumptions permanently; capture live capacity before every evacuation window.

Useful roles include:

- a newer Ryzen host with enough CPU/RAM for stateless APIs, workers, lightweight databases, ingress, and agent services;
- an older but capable general-purpose host for low-throughput infrastructure, mirrors, queues, dashboards, DNS/proxy, and selected stateful replicas;
- any already-running service mirror, provided its data freshness and promotion semantics are understood.

A relief host does not need to run inference efficiently to free the HX-370 for benchmarking.

## Phase 1: capture source pressure

```bash
bash scripts/capture_benchmark_host_pressure.sh results/pre-evacuation
```

Preserve process RSS, systemd services, Docker containers/stats, ports, mounts, disk usage, swap, AMD-SMI/ROCm state, and Neo4j inventory.

Record at minimum:

- `MemAvailable`;
- swap in use;
- Docker aggregate/container memory;
- top processes by RSS;
- any process using ROCm/GPU resources;
- disk/network-heavy background jobs;
- services that listen on externally used ports.

## Phase 2: preflight relief hosts

```bash
OUT=results/relief-preflight \
  bash scripts/preflight_benchmark_relief_hosts.sh <host-a> <host-b>
```

This is read-only. It checks SSH reachability, memory/swap/disk, Docker inventory, GPU inventory where relevant, running services/ports, and Neo4j state.

Do not migrate a service to a host that cannot comfortably absorb its steady-state memory plus burst headroom.

## Classification

Every source service/container belongs to one class before mutation:

### `must-stay`

Critical to host access or benchmark execution, for example networking, SSH/Tailscale path, required mounts, storage access, monitoring needed by the experiment, or device services.

### `stop-only`

Safe to stop temporarily and recreate locally later. Prefer this for disposable caches, batch workers, periodic jobs, indexing jobs, local dashboards, duplicate inference servers, or development-only containers.

### `migratable`

Stateless or externally backed service that can be started on a relief host, health-checked, and then stopped on the source.

### `stateful-migrate`

Service with persistent data that can move only after storage/data synchronization and application-level validation.

### `must-review`

Anything whose ownership, persistence, dependency chain, or failover semantics are uncertain. Neo4j belongs here until the exact primary/mirror role is proven.

Use `benchmarks/benchmark_service_evacuation.tsv` as the manifest.

## Stateful-service rule

Never use `docker stop`, `docker rm`, rsync, or service shutdown as a substitute for understanding the data role.

For Neo4j specifically, establish:

1. which instance is authoritative;
2. whether the other instance is a true replica, backup target, delayed mirror, or unrelated database;
3. edition/version compatibility;
4. last successful backup/snapshot;
5. database names and store IDs where applicable;
6. a read query/count/checksum or other validation that can demonstrate the promoted service contains the expected data;
7. rollback procedure.

Do not promote a loosely synchronized "mirror-like" Neo4j instance solely to free benchmark RAM.

## Safe migration sequence

For a migratable service:

1. capture source configuration/image/tag/env/volumes/networks;
2. confirm relief-host capacity and port availability;
3. start the target copy without stopping source;
4. perform target health checks;
5. if stateful, validate data/application state;
6. redirect traffic or confirm clients can reach target;
7. stop source copy;
8. capture post-evacuation pressure;
9. run benchmark;
10. restore source or deliberately retain target placement;
11. validate restored service;
12. capture final pressure and note any changed topology.

## Docker considerations

Before moving a container, preserve `docker inspect`, image digest/tag, compose labels, networks, bind mounts, named volumes, restart policy, capabilities/devices, health check, secrets/env-file source, and exposed ports.

Prefer redeploying from the repository/Compose source of truth rather than copying a live container filesystem.

If Compose project files cannot be located, classify the container `must-review` until its configuration can be reconstructed safely.

## Benchmark admission thresholds

For very large models, define target free-memory tiers rather than blindly stopping everything. Suggested experiment states:

```text
Tier A  normal-load baseline
Tier B  >= 32 GiB MemAvailable
Tier C  >= 48 GiB MemAvailable
Tier D  >= 64 GiB MemAvailable
Tier E  maximum practical / bare-benchmark
```

The agent should stop escalating once the target model/context has adequate headroom. A 58 GiB GLM does not require the same evacuation level as an 82 GiB DeepSeek Flash quant.

## What should move first

Favor maximum reclaimed resources per unit of migration risk:

1. duplicate/local model servers and inference containers;
2. stateless APIs, workers and scheduled agents;
3. indexing/batch/ingest processes;
4. dashboards, search frontends, web applications and proxies with easy redeploy;
5. caches/queues when state loss is acceptable or persistence is external;
6. stateful applications with verified replication/backup;
7. core databases only as a last resort.

This order keeps the benchmark host available while minimizing chances of damaging production data.

## Restore evidence

The experiment is not complete until evacuated services are accounted for. Preserve a restoration ledger with:

- service;
- original host;
- temporary host;
- image/version/config revision;
- validation before cutover;
- source stop time;
- target stop time;
- source restore time;
- final health status;
- any intentional permanent migration.

## Benchmark metadata

Every result directory should include the host state (`normal-load`, `quiesced`, `evacuated`, or `bare-benchmark`) and the corresponding pressure snapshot. This lets later analysis distinguish model/runtime gains from simply having 20 GiB more free memory or eliminating background CPU/I/O contention.
