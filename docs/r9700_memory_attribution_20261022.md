# Joyner R9700 — Read-only memory attribution (2026-10-22)

## Purpose

Before an IQ3_S Strata benchmark, Joyner needs an approved 64 GiB RAM upgrade and **at least 8 GiB MemAvailable**. The existing four llama.cpp servers must not be stopped or unloaded to make a benchmark appear admissible.

`scripts/r9700_memory_attribution.py` adds a deterministic, **read-only** report that inspects bounded kernel counters and cgroup-v2 hierarchy without changing RAM, cgroups, swap, services, GPU state, model weights or network routes. It refuses to report success when run anywhere other than `joyner` and always emits `model_launch_allowed=false` and `production_admission=NO_GO`.

### Reproduce with no installation

From x1-370, stream the script as input over the existing trusted SSH connection:

```bash
cat scripts/r9700_memory_attribution.py |
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes joyner python3 -
```

The optional `--proc-root` and `--cgroup-root` CLI flags support synthetic fixtures. They do not elevate privilege. Tests: `python -m pytest -q tests/test_r9700_memory_attribution.py`.

### First physical observation, October 22 morning

- Pinned Joyner OS RAM approximately **14.75 GiB installed, 2.93 GiB available**; swap approximately **2.87 GiB used**. Linux memory PSI avg60 (some) **0.00**.
- Kernel shared-memory counter `Shmem` approximately **2227.9 MiB**. The host `/dev/shm` filesystem contains virtually no data. Therefore the full kernel Shmem charge cannot be assumed to be named files visible in that mount.
- Cgroup root shared memory approximately **2227.9 MiB**. Visible `user.slice` aggregate approximately **637.4 MiB**. That nested charge is nearly entirely under `user-1001.slice` / `joyner-lane-1236.service` (~637.2 MiB).
- **Approximately 1554 MiB** remains after subtracting visible top-level child-group charges from the root aggregate. Its origin is **NOT ATTRIBUTED**, and hierarchical charges, root-level memory accounting, kernel state and accounting races require further investigation. This number must not be described as a memory leak or assigned to a single model without additional evidence.
- The four `llama-server` processes remain running; health checks for localhost ports 1234 and 1236 succeeded. No R9700 workload was started.

### Attribution rules and gates

Memory cgroup `memory.stat` values are **hierarchical**. Never sum a parent and its nested children; calculate only root minus *immediate* first-level children and label the difference unattributed. A shared-memory charge on a cgroup is **not equal to the PSS_Shmem or VmRSS of its main process**, and absent process maps / permissions must not be reported as zero. No cleanup or automatic remediation is justified by this snapshot.

The report fails closed for missing kernel counters, inconsistent memory/swap and missing PSI. It does not provide a `GO` path or benchmark authorization even when capacity later passes. The motherboard/RAM physical acceptance and independent model execution authorization remain separate.

**Safe next gate:** authorized physical DIMM inventory and compatible matched 2×32 GiB DDR5 UDIMM upgrade, memory stability tests, `>=60 GiB` OS-visible capacity and `>=8 GiB` available headroom, then an operator-approved isolated R9700 maintenance window. Review the source and full integrity evidence in [LMS issue #51](https://github.com/scottjoyner/lms/issues/51) and publisher-security boundaries in [Auto-Router PR #80](https://github.com/scottjoyner/auto-router/pull/80).