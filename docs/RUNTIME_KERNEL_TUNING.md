# Runtime, Kernel, and CPU Tuning Controls

The benchmark campaign should verify the execution path as well as the model configuration. Current llama.cpp exposes CPU affinity, NUMA, polling, process priority, Flash Attention, explicit device placement, MoE CPU placement, mmap/direct-I/O, mlock and other controls that can materially change latency or stability.

## CPU affinity and jitter

For promoted configurations, compare the default scheduler against strict physical-core placement and at least one pinned CPU range. Preserve:

- `--threads`;
- `--threads-batch`;
- `--cpu-range` / `--cpu-mask`;
- `--cpu-strict`;
- batch equivalents;
- p50/p95/p99 TTFT;
- CPU utilization and power.

The goal is not maximum CPU consumption; it is reducing scheduler jitter and preventing background/service work from stealing the cores used for prompt processing and host-resident MoE experts.

## NUMA / memory locality

When the platform exposes meaningful NUMA behavior, compare default placement with `--numa isolate` and an explicit `numactl` policy. Do not assume this helps on a single-node HX-370; the result is empirical. If page cache was populated under a different NUMA policy, treat the run as contaminated or explicitly drop/reload pages before comparison.

## Polling and scheduler priority

`--poll`, `--poll-batch`, `--prio`, and `--prio-batch` trade CPU/power for wakeup latency. Measure them only on a selected model/runtime profile after the larger model/context/placement questions are answered.

## Kernel-path A/B

Build and compare selected controls:

```text
HIP graphs ON / OFF
rocWMMA FlashAttention ON / OFF
FlashAttention auto / forced ON / OFF
```

These tests prove whether a claimed optimization is actually responsible for the observed speedup. Preserve build SHA and CMake flags as distinct runtime identities.

AMD has continued upstreaming ROCm/RDNA FlashAttention optimization work during 2026, so the benchmark should pin and record llama.cpp commit rather than treating "latest ROCm" as a stable execution path.

## Load-path controls

For large models compare:

```text
mmap
no-mmap
direct I/O
mlock where enough RAM remains
```

Separate model load time from steady-state inference. `mlock` should only be attempted with enough host-memory reserve; preventing swapping is useful, forcing the OS into memory pressure is not.

## SWA/cache controls

Models using sliding-window attention should test model-default behavior against `--swa-full` only when context quality or cache sizing justifies it. This can significantly alter memory requirements and should be treated as a separate context loadout.

## Tensor placement overrides

Current llama.cpp supports `--override-tensor` buffer placement. This is a research arm after layer-split and `--cpu-moe` baselines are stable. It can be used to test whether specific attention/router/expert tensors benefit disproportionately from dGPU residency on the OCuLink topology.

Do not begin with fine-grained tensor overrides: first establish the coarse placement frontier, then use telemetry/activation evidence to choose a small set of high-value overrides.

## Promotion rule

A runtime/kernel tweak is promoted only when it reproduces across multiple runs and improves at least one target metric without unacceptable regression in another. Record TTFT, TG, PP, power, host CPU usage, memory pressure and p99 behavior. Avoid "1% faster" claims when run-to-run noise overlaps the difference.
