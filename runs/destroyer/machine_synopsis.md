# LMS Machine Profile

- Generated UTC: `2026-07-16T02:36:10.224316+00:00`
- Hostname: `x1-370`
- Platform: `Linux-6.17.0-40-generic-x86_64-with-glibc2.39`
- Python: `3.12.3`

## CPU

- Model: `AMD Ryzen AI 9 HX 370 w/ Radeon 890M`
- Architecture: `x86_64`
- Logical processors: `24`
- Cores/socket: `12`
- Threads/core: `2`

## Memory

- Total RAM: `91.94 GiB`
- Available RAM: `29.04 GiB`
- Swap total: `24.0 GiB`

## Storage

- Root total: `877.75 GiB`
- Root free: `807.72 GiB`

## GPU / acceleration

- `c6:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 150e (rev c1)`

## LM Studio endpoints

| Base URL | Reachable | Models | Latency s | Error |
|---|:---:|---:|---:|---|
| `http://destroyer.tailcb8954.ts.net:1234/v1` | yes | 11 | 0.032 | `` |

## Recommendations

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Warnings

- nvidia runtime probe unavailable or failed: command not found: nvidia-smi
- rocm runtime probe unavailable or failed: command not found: rocm-smi
