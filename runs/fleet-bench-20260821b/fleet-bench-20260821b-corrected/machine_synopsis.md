# LMS Machine Profile

- Generated UTC: `2026-08-22T04:29:20.259834+00:00`
- Hostname: `x1-370`
- Platform: `Linux-7.0.0-29-generic-x86_64-with-glibc2.39`
- Python: `3.12.3`

## CPU

- Model: `AMD Ryzen AI 9 HX 370 w/ Radeon 890M`
- Architecture: `x86_64`
- Logical processors: `24`
- Cores/socket: `12`
- Threads/core: `2`

## Memory

- Total RAM: `89.69 GiB`
- Available RAM: `46.31 GiB`
- Swap total: `8.0 GiB`

## Storage

- Root total: `877.75 GiB`
- Root free: `524.24 GiB`

## GPU / acceleration

- `c7:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 7551 (rev c0)`
- `c9:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] Device 150e (rev c1)`

## LM Studio endpoints

| Base URL | Reachable | Models | Latency s | Error |
|---|:---:|---:|---:|---|
| `http://100.108.99.47:1234/v1` | yes | 32 | 0.016 | `` |
| `http://100.69.158.114:1234/v1` | yes | 9 | 0.005 | `` |
| `http://100.81.57.77:1234/v1` | yes | 32 | 0.004 | `` |
| `http://192.168.1.178:1234/v1` | yes | 2 | 0.153 | `` |
| `http://192.168.1.81:1234/v1` | yes | 1 | 0.004 | `` |

## Recommendations

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 5 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Warnings

- nvidia runtime probe unavailable or failed: command not found: nvidia-smi
- rocm runtime probe unavailable or failed: command not found: rocm-smi
