# LMS Agent Recommendations

- Generated UTC: `2026-08-05T01:21:59.687293+00:00`
- Run directory: `runs/agent-loop-test/run_20260804T202800Z/20260805t012159z`

## Machine synopsis

- System RAM is suitable for heavier local model testing and multi-model benchmark sweeps.
- GPU hardware is visible, but no NVIDIA/ROCm runtime was confirmed; expect CPU or limited acceleration unless LM Studio reports otherwise.
- 1 LM Studio endpoint(s) were reachable during profiling; benchmark these first.

## Task-specific routing

No benchmark rows found. Run `lms quick` first.

## Operating rules

- Prefer task-family routes over general routes.
- Use fallback routes when the preferred route is below threshold or unavailable.
- Fall back to a stronger model when deterministic evaluator scores are low.
- Treat routing as evidence-based guidance, not a guarantee.
