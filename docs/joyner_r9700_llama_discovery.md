# Joyner R9700: opt-in llama.cpp model catalog discovery

Scope: **inventory only, not model admission**. This change does not assert that an HTTP-healthy model is resident on the R9700 GPU, fast, high-quality, authenticated, leased, or routable.

## Physical read-only evidence (2026-10-21)

- Joyner is an ASUS PRIME B650M-A AX II / Ryzen 5 7600X host with Radeon AI PRO R9700 (PCI 1002:7551; 32 GiB VRAM).
- Its ~14.75 GiB system RAM and ~4 GiB exhausted swap are **not** sufficient for the planned IQ3_S production trial. The GPU was runtime-suspended/D3hot during inspection.
- Four `llama-server` processes are present. Two `scott`-owned loopback services respond to `/health` and `/props`: `joyner-toolcall` (MiniCPM5-1B Q4_K_M, port 1234), and `joyner-ling` (Ling 3.0 Tiny, port 1236).
- Two separate `joyner`-owned API-key-protected services use Tailnet ports 18320 and 18321. This change **does not probe, rebind, register, or alter them**.
- The existing live reporter on Joyner sees only port 1234 in `library` and reports `loaded=[]` because its loaded detector is LM Studio-specific. Do not interpret that empty array as proof the llama.cpp process is not running.

## Proposal

The upstream `fleet_node_reporter.py` has a disabled-by-default `--joyner-llama-discovery` option. On hostname `joyner` only, it performs bounded GET-only checks against `127.0.0.1:1234` and `127.0.0.1:1236`, requiring **all three**:

1. `/health` must say `status=ok`.
2. `/props` must match the hardcoded expected alias and **exact model file path**.
3. `/v1/models` must advertise that exact alias.

The reporter then adds the verified aliases to its `library` only. It does **not** add either to `loaded`, start a model, measure tokens/sec, inspect keys, run benchmark traffic, or take an R9700 GPU context. Local HTTP requests are hardcoded, size-limited and reject redirects. Missing/unhealthy/drifted services are excluded.

The existing default behavior remains unchanged when the new option is absent. The opt-in feature must **not** be activated on the running reporter until an independent review of Auto-Router's model-library consumers and their dispatch admission policy is complete. In particular, the current `/api/fleet/node-report` receiver accepts self-reported model metadata and cannot be treated as independent owner-bound model evidence.

## Safe offline checks

```bash
python3 -m pytest -q tests/test_joyner_llama_discovery.py
python3 fleet_node_reporter.py --dry-run --joyner-llama-discovery
```

Use the dry-run only on a trusted host where the expected services exist. The flag is a one-shot **local print**, with no POST or daemon changes. The preview's model library is for display/diagnostics, not routing or capacity planning.

## Separate R9700 deployment gates

1. Authorize and physically verify the exact matched two-DIMM 64 GB kit (manufacturer-backed TEAMGROUP TED564G5200C42DC01 candidate), motherboard A2/B2 population and BIOS-specific compatibility; validate resulting RAM capacity and stability.
2. Reserve exclusive GPU/host capacity without unexpectedly killing four existing inference processes, then independently verify pinned GGUF shard hashes and the SSD expert pack (a staged byte count alone is insufficient).
3. Measure real task-relevant quality, tool-call correctness, cold/warm TTFT, prefill/decode tokens/sec, and RAM/VRAM/SSD pressure with the pinned HIP engine under a controlled isolated window. Compare against a same-task, same-context baseline.
4. Require authenticated, fresh, owner-bound routing receipts, isolation, rollback, and a separate operator release decision. Successful GET probes and this opt-in reporter change **never** satisfy these gates.

**Disposition:** review-only proposal; no production install, model launch, fleet-report POST or deployment.