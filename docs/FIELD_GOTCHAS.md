# LM Studio Fleet Operations — Field Gotchas

Lessons from the 2026-08-23 tuning/benchmark campaign. Each one cost real
time; read before fleet ops.

## 1. `lms server start` binds to localhost by default

Any CLI restart of a remote node's server silently removes it from fleet
reachability. Always:

```bash
lms server start --bind 0.0.0.0
```

(Or set `LMS_SERVER_HOST`.) Symptom: `curl node:1234/v1/models` times out
while `lms ps` looks normal.

## 2. Per-model default config naming — two conventions

Directory: `~/.lmstudio/.internal/user-concrete-model-default-config/`

- **Hub/JIT models** (the ones that matter for routing): use
  `<publisher>/<model-id>.json`, e.g. `liquid/lfm2-24b-a2b.json`,
  `prism-ml/bonsai-27b.json`. A GGUF-filename-path JSON is silently ignored
  for these.
- **Local file-backed models**: `<publisher>/<dir>/<file>.gguf.json`, e.g.
  `mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf.json`.

Verify a config took effect by checking llama-server argv after load:
`tr '\0' '\n' /proc/<pid>/cmdline | grep -A1 flash-attn`.

## 3. CLI speculative-draft flags are broken (2.x)

`lms load --speculative-draft-simple --speculative-draft-model <key>` falls
back to interactive model selection even with `-y`. Configure speculation via
per-model default config keys (`llm.load.llama.speculativeDecoding.draftSimple`,
`.draftModel`) instead — failures there at least surface a readable CAUSE.

## 4. Speculation (MTP and draft) hurts throughput on Navi 48

GPU is compute-saturated; draft compute steals target cycles. Measured
~40% aggregate loss on ornith-1.5 (MTP) and worse with simple draft on
qwen3.8. Default: **speculation off** for GPU-resident models.

## 5. VRAM oversubscription wedges the whole stack, not just the request

Loading past VRAM (e.g. parallel 8 × 65k f16 KV) leads to HTTP 400s, hangs,
and a llama-server spinning at 100% CPU that survives SIGTERM. Recovery:
SIGKILL the backend, unload/reload with right-sized context/KV quant.
Prevention: KV q8_0 + explicit contextLength in per-model defaults.

## 6. Unified-memory nodes OOM-kill the daemon, not just the model

Strix Halo (xwing): oversized default contexts (262k) on shared RAM caused
kernel OOM kills of lm-studio, gateways, everything in the cgroup
(`suspected_oom=True`, 15–37 GB swap churn). Right-size contexts per model
on iGPU nodes.

## 7. Backend auto-updates can brick a node

LM Studio flipped xwing's backend preference to vulkan-avx2 2.29.1 which
SIGABRTs every engine spawn on gfx1151 ("Engine protocol runtime exited
before becoming healthy"). Fix:
`~/.lmstudio/.internal/backend-preferences-v1.json` → known-good version
(vulkan 2.28.2). ROCm 2.28.2 does NOT work on gfx1151.

## 8. Tailscale SSH check-mode breaks non-interactive sessions

Tunnels/services using Tailscale SSH can enter interactive re-auth loops
(xwing tunnel crash-looped 243× before detection). For machine-to-machine
paths prefer regular sshd with key auth over LAN/tailscale IPs.

## 9. dflash drafts need poolside's runtime

`poolside-laguna-s-2.1-dflash` GGUFs fail tensor-count validation on stock
llama.cpp backends (`expected 76, got 69`). No dflash support without
poolside's fork.

## 10. Fleet model IDs vs local paths

`lms ls` shows hub entries whose "location" column may say Local/xwing/etc.;
loading by hub key when only another node has it triggers JIT download or an
interactive picker. Copy weights over LAN (scp) into the same publisher dir
to register locally, then `lms ls` picks them up.
