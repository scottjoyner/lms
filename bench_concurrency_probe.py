#!/usr/bin/env python3
"""Per-model concurrent-request probe.

Documents the concurrency tradeoff the fleet actually cares about:
- small models: 2 concurrent sessions = more total throughput, acceptable latency
- big models (e.g. Hermes): 2 concurrent sessions stall / fail
- weak nodes (optiplex/lenovo/destroyer) choke, so they stay single-stream

For every (node, currently-loaded model):
  Phase A: 1 concurrent request  -> single-stream baseline (ttft, tps)
  Phase B: K concurrent requests  -> per-stream ttft/tps + aggregate throughput
Then it records the speed-hit (per-stream slowdown) vs throughput-gain, and a
status of OK / DEGRADED / STALL / FAIL.

Concurrency is capped: --max-concurrent (default 2, hard ceiling 4 per gating).
Struggling nodes are forced to 1. Speed-leader models are never tested above 2.

Run after the single-stream crash-doc pass so nodes aren't doubly loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "concurrency_probe"

# Single source of truth for discovery now lives in fleet_discover.py
# (fleet.toml + tailscale). Backwards-compatible alias.
sys.path.insert(0, str(HERE))
from fleet_discover import discover, retry  # noqa: E402
NODES = {n.name: n.url for n in discover()}

PROMPT = "Write a short, concrete paragraph explaining why distributed systems are harder than single-machine systems."
MAX_TOKENS = 256
HARD_CAP = 4  # user-gated ceiling


@dataclass
class Result:
    node: str
    model: str
    phase: str          # "single" | "conc"
    concurrency: int
    ttft_ms: float
    tps: float          # tokens / (end - start), whole-request tps
    gen_tps: float      # tokens / (end - first_token)
    tokens: int
    status: str         # OK | STALL | FAIL
    error: str


def _post_stream(base: str, model: str, timeout: float):
    """Open one streaming completion. Returns (ttft_s, tokens, gen_s, err)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    start = time.perf_counter()
    first_token = None
    tokens = 0
    text = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if first_token is None and obj.get("choices"):
                    delta = obj["choices"][0].get("delta", {})
                    if delta.get("content"):
                        first_token = time.perf_counter()
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens"):
                    tokens = int(usage["completion_tokens"])
                if obj.get("choices"):
                    c = obj["choices"][0].get("delta", {}).get("content")
                    if c:
                        text += len(c)
        end = time.perf_counter()
    except Exception as e:  # timeout / connection reset / etc
        end = time.perf_counter()
        if first_token is None:
            return None, 0, 0, f"STALL:{type(e).__name__}:{e}"
        return (first_token - start), tokens or max(1, text // 4), (end - first_token), f"PARTIAL:{type(e).__name__}"

    if first_token is None:
        return None, 0, 0, f"STALL:no_first_token"
    if tokens == 0:
        tokens = max(1, text // 4)
    return (first_token - start), tokens, (end - first_token), ""


def measure(base: str, model: str, concurrency: int, timeout: float):
    """Fire `concurrency` requests near-simultaneously; return list[Result-ish]."""
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_post_stream, base, model, timeout) for _ in range(concurrency)]
        for fut in as_completed(futs):
            ttft, toks, gen, err = fut.result()
            if err:
                out.append({"ttft": None, "tps": 0.0, "gen_tps": 0.0, "tokens": toks, "status": "FAIL" if "STALL" not in err else "STALL", "error": err})
            else:
                wall = ttft + gen
                out.append({"ttft": ttft * 1000, "tps": toks / wall, "gen_tps": toks / gen, "tokens": toks, "status": "OK", "error": ""})
    return out


def loaded_models(base: str) -> list[str]:
    def _get():
        with urllib.request.urlopen(base + "/models", timeout=8) as r:
            return json.load(r)
    try:
        data = retry(_get, retries=2, what=f"models {base}")
    except Exception:
        return []
    objs = data.get("data", data) if isinstance(data, dict) else data
    return [m.get("id") for m in objs if m.get("state") in (None, "loaded", "partially_loaded") or m.get("loaded", True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", type=int, default=2, help="top concurrency per model (default 2, hard cap 4)")
    ap.add_argument("--struggle-nodes", default="scott-optiplex-9030-aio,lenovo-ideapad-330s-15ikb,destroyer,deathstar",
                    help="comma list forced to concurrency 1 (deathstar: CPU maxed by other jobs, >7GB chokes)")
    ap.add_argument("--top-models", default="", help="comma list of speed leaders never tested above 2")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-request stall timeout (s)")
    ap.add_argument("--only", action="append", default=[], help="restrict to these nodes")
    ap.add_argument("--model-filter", default="", help="substring filter on model id")
    args = ap.parse_args()

    k = min(max(1, args.max_concurrent), HARD_CAP)
    struggle = {n.strip() for n in args.struggle_nodes.split(",") if n.strip()}
    top = {m.strip() for m in args.top_models.split(",") if m.strip()}
    filter_sub = args.model_filter.lower()

    targets = {n: u for n, u in NODES.items() if (not args.only or n in args.only)}
    from fleet_discover import live_nodes
    all_nodes = discover()
    live_nodeset = {n.name for n in live_nodes(
        [type(next(iter(all_nodes)))(name=k, url=v, via="cli") for k, v in targets.items()])}
    targets = {n: u for n, u in targets.items() if n in live_nodeset}
    if not targets:
        print("no live nodes to probe", flush=True)
        return 0

    rows: list[dict] = []
    for node, base in targets.items():
        caps = 1 if node in struggle else k
        models = loaded_models(base)
        if not models:
            print(f"[{node}] no loaded models / unreachable", flush=True)
            continue
        print(f"\n=== [{node}] {len(models)} loaded models, concurrency_cap={caps} ===", flush=True)
        for model in models:
            if filter_sub and filter_sub not in model.lower():
                continue
            mcap = min(caps, 2) if (top and any(t and t in model for t in top)) else caps
            # Phase A: single-stream baseline
            single = measure(base, model, 1, args.timeout)
            s = single[0]
            s_ok = s["status"] == "OK"
            s_ttft = s["ttft"] or 0.0
            s_tps = s["tps"]
            rows.append({"node": node, "model": model, "phase": "single", "concurrency": 1,
                         "ttft_ms": round(s_ttft, 1), "tps": round(s_tps, 2), "gen_tps": round(s["gen_tps"], 2),
                         "tokens": s["tokens"], "status": s["status"], "error": s["error"]})
            if not s_ok:
                print(f"  {model}: SINGLE {s['status']} ({s['error']}) -> skip concurrency", flush=True)
                continue
            if mcap < 2:
                print(f"  {model}: single OK (ttft={s_ttft:.0f}ms tps={s_tps:.1f}) [capped at 1]", flush=True)
                continue
            # Phase B: concurrent
            conc = measure(base, model, mcap, args.timeout)
            any_fail = any(c["status"] != "OK" for c in conc)
            c_ttft = sum(c["ttft"] for c in conc if c["ttft"] is not None) / max(1, sum(1 for c in conc if c["ttft"] is not None))
            c_tps_stream = sum(c["tps"] for c in conc) / len(conc)
            c_total = sum(c["tokens"] for c in conc) / max(1, (max(c["ttft"] for c in conc if c["ttft"]) / 1000 + 0.001))
            speed_hit = (1 - c_tps_stream / s_tps) * 100 if s_tps else 0.0
            gain = (c_total / s_tps - 1) * 100 if s_tps else 0.0
            overall = "FAIL" if any_fail else ("OK" if speed_hit <= 25 else ("DEGRADED" if speed_hit <= 60 else "POOR"))
            for i, c in enumerate(conc):
                rows.append({"node": node, "model": model, "phase": f"conc{i}", "concurrency": mcap,
                             "ttft_ms": round(c["ttft"] or 0.0, 1), "tps": round(c["tps"], 2), "gen_tps": round(c["gen_tps"], 2),
                             "tokens": c["tokens"], "status": c["status"], "error": c["error"]})
            rows.append({"node": node, "model": model, "phase": "summary", "concurrency": mcap,
                         "ttft_ms": round(c_ttft, 1), "tps": round(c_tps_stream, 2), "gen_tps": 0.0,
                         "tokens": 0, "status": overall,
                         "error": f"speed_hit={speed_hit:.0f}% gain={gain:.0f}%"})
            print(f"  {model}: single tps={s_tps:.1f} | conc{mcap} stream_tps={c_tps_stream:.1f} "
                  f"(hit {speed_hit:.0f}%) total={c_total:.1f} ({gain:+.0f}%) -> {overall}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = OUT / f"concurrency_probe_{stamp}.csv"
    cols = ["node", "model", "phase", "concurrency", "ttft_ms", "tps", "gen_tps", "tokens", "status", "error"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {path} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
