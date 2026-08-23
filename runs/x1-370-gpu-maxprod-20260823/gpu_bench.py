#!/usr/bin/env python3
"""Quick throughput benchmark against llama.cpp/LM Studio OpenAI-compatible endpoint."""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def gen(base, model, prompt, max_tokens, api_key=None):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    if "error" in d:
        raise RuntimeError(f"API error: {d['error']}")
    u = d.get("usage", {})
    ct = u.get("completion_tokens", 0)
    return ct, dt, (ct / dt if dt else 0)

def run(args):
    model = args.model
    def worker(i):
        return gen(args.base, model, args.prompt, args.max_tokens, args.api_key)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        results = list(ex.map(worker, range(args.conc)))
    wall = time.perf_counter() - t0
    total = sum(r[0] for r in results)
    lat = [r[1] for r in results]
    indiv = [r[2] for r in results]
    print(json.dumps({
        "conc": args.conc, "requests": args.conc, "wall_s": round(wall, 2),
        "total_completion_tokens": total,
        "aggregate_tps": round(total / wall, 1),
        "per_stream_tps": [round(x, 1) for x in indiv],
        "lat_min_s": round(min(lat), 2), "lat_max_s": round(max(lat), 2),
    }))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default="Write a detailed 300-word technical explanation of how asynchronous GPU compute queues work.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--conc", type=int, default=1)
    p.add_argument("--api-key", default=None)
    a = p.parse_args()
    if not a.model:
        with urllib.request.urlopen(a.base.rstrip("/") + "/models", timeout=10) as r:
            a.model = json.loads(r.read())["data"][0]["id"]
    run(a)
