#!/usr/bin/env python3
"""Benchmark every reachable fleet LM Studio node and drop the artifacts where the
auto-assist value layer expects them (``runs/<node>/``), so routing becomes
benchmarked-value-aware.

Each node is benchmarked with the full agent skill suite (one model at a time),
then ``lms_model_fit.py`` adds ``model_fit.csv``. Runs are named by the node's
short hostname so ``fleet.py``'s value index keys match (host = norm(node)).

Run in the background:  python3 bench_fleet.py
Re-run periodically by passing --loop (sleeps between full passes).
"""
from __future__ import annotations

import argparse
import csv
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# Reachable fleet endpoints (verified from the x1-370 host). Nodes that are asleep
# or not bound are skipped at runtime (a quick /v1/models probe gates each pass).
NODES = {
    "xwing": "http://xwing.tailcb8954.ts.net:1234/v1",
    "joyner": "http://joyner.tailcb8954.ts.net:1234/v1",
    "deathstar": "http://100.78.106.121:1234/v1",
    "beelink-ryzen-7-mini-pc": "http://100.85.72.121:1234/v1",
    "lenovo-ideapad-330s-15ikb": "http://scott-lenovo-ideapad-330s-15ikb.tailcb8954.ts.net:1234/v1",
    "scotts-macbook-air": "http://scotts-macbook-air.tailcb8954.ts.net:1234/v1",
    "destroyer": "http://destroyer.tailcb8954.ts.net:1234/v1",
    "scott-optiplex-9030-aio": "http://scott-optiplex-9030-aio.tailcb8954.ts.net:1234/v1",
    "x1-370": "http://127.0.0.1:1234/v1",                          # this host's own LM Studio
}

MAX_MODELS = int(__import__("os").environ.get("BENCH_MAX_MODELS", "50"))
CTX_TOKENS = int(__import__("os").environ.get("BENCH_MAX_CTX", "4096"))
TIMEOUT = int(__import__("os").environ.get("BENCH_TIMEOUT", "900"))


def _reachable(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url + "/v1/models", timeout=8)
        return True
    except Exception:
        return False


RUNNER_HOST = socket.gethostname()


def _stamp_host(run_dir: Path, node: str) -> None:
    """The benchmark runs from THIS host, so CSVs stamp host_name=<runner>, not the
    target node. The auto-assist value layer keys on host_name, so rewrite it to the
    target node or every node's data would wrongly collapse onto the runner."""
    for csv_path in run_dir.glob("*.csv"):
        try:
            rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
        except Exception:
            continue
        if not rows or "host_name" not in rows[0]:
            continue
        changed = False
        for r in rows:
            if r.get("host_name") != node:
                r["host_name"] = node
                changed = True
        if changed:
            fields = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            print(f"  stamped host_name={node} in {csv_path.name}", flush=True)


def bench_node(node: str, url: str) -> bool:
    out = RUNS / node
    print(f"\n=== [{node}] benchmark start ({url}) ===", flush=True)
    cmd = [
        sys.executable, str(HERE / "lms_cli.py"), "quick",
        "--endpoint", url,
        "--run-id", node,
        "--output-dir", str(RUNS),
        "--max-models", str(MAX_MODELS),
        "--repeats", "1",
        "--max-context-tokens", str(CTX_TOKENS),
        "--timeout", str(TIMEOUT),
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[{node}] lms quick failed rc={rc}", flush=True)
        return False
    fit = subprocess.call([sys.executable, str(HERE / "lms_model_fit.py"), str(out)])
    print(f"[{node}] model_fit rc={fit}", flush=True)
    _stamp_host(out, node)
    # Sanity: what we produced
    for f in ("capability_matrix.csv", "run_summary.csv", "model_fit.csv"):
        p = out / f
        print(f"  {f}: {'OK' if p.exists() else 'MISSING'}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="repeat passes forever")
    ap.add_argument("--sleep", type=int, default=3600, help="seconds between loops")
    ap.add_argument("--only", action="append", default=[], help="restrict to these nodes")
    ap.add_argument("--concurrency", type=int, default=4, help="nodes benchmarked in parallel")
    args = ap.parse_args()

    import concurrent.futures

    targets = {k: v for k, v in NODES.items() if (not args.only or k in args.only)}
    while True:
        # Probe reachability up front, then benchmark the live nodes concurrently
        # (each node is an independent machine, so parallel benchmarks are safe and
        # collect fleet data fastest).
        live = {n: u for n, u in targets.items() if _reachable(u)}
        skipped = set(targets) - set(live)
        for n in sorted(skipped):
            print(f"[{n}] skip: endpoint unreachable", flush=True)
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futs = {ex.submit(bench_node, n, u): n for n, u in live.items()}
            for fut in concurrent.futures.as_completed(futs):
                n = futs[fut]
                try:
                    if fut.result():
                        done += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[{n}] exception: {exc}", flush=True)
        print(f"\nPASS complete: {done}/{len(targets)} nodes benchmarked.", flush=True)
        if not args.loop:
            break
        print(f"sleeping {args.sleep}s before next pass...", flush=True)
        time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
