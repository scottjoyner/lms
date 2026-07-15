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
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

# Single source of truth for node discovery now lives in fleet_discover.py
# (fleet.toml + tailscale). This kills the duplicated NODES dict and the
# fragile ALIAS hacks that used to live here.
sys.path.insert(0, str(HERE))
from fleet_discover import discover, live_nodes, retry  # noqa: E402

# Backwards-compatible alias so other tooling importing NODES still works.
NODES = {n.name: n.url for n in discover()}

MAX_MODELS = int(__import__("os").environ.get("BENCH_MAX_MODELS", "50"))
CTX_TOKENS = int(__import__("os").environ.get("BENCH_MAX_CTX", "4096"))
TIMEOUT = int(__import__("os").environ.get("BENCH_TIMEOUT", "900"))
HARDWARE_AT_BENCH = __import__("os").environ.get("BENCH_CAPTURE_HW", "1") == "1"


def _reachable(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url + "/v1/models", timeout=8)
        return True
    except Exception:
        return False


def _capture_hw(node: str, url: str) -> None:
    """Co-capture REAL per-node hardware DURING the bench (Item 5).

    The old flow collected hardware once, manually, over SSH — so capacity
    numbers were stale vs the contended run. Now we snapshot it as part of the
    benchmark pass: local host via collect_node_profile.py directly, remote
    hosts over SSH (best-effort; a failure just skips the snapshot, it never
    fails the bench).
    """
    if not HARDWARE_AT_BENCH:
        return
    out = RUNS / node / "host_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "127.0.0.1" in url or "localhost" in url:
            proc = subprocess.run(
                [sys.executable, str(HERE / "collect_node_profile.py")],
                capture_output=True, text=True, timeout=30)
        else:
            # derive a likely SSH host from the tailscale URL
            host = url.split("://", 1)[-1].split(":")[0]
            if host.replace(".", "").isdigit():
                ssh_host = f"scott@{host}"
            else:
                ssh_host = host
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                 "-o", "StrictHostKeyChecking=no", ssh_host,
                 f"python3 - {(HERE / 'collect_node_profile.py').name}"],
                capture_output=True, text=True, timeout=40)
        payload = proc.stdout.strip()
        # keep only the last JSON object (script prints indented JSON)
        start = payload.rfind("{")
        if start >= 0:
            data = json.loads(payload[start:])
            data.setdefault("collected_at_utc", datetime.now(timezone.utc).isoformat())
            data["source"] = data.get("source") or "collect_node_profile.py (captured during bench)"
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  captured hardware -> {out.name} ({data.get('memory',{}).get('ram_total_gib')} GiB)", flush=True)
    except Exception as e:  # noqa: BLE001 - hardware is best-effort
        print(f"  hardware capture skipped for {node}: {e}", flush=True)


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
    # Co-capture hardware as part of the pass (Item 5) so capacity numbers
    # reflect the contended run, not a stale manual snapshot.
    _capture_hw(node, url)
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
    # Retry the whole node bench on transient failure (Item 8) instead of
    # hard-failing a node because of one LM Studio hiccup.
    try:
        rc = retry(lambda: subprocess.call(cmd), retries=2, what=f"bench {node}")
    except Exception as exc:  # noqa: BLE001
        print(f"[{node}] lms quick failed after retries: {exc}", flush=True)
        return False
    if rc != 0:
        print(f"[{node}] lms quick failed rc={rc}", flush=True)
        return False
    fit = subprocess.call([sys.executable, str(HERE / "lms_model_fit.py"), str(out)])
    print(f"[{node}] model_fit rc={fit}", flush=True)
    _stamp_host(out, node)
    # Sanity: what we produced
    for f in ("capability_matrix.csv", "run_summary.csv", "model_fit.csv"):
        p = out / f
        status = "OK" if p.exists() else "MISSING"
        print(f"  {f}: {status}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="repeat passes forever")
    ap.add_argument("--sleep", type=int, default=3600, help="seconds between loops")
    ap.add_argument("--only", action="append", default=[], help="restrict to these nodes")
    ap.add_argument("--concurrency", type=int, default=4, help="nodes benchmarked in parallel")
    args = ap.parse_args()

    import concurrent.futures

    # Resolve targets from the shared discovery module (fleet.toml + tailscale).
    all_nodes = discover()
    targets = {n.name: n.url for n in all_nodes if (not args.only or n.name in args.only)}
    while True:
        # Probe reachability up front WITH retry/backoff (Item 8), then benchmark
        # the live nodes concurrently (each node is an independent machine, so
        # parallel benchmarks are safe and collect fleet data fastest).
        live_nodeset = live_nodes([type(next(iter(all_nodes)))(
            name=k, url=v, via="cli") for k, v in targets.items()])
        live = {n.name: n.url for n in live_nodeset}
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
