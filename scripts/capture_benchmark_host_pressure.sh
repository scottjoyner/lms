#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-"benchmark-host-pressure-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$OUT"

{ date -u +%FT%TZ; hostname; uname -a; uptime; free -b; swapon --show --bytes 2>/dev/null || true; vmstat 1 5 2>/dev/null || true; } >"$OUT/host.txt"
ps -eo pid,ppid,comm,%cpu,%mem,rss,vsz,args --sort=-rss >"$OUT/processes-by-rss.txt" 2>/dev/null || true
systemctl --type=service --state=running --no-pager >"$OUT/systemd-running.txt" 2>&1 || true
ss -lntup >"$OUT/listening-ports.txt" 2>&1 || true
mount >"$OUT/mounts.txt" 2>&1 || true
df -hT >"$OUT/df.txt" 2>&1 || true

if command -v docker >/dev/null 2>&1; then
  docker ps --no-trunc >"$OUT/docker-ps.txt" 2>&1 || true
  docker stats --no-stream --no-trunc >"$OUT/docker-stats.txt" 2>&1 || true
  docker system df -v >"$OUT/docker-system-df.txt" 2>&1 || true
  docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Labels}}' >"$OUT/docker-labels.tsv" 2>&1 || true
  docker inspect $(docker ps -q) >"$OUT/docker-inspect-running.json" 2>&1 || true
fi

if command -v amd-smi >/dev/null 2>&1; then
  amd-smi static --json >"$OUT/amd-smi-static.json" 2>&1 || true
  amd-smi metric --json >"$OUT/amd-smi-metric.json" 2>&1 || true
fi
if command -v rocminfo >/dev/null 2>&1; then rocminfo >"$OUT/rocminfo.txt" 2>&1 || true; fi

# Common stateful-service probes. These are inventory only; no mutation occurs.
systemctl is-active neo4j >"$OUT/neo4j-systemd-state.txt" 2>&1 || true
if command -v docker >/dev/null 2>&1; then docker ps --filter name=neo4j --no-trunc >"$OUT/neo4j-docker-state.txt" 2>&1 || true; fi

echo "$OUT"
