#!/usr/bin/env bash
set -euo pipefail

# Hosts are SSH destinations, e.g. ryzen-395 deathstar. No changes are made.
OUT=${OUT:-"results/relief-host-preflight-$(date -u +%Y%m%dT%H%M%SZ)"}
shift_count=0
mkdir -p "$OUT"
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <ssh-host> [ssh-host...]" >&2
  exit 2
fi

for host in "$@"; do
  safe=${host//[^A-Za-z0-9._-]/_}
  dir="$OUT/$safe"; mkdir -p "$dir"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" 'true' >/dev/null 2>&1; then
    echo unreachable >"$dir/status.txt"
    continue
  fi
  echo reachable >"$dir/status.txt"
  ssh "$host" 'hostname; uname -a; uptime; free -b; swapon --show --bytes 2>/dev/null || true; df -hT' >"$dir/host.txt" 2>&1 || true
  ssh "$host" 'command -v docker >/dev/null && { docker ps --no-trunc; echo ===STATS===; docker stats --no-stream --no-trunc; echo ===DF===; docker system df; } || true' >"$dir/docker.txt" 2>&1 || true
  ssh "$host" 'command -v amd-smi >/dev/null && amd-smi metric --json || true; command -v rocminfo >/dev/null && rocminfo || true' >"$dir/gpu.txt" 2>&1 || true
  ssh "$host" 'systemctl --type=service --state=running --no-pager 2>/dev/null || true; ss -lntup 2>/dev/null || true' >"$dir/services-and-ports.txt" 2>&1 || true
  ssh "$host" 'systemctl is-active neo4j 2>/dev/null || true; command -v docker >/dev/null && docker ps --filter name=neo4j --no-trunc || true' >"$dir/neo4j-state.txt" 2>&1 || true
done

echo "$OUT"
