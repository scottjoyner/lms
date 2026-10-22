#!/usr/bin/env bash
set -euo pipefail

OUT=${1:?usage: sample_rocm_telemetry.sh OUT.jsonl [PID] [INTERVAL]}
PID=${2:-}
INTERVAL=${3:-1}
mkdir -p "$(dirname "$OUT")"

while true; do
  ts=$(date -u +%FT%TZ)
  tmp=$(mktemp)
  if command -v amd-smi >/dev/null 2>&1; then
    amd-smi metric --json >"$tmp" 2>/dev/null || printf '{}\n' >"$tmp"
  else
    printf '{}\n' >"$tmp"
  fi
  rss=null
  vms=null
  if [[ -n "$PID" && -r "/proc/$PID/status" ]]; then
    rss=$(awk '/^VmRSS:/ {print $2*1024}' "/proc/$PID/status" 2>/dev/null || echo null)
    vms=$(awk '/^VmSize:/ {print $2*1024}' "/proc/$PID/status" 2>/dev/null || echo null)
  fi
  python3 - "$ts" "$PID" "$rss" "$vms" "$tmp" >>"$OUT" <<'PY'
import json,sys
from pathlib import Path
stamp,pid,rss,vms,path=sys.argv[1:]
try:
    amd=json.loads(Path(path).read_text())
except Exception:
    amd={}
def conv(v):
    try:return int(v)
    except:return None
print(json.dumps({"ts":stamp,"pid":int(pid) if pid else None,"rss_bytes":conv(rss),"vms_bytes":conv(vms),"amd_smi":amd},sort_keys=True))
PY
  rm -f "$tmp"
  sleep "$INTERVAL"
done
