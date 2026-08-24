#!/usr/bin/env bash
# fleet-gen-health: alert when AssistX fleet-gen task success rate collapses.
# Catches exactly today's failure mode: pipeline silently broken for hours/days.
set -Eeuo pipefail

OUT=/media/scott/SSD_4TB/recovery/fleet-gen-health
mkdir -p "$OUT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT="$OUT/fleet-gen-health-$STAMP.txt"
exec >"$REPORT" 2>&1

MIN_SAMPLE=${MIN_SAMPLE:-8}          # need at least N finished tasks to judge
FAIL_RATIO_PCT=${FAIL_RATIO_PCT:-60} # alert when FAILED share exceeds this

notify() {
  local msg=$1 key sig last
  sig=$(echo "$msg" | sed -E 's/[0-9]+//g' | md5sum | cut -c1-12)
  last=$(cat /home/scott/fleet-watchdog/state/fleet-gen.last_alert 2>/dev/null || true)
  if [ "$sig" = "${last:-}" ]; then
    echo "PUSHCUT: deduped"
    return 0
  fi
  key=$(grep -oP 'PUSHCUT_API_KEY=\K.*' /home/scott/fleet-watchdog/.env 2>/dev/null || true)
  if [ -n "${key:-}" ]; then
    curl -sS -m 10 -X POST "https://api.pushcut.io/${key}/notifications/Signal%20Agent" \
      -H "Content-Type: application/json" \
      --data "$(python3 -c 'import json,sys; print(json.dumps({"title":"🚨 x1-370 fleet-gen health","text":sys.argv[1]}))' "$msg")" >/dev/null 2>&1 || true
    mkdir -p /home/scott/fleet-watchdog/state
    echo "$sig" > /home/scott/fleet-watchdog/state/fleet-gen.last_alert
    echo "PUSHCUT: notified: $msg"
  else
    echo "PUSHCUT: no key — alert NOT sent: $msg"
  fi
}

python3 - "$MIN_SAMPLE" "$FAIL_RATIO_PCT" <<'PY' || notify "$(cat $REPORT)"
import sys, time
from neo4j import GraphDatabase

min_sample = int(sys.argv[1]); max_fail_pct = int(sys.argv[2])
now = int(time.time() * 1000)
cutoff = now - 15 * 60 * 1000
drv = GraphDatabase.driver("bolt://100.64.43.123:7687", auth=("neo4j", "knowledge_graph_2026"))
with drv.session(database="assistx") as s:
    done = s.run("MATCH (t:Task {status:'DONE'}) WHERE t.updated_at_ts > $c RETURN count(t) AS c", c=cutoff).single()["c"]
    failed = s.run("MATCH (t:Task {status:'FAILED'}) WHERE t.updated_at_ts > $c RETURN count(t) AS c", c=cutoff).single()["c"]
    dns = s.run("MATCH (t:Task {status:'FAILED'}) WHERE t.updated_at_ts > $c AND t.result_json CONTAINS 'name resolution' RETURN count(t) AS c", c=cutoff).single()["c"]
drv.close()
total = done + failed
print(f"window=15m done={done} failed={failed} dns_type={dns}")
if total < min_sample:
    print("sample too small — skipping judgment")
    sys.exit(0)
fail_pct = round(failed * 100 / total)
if fail_pct >= max_fail_pct:
    print(f"FAIL: fleet-gen failure rate {fail_pct}% ({failed}/{total}) exceeds {max_fail_pct}%")
    sys.exit(1)
print(f"OK: failure rate {fail_pct}% ({done} DONE)")
PY
