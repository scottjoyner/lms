#!/usr/bin/env bash
set -euo pipefail

LMS_FLEET_ROLE_POLICY="${LMS_FLEET_ROLE_POLICY:-$HOME/.config/lms-fleet/fleet-role-policy.v1.json}"
LMS_FLEET_MATRIX_OUT="${LMS_FLEET_MATRIX_OUT:-$HOME/.local/state/lms-fleet/fleet-routing-matrix.json}"
LMS_FLEET_COMPARISON_DIR="${LMS_FLEET_COMPARISON_DIR:-$HOME/lms-fleet-runs/compare}"
LMS_FLEET_MIN_DISCOVERED_NODES="${LMS_FLEET_MIN_DISCOVERED_NODES:-3}"
ASSISTX_BASE_URL="${ASSISTX_BASE_URL:-http://127.0.0.1:8000}"
LMS_FLEET_PUBLISH_TO_ASSISTX="${LMS_FLEET_PUBLISH_TO_ASSISTX:-true}"

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale CLI is required" >&2
  exit 2
fi
if ! command -v lms-fleet-routing-matrix >/dev/null 2>&1; then
  echo "lms-fleet-routing-matrix is not installed" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 2
fi
if [[ ! -r "$LMS_FLEET_ROLE_POLICY" ]]; then
  echo "fleet role policy is not readable: $LMS_FLEET_ROLE_POLICY" >&2
  exit 2
fi
if [[ "$LMS_FLEET_PUBLISH_TO_ASSISTX" == "true" ]]; then
  if [[ -z "${BASIC_AUTH_USER:-}" || -z "${BASIC_AUTH_PASS:-}" ]]; then
    echo "BASIC_AUTH_USER and BASIC_AUTH_PASS are required for AssistX publication" >&2
    exit 2
  fi
fi

mkdir -p "$(dirname "$LMS_FLEET_MATRIX_OUT")"
arguments=(
  --policy "$LMS_FLEET_ROLE_POLICY"
  --out "$LMS_FLEET_MATRIX_OUT"
)

if [[ -d "$LMS_FLEET_COMPARISON_DIR" ]]; then
  while IFS= read -r -d '' comparison; do
    arguments+=(--comparison "$comparison")
  done < <(
    find "$LMS_FLEET_COMPARISON_DIR" -maxdepth 1 -type f -name '*.json' -print0 \
      | sort -z
  )
fi

if [[ "$LMS_FLEET_PUBLISH_TO_ASSISTX" == "true" ]]; then
  arguments+=(--assistx-url "$ASSISTX_BASE_URL")
fi

lms-fleet-routing-matrix "${arguments[@]}"

python3 - "$LMS_FLEET_MATRIX_OUT" "$LMS_FLEET_MIN_DISCOVERED_NODES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
minimum = int(sys.argv[2])
document = json.loads(path.read_text(encoding="utf-8"))
summary = document.get("summary") or {}
count = int(summary.get("tailnet_nodes") or 0)
online = int(summary.get("online_nodes") or 0)
if count < minimum:
    raise SystemExit(
        f"tailnet discovery gate failed: discovered={count}, required={minimum}"
    )
if document.get("admission", {}).get("admitted") is not False:
    raise SystemExit("routing matrix must remain non-admitting")
print(
    f"fleet routing matrix refreshed: discovered={count} online={online} "
    f"output={path}"
)
PY
