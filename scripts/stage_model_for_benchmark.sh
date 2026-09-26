#!/usr/bin/env bash
set -euo pipefail

SOURCE=${SOURCE:?set SOURCE=/nas/path/model.gguf}
STAGE_ROOT=${STAGE_ROOT:-/mnt/ssd/models/benchmark-stage}
MIN_FREE_GIB=${MIN_FREE_GIB:-40}
KEEP_AFTER=${KEEP_AFTER:-0}
VERIFY_SHA=${VERIFY_SHA:-1}
OUT=${OUT:-"results/model-stage-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$OUT" "$STAGE_ROOT"

[[ -f "$SOURCE" ]] || { echo "source missing: $SOURCE" >&2; exit 2; }
source_bytes=$(stat -c %s "$SOURCE")
source_name=$(basename "$SOURCE")
free_bytes=$(df -B1 --output=avail "$STAGE_ROOT" | tail -1 | tr -d ' ')
reserve_bytes=$((MIN_FREE_GIB*1024*1024*1024))
required=$((source_bytes+reserve_bytes))

{
  echo "source=$SOURCE"
  echo "stage_root=$STAGE_ROOT"
  echo "source_bytes=$source_bytes"
  echo "free_bytes_before=$free_bytes"
  echo "reserve_bytes=$reserve_bytes"
  df -hT "$SOURCE" "$STAGE_ROOT" 2>&1 || true
} >"$OUT/preflight.txt"

if (( free_bytes < required )); then
  echo "SKIP: insufficient SSD space. Need model bytes + ${MIN_FREE_GIB} GiB reserve." | tee "$OUT/status.txt"
  exit 3
fi

sha=$(sha256sum "$SOURCE" | awk '{print $1}')
dest="$STAGE_ROOT/${sha:0:16}-$source_name"

if [[ -f "$dest" ]]; then
  existing=$(stat -c %s "$dest")
  if [[ "$existing" == "$source_bytes" ]]; then
    echo "reuse-existing" >"$OUT/status.txt"
  else
    rm -f "$dest"
  fi
fi

if [[ ! -f "$dest" ]]; then
  tmp="$dest.partial"
  rm -f "$tmp"
  start=$(date +%s)
  if command -v rsync >/dev/null 2>&1; then
    rsync --info=progress2 --partial --inplace "$SOURCE" "$tmp" | tee "$OUT/copy.log"
  else
    cp --reflink=auto --sparse=always "$SOURCE" "$tmp"
  fi
  sync "$tmp" || true
  mv "$tmp" "$dest"
  end=$(date +%s)
  echo $((end-start)) >"$OUT/copy-seconds.txt"
  echo "staged" >"$OUT/status.txt"
fi

if [[ "$VERIFY_SHA" == 1 ]]; then
  dest_sha=$(sha256sum "$dest" | awk '{print $1}')
  [[ "$dest_sha" == "$sha" ]] || { echo "SHA mismatch" >&2; exit 4; }
fi

{
  echo "sha256=$sha"
  echo "staged_path=$dest"
  echo "source_fs=$(df -T "$SOURCE" | tail -1)"
  echo "stage_fs=$(df -T "$STAGE_ROOT" | tail -1)"
  df -hT "$SOURCE" "$STAGE_ROOT" 2>&1 || true
} >"$OUT/result.txt"

printf '%s\n' "$dest"

if [[ "$KEEP_AFTER" != 1 ]]; then
  cat >"$OUT/cleanup.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
rm -f -- '$dest'
df -hT '$STAGE_ROOT'
EOF
  chmod +x "$OUT/cleanup.sh"
fi
