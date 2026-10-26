#!/usr/bin/env bash
set -euo pipefail

NAS_ROOT=${NAS_ROOT:-/mnt/nas/models}
SSD_ROOT=${SSD_ROOT:-/mnt/ssd/models/benchmark-stage}
OUT=${OUT:-"results/storage-preflight-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$OUT"

{
  date -u +%FT%TZ
  hostname
  echo '=== df -hT ==='
  df -hT
  echo '=== df -B1 ==='
  df -B1
  echo '=== mounts ==='
  mount
  echo '=== block devices ==='
  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL 2>/dev/null || true
} >"$OUT/storage.txt"

for root in "$NAS_ROOT" "$SSD_ROOT"; do
  safe=${root//\//_}
  if [[ -e "$root" ]]; then
    df -hT "$root" >"$OUT/df${safe}.txt" 2>&1 || true
    stat -f "$root" >"$OUT/statfs${safe}.txt" 2>&1 || true
  else
    echo missing >"$OUT/missing${safe}.txt"
  fi
done

# Capture likely model-file consumers without traversing the full NAS recursively.
find "$SSD_ROOT" -maxdepth 2 -type f \( -name '*.gguf' -o -name '*.safetensors' \) -printf '%s\t%p\n' 2>/dev/null | sort -nr >"$OUT/ssd-model-files.tsv" || true

echo "$OUT"
