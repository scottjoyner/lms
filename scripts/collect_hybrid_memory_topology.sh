#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-hybrid-topology}
mkdir -p "$OUT"

{ date -u +%FT%TZ; uname -a; free -b; lsmem 2>/dev/null || true; numactl --hardware 2>/dev/null || true; } >"$OUT/host-memory.txt"
rocminfo >"$OUT/rocminfo.txt" 2>&1 || true
command -v amd-smi >/dev/null 2>&1 && amd-smi static --json >"$OUT/amd-smi-static.json" 2>&1 || true
command -v amd-smi >/dev/null 2>&1 && amd-smi metric --json >"$OUT/amd-smi-metric.json" 2>&1 || true
lspci -nn >"$OUT/lspci.txt" 2>&1 || true
lspci -tv >"$OUT/lspci-tree.txt" 2>&1 || true

# Preserve negotiated PCIe speed/width and BAR sizing for display adapters.
if command -v lspci >/dev/null 2>&1; then
  while read -r bdf; do
    safe=${bdf//[:.]/_}
    lspci -s "$bdf" -vv >"$OUT/pci-${safe}.txt" 2>&1 || true
  done < <(lspci -D | awk '/VGA compatible controller|Display controller|3D controller/{print $1}')
fi

for f in /sys/class/drm/card*/device/{current_link_speed,current_link_width,max_link_speed,max_link_width,resource,resource0_wc}; do
  [[ -e "$f" ]] || continue
  printf '%s\n' "### $f" >>"$OUT/drm-pcie.txt"
  cat "$f" >>"$OUT/drm-pcie.txt" 2>/dev/null || true
  printf '\n' >>"$OUT/drm-pcie.txt"
done

if [[ -n "${LLAMA_BUILD:-}" && -x "$LLAMA_BUILD/bin/llama-bench" ]]; then
  "$LLAMA_BUILD/bin/llama-bench" --list-devices >"$OUT/llama-devices.txt" 2>&1 || true
fi

# Optional ROCm bandwidth utilities differ by packaging. Run whichever exists.
for cmd in rocm-bandwidth-test rocm_bandwidth_test; do
  if command -v "$cmd" >/dev/null 2>&1; then "$cmd" >"$OUT/${cmd}.txt" 2>&1 || true; break; fi
done

cat >"$OUT/NOTES.txt" <<'EOF'
Interpret the BIOS-reserved iGPU VRAM separately from GPU-accessible UMA/GTT memory.
A small preallocated VRAM value (for example 2 GiB) does not prove the iGPU is
limited to that amount for compute. Use rocminfo, llama device inventory, AMD-SMI,
and observed allocations as the evidence source.

For OCuLink, preserve negotiated PCIe generation and lane width. The benchmark
must distinguish model residency in dGPU VRAM, APU/UMA memory, ordinary CPU
memory, and any repeated cross-link traffic during evaluation.
EOF

echo "$OUT"
