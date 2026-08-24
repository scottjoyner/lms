#!/usr/bin/env bash
# Restore x1-370 tuned LM Studio configuration after crash/reinstall.
# Usage: bash restore-x1-370-lmstudio.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
LMSTUDIO="${HOME}/.lmstudio"
SYSTEMD="${HOME}/.config/systemd/user"

mkdir -p "${LMSTUDIO}/.internal/user-concrete-model-default-config/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF"
mkdir -p "${LMSTUDIO}/.internal/user-concrete-model-default-config/poolside"
mkdir -p "${SYSTEMD}"

cp "${SRC}/Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf.json" \
  "${LMSTUDIO}/.internal/user-concrete-model-default-config/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF/"
cp "${SRC}/poolside/laguna-s-2.1.json" \
  "${LMSTUDIO}/.internal/user-concrete-model-default-config/poolside/"
cp "${SRC}/lm-studio-model-autoload.service" "${SYSTEMD}/"

systemctl --user daemon-reload
systemctl --user enable lm-studio-model-autoload.service

echo "Config restored."
echo "  - ornith-1.5: FA on, KV q8_0/q8_0, parallel 16, ctx 65536, MTP off (default)"
echo "  - laguna-s:   ctx 32768, parallel 4, FA on, KV q8_0, no speculation"
echo "Start serving with:"
echo "  systemctl --user start lm-studio-model-autoload.service"
