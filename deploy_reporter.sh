#!/usr/bin/env bash
# Deploy the fleet node reporter to a remote tailnet node as a user-level service.
#
# The deployed service is self-contained under:
#   ~/.local/share/hermes-fleet-reporter/
#
# Optional evidence/runtime configuration is read from:
#   ~/.config/lms/runtime-evidence/reporter.env
#
# Supported variables in reporter.env:
#   FLEET_RUNTIME_URLS
#   FLEET_RUNTIME_WITNESSES
#   FLEET_RUNTIME_CONTINUITY_SIGNING_KEY
#   FLEET_RUNTIME_CONTINUITY_IDENTITY
#   LMSTUDIO_URL
#
# The config file is never copied from the deploying host. It is node-local
# operator state and may contain paths to node-scoped private keys.
#
# Platform service:
#   Linux -> systemd user service
#   macOS -> launchd agent
#
# Usage:
#   ./deploy_reporter.sh <user@host> [router-url]
set -euo pipefail

TARGET="${1:?usage: deploy_reporter.sh <user@host> [router-url]}"
ROUTER_URL="${2:-${ROUTER_URL:-http://100.64.43.123:8088}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPORTER="$HERE/fleet_node_reporter.py"
PACKAGE_DIR="$HERE/src/lms_agent_bench"
REMOTE_ROOT=".local/share/hermes-fleet-reporter"

if [[ ! -f "$REPORTER" ]]; then
  echo "fleet_node_reporter.py not found next to this script ($REPORTER)" >&2
  exit 1
fi
if [[ ! -d "$PACKAGE_DIR" ]]; then
  echo "lms_agent_bench package not found at $PACKAGE_DIR" >&2
  exit 1
fi

echo "Deploying fleet node reporter to $TARGET (router=$ROUTER_URL)"

SSH=(ssh -o StrictHostKeyChecking=no "$TARGET")
REMOTE_UNAME="$("${SSH[@]}" 'uname' 2>/dev/null || echo Unknown)"
RHOME="$("${SSH[@]}" 'printf "%s" "$HOME"')"

# Choose and validate a remote Python before mutating the service. The current
# evidence code uses Python 3.10+ syntax, so an older Apple/Xcode Python is not
# sufficient.
REMOTE_PYTHON="$("${SSH[@]}" 'for p in python3 python; do if command -v "$p" >/dev/null 2>&1 && "$p" -c '"'"'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'"'"' 2>/dev/null; then command -v "$p"; exit 0; fi; done; exit 1' || true)"
if [[ -z "$REMOTE_PYTHON" ]]; then
  echo "remote node requires Python >=3.10 before reporter deployment" >&2
  exit 2
fi

echo "Remote platform=$REMOTE_UNAME python=$REMOTE_PYTHON"

# Stage a complete source bundle. This avoids the historical failure mode where
# only fleet_node_reporter.py was copied but its lms_agent_bench imports were
# unavailable on the remote node.
"${SSH[@]}" "mkdir -p ~/$REMOTE_ROOT/src/lms_agent_bench ~/.config/lms/runtime-evidence"
scp -q -o StrictHostKeyChecking=no "$REPORTER" "$TARGET:$REMOTE_ROOT/fleet_node_reporter.py"
tar -C "$HERE/src" -cf - lms_agent_bench | "${SSH[@]}" "tar -C ~/$REMOTE_ROOT/src -xf -"

# Install a stable wrapper that loads optional node-local evidence settings.
"${SSH[@]}" 'bash -s' <<RMT
set -euo pipefail
ROOT="$HOME/$REMOTE_ROOT"
cat > "$ROOT/run-reporter.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/.local/share/hermes-fleet-reporter"
CONFIG="$HOME/.config/lms/runtime-evidence/reporter.env"
if [[ -f "$CONFIG" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$CONFIG"
  set +a
fi

args=(
  --router-url "$ROUTER_URL"
  --interval 30
)
if [[ -n "${LMSTUDIO_URL:-}" ]]; then
  args+=(--lmstudio-url "$LMSTUDIO_URL")
fi
if [[ -n "${FLEET_RUNTIME_CONTINUITY_SIGNING_KEY:-}" ]]; then
  args+=(--runtime-continuity-signing-key "$FLEET_RUNTIME_CONTINUITY_SIGNING_KEY")
fi
if [[ -n "${FLEET_RUNTIME_CONTINUITY_IDENTITY:-}" ]]; then
  args+=(--runtime-continuity-identity "$FLEET_RUNTIME_CONTINUITY_IDENTITY")
fi

exec "$REMOTE_PYTHON" "$ROOT/fleet_node_reporter.py" "${args[@]}"
EOF
chmod 700 "$ROOT/run-reporter.sh"
RMT

if [[ "$REMOTE_UNAME" == "Darwin" ]]; then
  echo "Remote is macOS -> installing launchd agent"
  "${SSH[@]}" 'bash -s' <<RMT
set -euo pipefail
ROOT="$HOME/$REMOTE_ROOT"
PLIST="$HOME/Library/LaunchAgents/com.hermes.fleet-node-reporter.plist"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hermes.fleet-node-reporter</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${RHOME}/$REMOTE_ROOT/run-reporter.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${RHOME}/fleet-node-reporter.log</string>
    <key>StandardErrorPath</key>
    <string>${RHOME}/fleet-node-reporter.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
        <key>PYTHONPATH</key>
        <string>${RHOME}/$REMOTE_ROOT/src</string>
    </dict>
</dict>
</plist>
EOF
plutil -lint "$PLIST"
launchctl bootout "gui/$(id -u)/com.hermes.fleet-node-reporter" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/com.hermes.fleet-node-reporter"
sleep 2
launchctl print "gui/$(id -u)/com.hermes.fleet-node-reporter" | head -40
RMT
else
  echo "Remote is Linux (or unknown) -> installing systemd user service"
  "${SSH[@]}" 'bash -s' <<RMT
set -euo pipefail
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/fleet-node-reporter.service" <<EOF
[Unit]
Description=Fleet node reporter (read-only runtime evidence to auto-router)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=%h/$REMOTE_ROOT/src
ExecStart=/bin/bash %h/$REMOTE_ROOT/run-reporter.sh
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now fleet-node-reporter
loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user --no-pager status fleet-node-reporter | head -40
RMT
fi

echo "Reporter bundle deployed. Node-local evidence config remains under:"
echo "  ~/.config/lms/runtime-evidence/reporter.env"
