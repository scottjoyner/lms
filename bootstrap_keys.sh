#!/bin/bash
# Run this in a terminal on a fleet node (via RustDesk / tailscale SSH).
# Installs the runner's public keys so the runner can SSH in and collect hardware.
set -e
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat >> ~/.ssh/authorized_keys <<'EOF'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKPYyFx2ucZylkPOV7xEDRor/5JyLgzzKLqSq4hTeKbH hermes-agent@x1-370
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOkodatvrCSygQIGtV4nH2wPQ0ws+KNexc+Aa0WeDkSy hermes@kipnerter
EOF
chmod 600 ~/.ssh/authorized_keys
echo "Keys installed on $(hostname). Runner can now SSH in."
