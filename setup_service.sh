#!/usr/bin/env bash
# =============================================================
#  setup_service.sh
#  Run this ONCE as the "test" user to install and enable
#  the AcroBot systemd user service.
#
#  Usage:
#    chmod +x setup_service.sh
#    ./setup_service.sh
# =============================================================

set -euo pipefail

PROJECT_DIR="/home/test/Desktop/Acrobot-2.2"
VENV_PYTHON="$PROJECT_DIR/acrobot-env/bin/python"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="acrobot.service"

echo ""
echo "============================================================"
echo "  AcroBot 2.2 — Service Installer"
echo "============================================================"

# ── Step 1: Pre-flight checks ─────────────────────────────
echo ""
echo "[1/6] Running pre-flight checks..."

if [ ! -f "$PROJECT_DIR/acrobot.py" ]; then
    echo "  ❌ acrobot.py not found at $PROJECT_DIR"
    exit 1
fi
echo "  ✅ acrobot.py found"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "  ❌ Virtual env not found at $VENV_PYTHON"
    echo "     Create it with: python3 -m venv $PROJECT_DIR/acrobot-env"
    exit 1
fi
echo "  ✅ Virtual environment found"

if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "  ❌ .env file not found at $PROJECT_DIR/.env"
    echo "     Create it and add GROQ_API_KEY=your_key_here"
    exit 1
fi
echo "  ✅ .env file found"

if ! grep -q "GROQ_API_KEY" "$PROJECT_DIR/.env"; then
    echo "  ❌ GROQ_API_KEY not found inside .env"
    exit 1
fi
echo "  ✅ GROQ_API_KEY present in .env"

# ── Step 2: Verify user UID (needed for PulseAudio path) ──
echo ""
echo "[2/6] Checking user UID..."
USER_UID=$(id -u)
if [ "$USER_UID" -ne 1000 ]; then
    echo "  ⚠️  Your UID is $USER_UID (not 1000)."
    echo "     Update PULSE_RUNTIME_PATH in acrobot.service to:"
    echo "     Environment=PULSE_RUNTIME_PATH=/run/user/$USER_UID/pulse"
    read -p "  Continue anyway? [y/N] " confirm
    [ "$confirm" = "y" ] || exit 1
else
    echo "  ✅ UID=1000 — PulseAudio path is correct"
fi

# ── Step 3: Install service file ──────────────────────────
echo ""
echo "[3/6] Installing service file..."
mkdir -p "$SERVICE_DIR"
cp "$(dirname "$0")/acrobot.service" "$SERVICE_DIR/$SERVICE_NAME"
echo "  ✅ Copied to $SERVICE_DIR/$SERVICE_NAME"

# ── Step 4: Enable linger ─────────────────────────────────
# linger = user session (and its services) start at boot
# WITHOUT requiring a physical login.
echo ""
echo "[4/6] Enabling loginctl linger for user '$USER'..."
sudo loginctl enable-linger "$USER"
echo "  ✅ Linger enabled — service will start at boot without login"

# ── Step 5: Enable and start ──────────────────────────────
echo ""
echo "[5/6] Enabling and starting acrobot.service..."
systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user start  "$SERVICE_NAME"

# Brief wait then check
sleep 3
STATUS=$(systemctl --user is-active "$SERVICE_NAME" 2>/dev/null || true)

if [ "$STATUS" = "active" ]; then
    echo "  ✅ Service is RUNNING"
else
    echo "  ⚠️  Service status: $STATUS"
    echo "     Check logs with:"
    echo "       journalctl --user -u $SERVICE_NAME -n 50"
fi

# ── Step 6: Summary ───────────────────────────────────────
echo ""
echo "[6/6] Setup complete."
echo ""
echo "  Useful commands:"
echo "    journalctl --user -u acrobot.service -f      # live logs"
echo "    systemctl --user status acrobot.service       # quick status"
echo "    systemctl --user restart acrobot.service      # manual restart"
echo "    systemctl --user stop acrobot.service         # stop"
echo ""
echo "  ⚠️  NEXT STEP: Reboot now and verify the chatbot starts automatically."
echo "    sudo reboot"
echo ""
echo "  Only run lockdown.sh AFTER you have confirmed it works post-reboot."
echo "============================================================"
