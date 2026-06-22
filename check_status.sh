#!/usr/bin/env bash
# =============================================================
#  check_status.sh
#  Run anytime to see a complete health snapshot of AcroBot.
#  No root needed.
#
#  Usage:
#    chmod +x check_status.sh
#    ./check_status.sh
# =============================================================

echo ""
echo "============================================================"
echo "  AcroBot 2.2 — System Status"
echo "============================================================"

# ── Service status ────────────────────────────────────────
echo ""
echo "▶ SERVICE"
STATUS=$(systemctl --user is-active acrobot.service 2>/dev/null || echo "unknown")
ENABLED=$(systemctl --user is-enabled acrobot.service 2>/dev/null || echo "unknown")
echo "   Active  : $STATUS"
echo "   Enabled : $ENABLED"

# ── Last 10 log lines ─────────────────────────────────────
echo ""
echo "▶ LAST 10 LOG LINES"
journalctl --user -u acrobot.service -n 10 --no-pager 2>/dev/null || \
    echo "   (no logs yet)"

# ── Audio devices ─────────────────────────────────────────
echo ""
echo "▶ AUDIO DEVICES"
echo "   Playback:"
aplay  -l 2>/dev/null | grep -E "^card" | sed 's/^/     /' || echo "     (none found)"
echo "   Capture:"
arecord -l 2>/dev/null | grep -E "^card" | sed 's/^/     /' || echo "     (none found)"

# ── Internet connectivity ─────────────────────────────────
echo ""
echo "▶ NETWORK"
if timeout 2 bash -c "echo > /dev/tcp/8.8.8.8/53" 2>/dev/null; then
    echo "   Internet : ✅ reachable (8.8.8.8:53)"
else
    echo "   Internet : ❌ not reachable"
fi
if timeout 3 curl -sf --head https://api.groq.com > /dev/null 2>&1; then
    echo "   Groq API : ✅ reachable"
else
    echo "   Groq API : ❌ not reachable"
fi

# ── Linger status ─────────────────────────────────────────
echo ""
echo "▶ LINGER (auto-start without login)"
USER_NAME=$(whoami)
if loginctl show-user "$USER_NAME" 2>/dev/null | grep -q "Linger=yes"; then
    echo "   Linger   : ✅ enabled for '$USER_NAME'"
else
    echo "   Linger   : ❌ NOT enabled — service won't start at boot"
    echo "   Fix with : sudo loginctl enable-linger $USER_NAME"
fi

# ── SSH status ────────────────────────────────────────────
echo ""
echo "▶ SSH"
if systemctl is-active --quiet ssh 2>/dev/null; then
    PW_AUTH=$(grep -E "^PasswordAuthentication" /etc/ssh/sshd_config 2>/dev/null || echo "not set")
    echo "   SSH      : running"
    echo "   Password : $PW_AUTH"
else
    echo "   SSH      : disabled (no remote access)"
fi

# ── Boot target ───────────────────────────────────────────
echo ""
echo "▶ BOOT TARGET"
TARGET=$(systemctl get-default 2>/dev/null || echo "unknown")
echo "   Default  : $TARGET"
[ "$TARGET" = "multi-user.target" ] && echo "              (CLI — correct for production)"
[ "$TARGET" = "graphical.target"  ] && echo "              (Desktop GUI — consider switching to CLI)"

# ── .env security ─────────────────────────────────────────
echo ""
echo "▶ .ENV FILE"
ENV_PATH="/home/test/Desktop/Acrobot-2.2/.env"
if [ -f "$ENV_PATH" ]; then
    PERMS=$(stat -c "%a %U" "$ENV_PATH")
    echo "   Perms    : $PERMS"
    [ "$(stat -c '%a' "$ENV_PATH")" = "600" ] && echo "   Security : ✅ owner-only" || \
        echo "   Security : ⚠️  too open — run: chmod 600 $ENV_PATH"
else
    echo "   ❌ Not found at $ENV_PATH"
fi

echo ""
echo "============================================================"
echo "  Tip: Watch live logs with:"
echo "    journalctl --user -u acrobot.service -f"
echo "============================================================"
echo ""
