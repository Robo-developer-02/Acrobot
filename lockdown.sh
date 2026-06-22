#!/usr/bin/env bash
# =============================================================
#  lockdown.sh
#
#  Hardens the Raspberry Pi 4 for production deployment.
#  Run this ONLY after setup_service.sh and a reboot test
#  confirm the chatbot starts correctly on its own.
#
#  What this script does:
#   [A] Boot → CLI autologin (no desktop, saves ~300MB RAM)
#   [B] Disables SSH password login (key-only or fully off)
#   [C] Sets a strong admin password you choose
#   [D] Removes "test" user from sudo group (optional, ask)
#   [E] Disables Bluetooth (not needed, reduces attack surface)
#   [F] Disables unnecessary services (cups, avahi, triggerhappy)
#   [G] Makes the .env file readable only by "test" user
#   [H] Locks the project directory permissions
#
#  Usage:
#    chmod +x lockdown.sh
#    sudo ./lockdown.sh
# =============================================================

set -euo pipefail

# Must run as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Run this script with sudo: sudo ./lockdown.sh"
    exit 1
fi

PROJECT_DIR="/home/test/Desktop/Acrobot-2.2"
ADMIN_USER="test"

echo ""
echo "============================================================"
echo "  AcroBot 2.2 — Production Lockdown"
echo "============================================================"
echo ""
echo "  ⚠️  WARNING: This will restrict system access."
echo "  Make sure the chatbot service is working before continuing."
echo ""
read -p "  Type YES to continue: " confirm
[ "$confirm" = "YES" ] || { echo "Aborted."; exit 0; }

# ── [A] Boot to CLI with autologin ────────────────────────
echo ""
echo "[A] Configuring boot to CLI with autologin..."

# raspi-config stores boot behaviour in /etc/systemd system links.
# We replicate what raspi-config does: boot target = multi-user (CLI),
# autologin for the "test" user so the user session starts (needed for
# the systemd --user service and PulseAudio).

# Set default boot target to multi-user (CLI), not graphical
systemctl set-default multi-user.target

# Configure getty autologin for "test" on tty1
mkdir -p /etc/systemd/system/getty@tty1.service.d/
cat > /etc/systemd/system/getty@tty1.service.d/autologin.conf << UNIT
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $ADMIN_USER --noclear %I \$TERM
UNIT

echo "  ✅ Boot target: CLI + autologin as '$ADMIN_USER'"
echo "     (Desktop GUI disabled — saves ~300 MB RAM for the chatbot)"

# ── [B] SSH hardening ─────────────────────────────────────
echo ""
echo "[B] SSH configuration..."
echo "  Choose SSH policy:"
echo "    1) Disable SSH completely (most secure, no remote access)"
echo "    2) Keep SSH but only with a key (no password login)"
echo "    3) Leave SSH as-is"
read -p "  Enter choice [1/2/3]: " ssh_choice

case "$ssh_choice" in
    1)
        systemctl disable ssh 2>/dev/null || true
        systemctl stop    ssh 2>/dev/null || true
        echo "  ✅ SSH disabled completely"
        echo "  ⚠️  You can only access the Pi physically now."
        echo "     To re-enable later: sudo systemctl enable --now ssh"
        ;;
    2)
        if [ ! -d "/home/$ADMIN_USER/.ssh" ]; then
            echo "  ⚠️  No ~/.ssh directory found for '$ADMIN_USER'."
            echo "     You MUST add your public key to /home/$ADMIN_USER/.ssh/authorized_keys"
            echo "     BEFORE disabling password auth, or you will be locked out."
            read -p "     Have you already added your SSH key? [y/N] " key_confirm
            [ "$key_confirm" = "y" ] || {
                echo "  Skipping SSH hardening. Do it manually after adding your key."
                break
            }
        fi
        SSHD_CONF="/etc/ssh/sshd_config"
        cp "$SSHD_CONF" "${SSHD_CONF}.bak"
        sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/'   "$SSHD_CONF"
        sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/'                 "$SSHD_CONF"
        sed -i 's/^#*ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' "$SSHD_CONF"
        # Add if not present
        grep -q "^PasswordAuthentication" "$SSHD_CONF" || echo "PasswordAuthentication no" >> "$SSHD_CONF"
        systemctl restart ssh
        echo "  ✅ SSH: password login disabled, key-only access enabled"
        ;;
    3)
        echo "  ⏭  SSH left unchanged"
        ;;
    *)
        echo "  Invalid choice — SSH left unchanged"
        ;;
esac

# ── [C] Set a strong admin password ───────────────────────
echo ""
echo "[C] Setting admin password for user '$ADMIN_USER'..."
echo "  Choose a strong password — this is the only way back into the system."
passwd "$ADMIN_USER"
echo "  ✅ Password updated"

# ── [D] Sudo restriction (optional) ──────────────────────
echo ""
echo "[D] Sudo access for '$ADMIN_USER'..."
echo "  Currently '$ADMIN_USER' has full sudo access."
echo "  For a production box, you can remove it — but you will need"
echo "  to create a separate maintenance account for admin tasks."
echo ""
read -p "  Remove sudo from '$ADMIN_USER'? [y/N] " sudo_choice
if [ "$sudo_choice" = "y" ]; then
    # Create a maintenance admin account first
    read -p "  Enter name for new maintenance admin account: " MAINT_USER
    if id "$MAINT_USER" &>/dev/null; then
        echo "  User '$MAINT_USER' already exists — adding to sudo"
    else
        adduser "$MAINT_USER"
    fi
    usermod -aG sudo "$MAINT_USER"
    deluser "$ADMIN_USER" sudo
    echo "  ✅ Sudo removed from '$ADMIN_USER'"
    echo "  ✅ '$MAINT_USER' now has sudo access"
    echo "  ⚠️  Use '$MAINT_USER' for any future admin tasks."
else
    echo "  ⏭  Sudo access left unchanged for '$ADMIN_USER'"
fi

# ── [E] Disable Bluetooth ─────────────────────────────────
echo ""
echo "[E] Disabling Bluetooth (not needed for chatbot)..."
systemctl disable bluetooth 2>/dev/null || true
systemctl stop    bluetooth 2>/dev/null || true
# Also disable in config.txt to save power
if ! grep -q "dtoverlay=disable-bt" /boot/config.txt 2>/dev/null && \
   ! grep -q "dtoverlay=disable-bt" /boot/firmware/config.txt 2>/dev/null; then
    # Try both paths (Bookworm uses /boot/firmware, Bullseye uses /boot)
    BOOT_CFG="/boot/config.txt"
    [ -f "/boot/firmware/config.txt" ] && BOOT_CFG="/boot/firmware/config.txt"
    echo "dtoverlay=disable-bt" >> "$BOOT_CFG"
fi
echo "  ✅ Bluetooth disabled"

# ── [F] Disable unnecessary services ──────────────────────
echo ""
echo "[F] Disabling unnecessary background services..."

SERVICES_TO_DISABLE=(
    "cups"           # printing
    "cups-browsed"   # print discovery
    "avahi-daemon"   # mDNS/Bonjour — not needed for a voice bot
    "triggerhappy"   # hotkey daemon
    "hciuart"        # Bluetooth UART (already disabled above)
    "wpa_supplicant" # only disable if using Ethernet, not WiFi
)

for svc in "${SERVICES_TO_DISABLE[@]}"; do
    if systemctl list-unit-files --quiet "$svc.service" &>/dev/null; then
        systemctl disable "$svc" 2>/dev/null || true
        systemctl stop    "$svc" 2>/dev/null || true
        echo "  ✅ $svc disabled"
    else
        echo "  ⏭  $svc not installed"
    fi
done

# ── [G] Lock down .env file ───────────────────────────────
echo ""
echo "[G] Securing .env file (contains API key)..."
if [ -f "$PROJECT_DIR/.env" ]; then
    chown "$ADMIN_USER:$ADMIN_USER" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"     # owner read/write only
    echo "  ✅ .env: permissions set to 600 (owner-only)"
else
    echo "  ⚠️  .env not found at $PROJECT_DIR — skipping"
fi

# ── [H] Lock project directory ────────────────────────────
echo ""
echo "[H] Locking project directory permissions..."
chown -R "$ADMIN_USER:$ADMIN_USER" "$PROJECT_DIR"
chmod 750 "$PROJECT_DIR"              # owner full, group read/exec, others nothing
echo "  ✅ $PROJECT_DIR: permissions set to 750"

# ── Summary ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Lockdown complete. Summary:"
echo ""
echo "  [A] Boot → CLI autologin as '$ADMIN_USER' (no desktop)"
echo "  [B] SSH: see choice above"
echo "  [C] Admin password: updated"
echo "  [D] Sudo: see choice above"
echo "  [E] Bluetooth: disabled"
echo "  [F] Unnecessary services: disabled"
echo "  [G] .env: chmod 600 (API key secured)"
echo "  [H] Project dir: chmod 750"
echo ""
echo "  ⚠️  IMPORTANT — Reboot now to verify everything still works:"
echo "    sudo reboot"
echo ""
echo "  After reboot, verify with:"
echo "    journalctl --user -u acrobot.service -n 30"
echo "============================================================"
