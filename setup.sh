#!/usr/bin/env bash
# Raspberry Pi 5 + AI HAT+ setup for Be More Agent.
# This installer deliberately keeps hardware-package failures visible instead
# of aborting halfway and leaving no Python virtual environment.
set -u -o pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

fail() {
    printf '%b\n' "${RED}ERROR: $*${NC}" >&2
    exit 1
}

warn() {
    printf '%b\n' "${YELLOW}WARNING: $*${NC}" >&2
}

ok() {
    printf '%b\n' "${GREEN}$*${NC}"
}

apt_install_required() {
    local pkg="$1"
    echo "  installing required package: $pkg"
    if ! sudo apt-get install -y "$pkg"; then
        fail "Could not install required package '$pkg'. Fix the APT error above, then rerun ./setup.sh."
    fi
}

apt_install_optional() {
    local pkg="$1"
    echo "  installing optional package: $pkg"
    if ! sudo apt-get install -y "$pkg"; then
        warn "Optional package '$pkg' could not be installed. The rest of setup will continue."
        return 1
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "${GREEN}Be More Agent - Raspberry Pi 5 + Hailo setup${NC}"
echo

MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || true)"
ARCH="$(uname -m)"
OS_ID="$(. /etc/os-release 2>/dev/null; echo "${ID:-unknown}")"
OS_CODENAME="$(. /etc/os-release 2>/dev/null; echo "${VERSION_CODENAME:-unknown}")"
printf 'Detected: %s | arch=%s | OS=%s/%s\n' "${MODEL:-unknown board}" "$ARCH" "$OS_ID" "$OS_CODENAME"

if [[ "$MODEL" != *"Raspberry Pi 5"* ]]; then
    warn "This setup is designed for Raspberry Pi 5."
fi
if [[ "$ARCH" != "aarch64" ]]; then
    fail "A 64-bit Raspberry Pi OS installation is required (detected architecture: $ARCH)."
fi
if [[ "$OS_ID" != "raspbian" && "$OS_ID" != "debian" ]]; then
    warn "Official Hailo packages are expected from Raspberry Pi OS repositories. Detected OS ID: $OS_ID."
fi
if [[ "$OS_CODENAME" != "trixie" ]]; then
    warn "Current Raspberry Pi AI HAT+ documentation targets 64-bit Raspberry Pi OS Trixie. Detected: $OS_CODENAME."
fi

# ---------------------------------------------------------------------------
# 1. Base packages - install separately so one unavailable optional package
#    does not prevent venv creation.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/9] Updating APT metadata...${NC}"
sudo apt-get update || fail "apt update failed. Check networking and Raspberry Pi OS repositories."

echo -e "${YELLOW}[2/9] Installing required system packages...${NC}"
for pkg in \
    python3 python3-pip python3-venv python3-dev python3-tk \
    i2c-tools raspi-config pciutils \
    libasound2-dev portaudio19-dev liblapack-dev libblas-dev \
    cmake build-essential espeak-ng git curl wget dkms; do
    apt_install_required "$pkg"
done

# Raspberry Pi-specific Python/camera packages. These are expected on the
# official Raspberry Pi OS repository, but are kept separate for diagnostics.
for pkg in python3-picamera2 python3-gpiozero python3-lgpio rpicam-apps; do
    apt_install_optional "$pkg" || true
done

# ---------------------------------------------------------------------------
# 2. I2C. raspi-config is safer than appending dtparam into an arbitrary
#    conditional section in config.txt.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[3/9] Enabling I2C for the GY-521...${NC}"
I2C_REBOOT_REQUIRED=0
if command_exists raspi-config; then
    if sudo raspi-config nonint do_i2c 0; then
        echo "  raspi-config enabled I2C."
        I2C_REBOOT_REQUIRED=1
    else
        warn "raspi-config could not enable I2C; applying a config.txt fallback."
    fi
fi

BOOT_CONFIG="/boot/firmware/config.txt"
if [[ ! -f "$BOOT_CONFIG" && -f /boot/config.txt ]]; then
    BOOT_CONFIG="/boot/config.txt"
fi

if [[ -f "$BOOT_CONFIG" ]]; then
    # Add an explicit [all] block so settings are not accidentally placed
    # under a preceding [cm4], [pi4], or other conditional filter.
    if ! grep -Eq '^[[:space:]]*dtparam=i2c_arm=on([[:space:]]|$)' "$BOOT_CONFIG"; then
        {
            echo
            echo "# Be More Agent hardware"
            echo "[all]"
            echo "dtparam=i2c_arm=on"
        } | sudo tee -a "$BOOT_CONFIG" >/dev/null
        I2C_REBOOT_REQUIRED=1
    fi
else
    warn "Could not locate Raspberry Pi config.txt."
fi

# i2c-dev can be loaded immediately, but the Pi I2C controller itself may not
# appear until reboot after dtparam is changed.
echo i2c-dev | sudo tee /etc/modules-load.d/be-more-agent-i2c.conf >/dev/null
sudo modprobe i2c-dev 2>/dev/null || true

if [[ -e /dev/i2c-1 ]]; then
    ok "I2C bus /dev/i2c-1 is present."
    i2cdetect -y 1 || true
else
    warn "/dev/i2c-1 is not present yet. Reboot after setup, then run: i2cdetect -y 1"
fi

# ---------------------------------------------------------------------------
# 3. MAX98357A I2S amplifier.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[4/9] Configuring MAX98357A I2S audio...${NC}"
AUDIO_REBOOT_REQUIRED=0
if [[ -f "$BOOT_CONFIG" ]]; then
    if ! grep -Eq '^[[:space:]]*dtoverlay=max98357a([,[:space:]]|$)' "$BOOT_CONFIG"; then
        {
            echo
            echo "# Be More Agent MAX98357A: BCLK GPIO18, LRCLK GPIO19, DIN GPIO21"
            echo "[all]"
            echo "dtoverlay=max98357a,no-sdmode"
        } | sudo tee -a "$BOOT_CONFIG" >/dev/null
        AUDIO_REBOOT_REQUIRED=1
    fi
fi

# ---------------------------------------------------------------------------
# 4. Create project directories and venv EARLY. This specifically prevents
#    later Piper/Hailo/Ollama failures from causing a missing venv.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[5/9] Creating Python environment...${NC}"
mkdir -p piper voices assets
for d in greeting_sounds thinking_sounds ack_sounds error_sounds; do mkdir -p "sounds/$d"; done
for d in idle listening thinking speaking error capturing warmup; do mkdir -p "faces/$d"; done

if [[ -d venv && ! -x venv/bin/python ]]; then
    warn "Existing ./venv is incomplete. Removing and recreating it."
    rm -rf venv
fi
if [[ ! -d venv ]]; then
    python3 -m venv --system-site-packages venv || fail "python3 -m venv failed even though python3-venv is installed."
fi
# Ensure apt-installed Picamera2/Hailo bindings remain visible.
if [[ -f venv/pyvenv.cfg ]]; then
    sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' venv/pyvenv.cfg || true
fi
# shellcheck disable=SC1091
source venv/bin/activate || fail "Created venv but could not activate venv/bin/activate."
python -m pip install --upgrade pip wheel setuptools || fail "Could not upgrade pip tooling inside venv."
python -m pip install --no-cache-dir -r requirements.txt || fail "Python dependency installation failed."
ok "Python venv created at $BASE_DIR/venv"

# ---------------------------------------------------------------------------
# 5. Hailo AI HAT+ packages. For the 26-TOPS Hailo-8, the official package is
#    hailo-all. Do not install hailo-h10-all (that is AI HAT+ 2 / Hailo-10H).
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[6/9] Installing/verifying Hailo AI HAT+ support...${NC}"
HAILO_OK=0
if command_exists hailortcli; then
    HAILO_OK=1
else
    if apt-cache show hailo-all >/dev/null 2>&1; then
        if sudo apt-get install -y hailo-all; then
            hash -r
            if command_exists hailortcli; then
                HAILO_OK=1
            fi
        else
            warn "hailo-all installation failed. Review the APT error above."
        fi
    else
        warn "The 'hailo-all' package is not available from your configured repositories."
        echo "  Raspberry Pi's current AI HAT+ instructions require 64-bit Raspberry Pi OS Trixie"
        echo "  with Raspberry Pi repositories enabled."
        echo "  OS: $OS_ID/$OS_CODENAME"
        echo "  Raspberry Pi repository lines detected:"
        grep -RhsE '^[[:space:]]*deb .*raspberrypi|^[[:space:]]*URIs:.*raspberrypi' \
            /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null | sed 's/^/    /' || true
    fi
fi

if [[ "$HAILO_OK" -eq 1 ]]; then
    echo "  hailortcli: $(command -v hailortcli)"
    # A reboot can be required after DKMS/package installation.
    hailortcli fw-control identify || warn "hailortcli is installed but the Hailo device is not ready. Reboot, then check again."
else
    warn "Hailo runtime is not ready. The agent can still use local CPU Qwen/Thor, but Hailo camera detection will remain disabled."
    if command_exists lspci; then
        echo "  PCIe visibility:"
        lspci -nn | grep -iE 'hailo|co-processor' | sed 's/^/    /' || echo "    No Hailo PCIe device found by lspci."
    fi
fi

# ---------------------------------------------------------------------------
# 6. Piper TTS. Failure is non-fatal: preserve diagnostics and continue.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[7/9] Installing Piper TTS...${NC}"
if [[ ! -x piper/piper ]]; then
    rm -rf piper/*
    if wget -O /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz; then
        if ! tar -xzf /tmp/piper.tar.gz -C piper --strip-components=1; then
            warn "Piper archive downloaded but could not be extracted."
        fi
        rm -f /tmp/piper.tar.gz
    else
        warn "Piper binary download failed. TTS will not work until Piper is installed."
    fi
fi
if [[ ! -f piper/en_GB-semaine-medium.onnx ]]; then
    wget -O piper/en_GB-semaine-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx || warn "Could not download Piper voice model."
fi
if [[ ! -f piper/en_GB-semaine-medium.onnx.json ]]; then
    wget -O piper/en_GB-semaine-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json || warn "Could not download Piper voice metadata."
fi
curl -fL -o voices/bmo-custom.onnx \
    https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx || true
curl -fL -o voices/bmo-custom.onnx.json \
    https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx.json || true

# ---------------------------------------------------------------------------
# 7. whisper.cpp
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[8/9] Installing whisper.cpp speech recognition...${NC}"
if [[ ! -d whisper.cpp/.git ]]; then
    rm -rf whisper.cpp
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git whisper.cpp || warn "Could not clone whisper.cpp."
fi
if [[ -f whisper.cpp/CMakeLists.txt ]]; then
    cmake -S whisper.cpp -B whisper.cpp/build -DWHISPER_BUILD_EXAMPLES=ON && \
        cmake --build whisper.cpp/build -j"$(nproc)" || warn "whisper.cpp build failed."
    if [[ ! -f whisper.cpp/models/ggml-base.en.bin && -x whisper.cpp/models/download-ggml-model.sh ]]; then
        bash whisper.cpp/models/download-ggml-model.sh base.en || warn "Could not download whisper.cpp base.en model."
    fi
fi

# ---------------------------------------------------------------------------
# 8. Ollama/Qwen + wake word
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[9/9] Installing local Qwen and wake word...${NC}"
if ! command_exists ollama; then
    curl -fsSL https://ollama.com/install.sh | sh || warn "Ollama installation failed. Thor mode can still be used after configuration."
fi
if command_exists ollama; then
    sudo systemctl enable --now ollama >/dev/null 2>&1 || true
    ollama pull qwen3.5:2b || warn "Could not pull qwen3.5:2b. You can retry later with: ollama pull qwen3.5:2b"
fi

chmod +x install_wake_words.sh 2>/dev/null || true
if ! bash install_wake_words.sh; then
    warn "Wake-word installation was incomplete. You can retry later with: ./install_wake_words.sh"
fi

chmod +x start_agent.sh setup_jetson_thor.sh hardware_check.py install_wake_words.sh 2>/dev/null || true

echo
ok "Setup reached the end successfully."
echo "Expected wiring:"
echo "  GY-521 SDA -> GPIO2 / pin 3"
echo "  GY-521 SCL -> GPIO3 / pin 5"
echo "  GY-521 INT -> GPIO24 / pin 18"
echo "  MAX98357A DIN -> GPIO21 / pin 40"
echo "  MAX98357A BCLK -> GPIO18 / pin 12"
echo "  MAX98357A LRCLK -> GPIO19 / pin 35"
echo
if [[ "$I2C_REBOOT_REQUIRED" -eq 1 || "$AUDIO_REBOOT_REQUIRED" -eq 1 ]]; then
    printf '%b\n' "${YELLOW}REBOOT REQUIRED before I2C/I2S hardware checks.${NC}"
fi
echo "Next:"
echo "  sudo reboot"
echo "  cd '$BASE_DIR'"
echo "  source venv/bin/activate"
echo "  python hardware_check.py"
echo "  ./start_agent.sh"
