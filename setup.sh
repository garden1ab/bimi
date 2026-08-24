#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo -e "${GREEN}Be More Agent - Raspberry Pi 5 + Hailo setup${NC}"

if [ "$(uname -m)" != "aarch64" ]; then
    echo -e "${YELLOW}Warning: this setup is intended for 64-bit Raspberry Pi OS on Raspberry Pi 5.${NC}"
fi

# ---------------------------------------------------------------------------
# 1. Raspberry Pi / Hailo / audio / I2C system packages
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[1/8] Installing Raspberry Pi system dependencies...${NC}"
sudo apt update
sudo apt install -y \
    python3-tk python3-dev python3-venv python3-picamera2 python3-gpiozero python3-lgpio \
    libasound2-dev portaudio19-dev liblapack-dev libblas-dev cmake build-essential \
    espeak-ng git curl wget i2c-tools rpicam-apps dkms

# AI HAT+ (Hailo-8 / Hailo-8L) package. This is intentionally hailo-all,
# not hailo-h10-all; the latter is for the AI HAT+ 2 / Hailo-10H.
if apt-cache show hailo-all >/dev/null 2>&1; then
    sudo apt install -y hailo-all
else
    echo -e "${RED}hailo-all is not available from the configured Raspberry Pi OS repositories.${NC}"
    echo "Update to current 64-bit Raspberry Pi OS Trixie and rerun setup if Hailo support is missing."
fi

# ---------------------------------------------------------------------------
# 2. Configure I2C and MAX98357A I2S audio pins
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[2/8] Configuring I2C and MAX98357A I2S audio...${NC}"
BOOT_CONFIG="/boot/firmware/config.txt"
if [ ! -f "$BOOT_CONFIG" ] && [ -f /boot/config.txt ]; then
    BOOT_CONFIG="/boot/config.txt"
fi

add_boot_line() {
    local line="$1"
    if ! grep -Fxq "$line" "$BOOT_CONFIG"; then
        echo "$line" | sudo tee -a "$BOOT_CONFIG" >/dev/null
        echo "  added: $line"
    fi
}

if [ -f "$BOOT_CONFIG" ]; then
    add_boot_line "dtparam=i2c_arm=on"
    # MAX98357A standard Raspberry Pi I2S mapping:
    # BCLK GPIO18, LRCLK/FS GPIO19, DIN receives Pi DOUT on GPIO21.
    # no-sdmode is used because the requested wiring does not include SD_MODE.
    add_boot_line "dtoverlay=max98357a,no-sdmode"
else
    echo -e "${RED}Could not find Raspberry Pi boot config; configure I2C/I2S manually.${NC}"
fi

# ---------------------------------------------------------------------------
# 3. Project folders
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[3/8] Creating runtime folders...${NC}"
mkdir -p piper voices assets
for d in greeting_sounds thinking_sounds ack_sounds error_sounds; do mkdir -p "sounds/$d"; done
for d in idle listening thinking speaking error capturing warmup; do mkdir -p "faces/$d"; done

# ---------------------------------------------------------------------------
# 4. Piper TTS
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[4/8] Installing Piper TTS and voice...${NC}"
if [ ! -x piper/piper ]; then
    rm -rf piper/*
    wget -O /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
    tar -xzf /tmp/piper.tar.gz -C piper --strip-components=1
    rm -f /tmp/piper.tar.gz
fi
if [ ! -f piper/en_GB-semaine-medium.onnx ]; then
    wget -O piper/en_GB-semaine-medium.onnx \
        https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx || true
fi
if [ ! -f piper/en_GB-semaine-medium.onnx.json ]; then
    wget -O piper/en_GB-semaine-medium.onnx.json \
        https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/semaine/medium/en_GB-semaine-medium.onnx.json || true
fi

# Preserve the original project's optional custom voice downloads.
curl -fL -o voices/bmo-custom.onnx \
    https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx || true
curl -fL -o voices/bmo-custom.onnx.json \
    https://github.com/brenpoly/be-more-agent/releases/latest/download/bmo.onnx.json || true

# ---------------------------------------------------------------------------
# 5. Python virtual environment. System packages are required so Picamera2,
#    lgpio, and Hailo's apt-installed Python bindings are visible.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[5/8] Installing Python dependencies...${NC}"
if [ ! -d venv ]; then
    python3 -m venv --system-site-packages venv
else
    # Make an existing environment see Raspberry Pi/Hailo apt packages.
    if [ -f venv/pyvenv.cfg ]; then
        sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' venv/pyvenv.cfg || true
    fi
fi
source venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install --force-reinstall --no-cache-dir sounddevice
python -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 6. whisper.cpp local STT
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[6/8] Installing whisper.cpp speech recognition...${NC}"
if [ ! -d whisper.cpp/.git ]; then
    rm -rf whisper.cpp
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git whisper.cpp
fi
cmake -S whisper.cpp -B whisper.cpp/build -DWHISPER_BUILD_EXAMPLES=ON
cmake --build whisper.cpp/build -j"$(nproc)"
if [ ! -f whisper.cpp/models/ggml-base.en.bin ]; then
    bash whisper.cpp/models/download-ggml-model.sh base.en
fi

# ---------------------------------------------------------------------------
# 7. Ollama + local Qwen model. On the 26 TOPS Hailo-8 HAT, Ollama runs on
#    the Pi CPU; Hailo-8 accelerates the camera perception path instead.
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[7/8] Installing local Qwen AI...${NC}"
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
sudo systemctl enable --now ollama >/dev/null 2>&1 || true
ollama pull qwen3.5:2b

# ---------------------------------------------------------------------------
# 8. Wake word and final checks
# ---------------------------------------------------------------------------
echo -e "${YELLOW}[8/8] Installing wake word and checking hardware...${NC}"
if [ ! -f wakeword.onnx ]; then
    curl -fL -o wakeword.onnx \
        https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/hey_jarvis_v0.1.onnx
fi

chmod +x start_agent.sh setup_jetson_thor.sh 2>/dev/null || true

if command -v hailortcli >/dev/null 2>&1; then
    echo "Hailo device check:"
    hailortcli fw-control identify || true
fi

echo
printf '%b\n' "${GREEN}Setup complete.${NC}"
echo "Wiring expected by config.json:"
echo "  GY-521 SDA -> GPIO2 (pin 3), SCL -> GPIO3 (pin 5), INT -> GPIO24 (pin 18)"
echo "  MAX98357A DIN -> GPIO21 (pin 40), BCLK -> GPIO18 (pin 12), LRCLK -> GPIO19 (pin 35)"
echo "A reboot is required if I2C or the MAX98357A overlay was newly enabled."
echo "After reboot: source venv/bin/activate && python hardware_check.py && python agent.py"
