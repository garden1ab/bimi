#!/usr/bin/env bash
set -u -o pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

if [[ ! -x venv/bin/python ]]; then
    echo "ERROR: ./venv is missing. Run ./setup.sh first." >&2
    exit 1
fi

source venv/bin/activate
mkdir -p wakewords

echo "Downloading official openWakeWord feature + pretrained models..."
python - <<'PY_WAKE'
import openwakeword
openwakeword.utils.download_models(model_names=["hey_jarvis", "hey_mycroft"])
print("Official wake-word models installed.")
PY_WAKE

echo "Downloading community Hey BMO classifier..."
if ! curl -fL -o wakewords/hey_bmo.onnx \
  'https://huggingface.co/amrmantawi/bmo-openwakeword/resolve/main/hey_beemo.onnx?download=true'; then
    echo "WARNING: Hey BMO download failed. Hey Jarvis and Hey Mycroft remain available." >&2
    rm -f wakewords/hey_bmo.onnx
fi

# Legacy fallback used by older configs/builds.
if [[ ! -f wakeword.onnx ]]; then
    curl -fL -o wakeword.onnx \
      https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx \
      || echo "WARNING: legacy Hey Jarvis fallback download failed." >&2
fi

python - <<'PY_VERIFY'
from pathlib import Path
import openwakeword
print("\nWake-word status:")
for phrase, name in [("Hey Jarvis", "hey_jarvis"), ("Hey Mycroft", "hey_mycroft")]:
    meta = openwakeword.MODELS.get(name, {})
    path = str(meta.get("model_path", "")).replace(".tflite", ".onnx")
    print(f"  {phrase:12} {'OK' if path and Path(path).exists() else 'MISSING'}  {path}")
bmo = Path("wakewords/hey_bmo.onnx")
print(f"  {'Hey BMO':12} {'OK' if bmo.exists() else 'MISSING'}  {bmo}")
PY_VERIFY

echo
echo "Wake words configured: Hey BMO, Hey Jarvis, Hey Mycroft"
