# Be More Agent — Raspberry Pi 5 + Hailo + Jetson Thor Edition

This version extends the original Be More Agent for the following target hardware:

- Raspberry Pi 5, 16 GB
- Raspberry Pi AI HAT+ 26 TOPS (Hailo-8)
- GY-521 / MPU-6050 accelerometer + gyro
- MAX98357A I2S amplifier
- Raspberry Pi Camera Module
- USB microphone
- Optional Jetson AGX Thor on the local network for larger LLM/VLM inference

The application remains offline-first. The default AI backend is the Raspberry Pi itself. You can switch to the Jetson Thor by speech when more model capacity is needed.

## Important: 26 TOPS Hailo-8 limitation

The 26 TOPS Raspberry Pi AI HAT+ contains a **Hailo-8**. Hailo-8 accelerates vision networks such as YOLO, segmentation, pose estimation, CLIP, and related neural workloads, but it **does not support Hailo's LLM/VLM GenAI runtime**. Hailo LLM/VLM support is provided by the newer Hailo-10H / Raspberry Pi AI HAT+ 2.

Because of that hardware limitation, this build uses the 26 TOPS HAT in the most useful supported role:

1. **Local text/VLM:** Qwen runs through Ollama on the Raspberry Pi 5 CPU.
2. **Hailo-8:** the camera image is also passed through Hailo object detection.
3. **Visual question:** the Hailo detections are supplied to the Qwen VLM as scene hints.
4. **Thor mode:** text and vision inference can be moved to a Jetson AGX Thor on the LAN.

If you later replace the AI HAT+ with an AI HAT+ 2 / Hailo-10H, the backend abstraction can be pointed at Hailo-Ollama without redesigning the agent.

---

## Wiring

### GY-521 / MPU-6050

| GY-521 | Raspberry Pi 5 | Physical pin |
|---|---|---:|
| VCC | 3.3 V | 1 or 17 |
| GND | GND | 6, 9, 14, etc. |
| SDA | GPIO2 / SDA1 | 3 |
| SCL | GPIO3 / SCL1 | 5 |
| INT | GPIO24 | 18 |

The code expects I2C bus 1 and address `0x68`.

### MAX98357A I2S amplifier

The MAX98357A is an **I2S** amplifier, not an I2C amplifier.

| MAX98357A | Raspberry Pi 5 | Physical pin |
|---|---|---:|
| DIN | GPIO21 / PCM_DOUT | 40 |
| BCLK | GPIO18 / PCM_CLK | 12 |
| LRC / LRCLK | GPIO19 / PCM_FS | 35 |
| GND | GND | any GND |
| VIN | suitable supply for your module | module-dependent |

`setup.sh` adds:

```text
dtparam=i2c_arm=on
dtoverlay=max98357a,no-sdmode
```

A reboot is required after those boot settings are added.

### USB microphone

`config.json` defaults to:

```json
"input_device": "USB"
```

The application selects the first capture device containing `USB` in its PortAudio/ALSA name. Set this to a more specific name or numeric device index if you have multiple USB audio devices.

---

## AI architecture

### Default local mode

Default model:

```text
qwen3.5:2b
```

Qwen 3.5 supports text, image input, and tools in Ollama. On the Pi 5 this model runs on the CPU because Hailo-8 cannot execute an LLM/VLM.

The local AI has native tool definitions for:

- `get_time`
- `search_web`
- `look_at_camera`
- `get_motion_state`

The LLM is not asked to emit JSON action strings anymore. Tool calls are handled through the model/backend tool interface.

### Hailo camera perception

`vision.py` uses Picamera2's Hailo device interface and a compatible Hailo YOLO HEF installed by the Raspberry Pi Hailo packages. When the user asks the robot to look at something:

1. `rpicam-still` captures the Pi Camera image.
2. The image is passed through Hailo-8 object detection.
3. The active Qwen VLM receives the image.
4. Hailo detections are included as additional scene hints.
5. The final camera result returns to the tool-calling text model.

If Hailo inference is unavailable, the VLM still receives the image and can answer; the application logs the Hailo error rather than disabling vision completely.

### Motion awareness

`hardware.py` initializes the MPU-6050 and monitors it continuously. It uses:

- accelerometer magnitude changes
- acceleration vector changes
- gyroscope activity
- the MPU-6050 motion interrupt on GPIO24

Every LLM turn receives a live system context similar to:

```text
The robot is being moved or was moved 0.4 seconds ago.
Accelerometer XYZ=(...) g; gyro XYZ=(...) deg/s.
```

The model can also explicitly call `get_motion_state`.

This gives the assistant physical-state awareness without forcing it to speak every time the chassis is touched.

---

## Jetson AGX Thor mode

The default remote backend is named `thor` and expects an Ollama server at:

```text
http://jetson-thor.local:11434
```

The default Thor text and vision model is:

```text
qwen3.5:27b
```

Run the included helper **on the Jetson Thor**:

```bash
chmod +x setup_jetson_thor.sh
./setup_jetson_thor.sh
```

That installs/updates native Ollama, exposes it on the LAN, and pulls the configured Qwen model. If mDNS does not resolve `jetson-thor.local`, replace `ai.backends.thor.base_url` in `config.json` with the Thor's LAN IP, for example:

```json
"base_url": "http://192.168.1.50:11434"
```

Binding Ollama to `0.0.0.0` exposes the service to machines that can reach that network interface. Restrict port 11434 to your trusted LAN using your firewall/router.

### Optional vLLM / OpenAI-compatible Thor backend

`ai_backend.py` also supports an OpenAI-compatible endpoint. Change the Thor backend to something like:

```json
{
  "type": "vllm",
  "base_url": "http://192.168.1.50:8000/v1",
  "api_key": "not-needed",
  "text_model": "Qwen/Qwen3.5-27B",
  "vision_model": "Qwen/Qwen3.5-27B"
}
```

You can also use separate `text_base_url` and `vision_base_url` values if you run separate text and VLM servers.

---

## Speech switching

Backend switching is handled before the prompt reaches the LLM.

Examples:

```text
Switch to Thor.
Use the server.
Switch to local.
Use the Raspberry Pi.
What model are you using?
Switch model to qwen3.5 2b.
Switch vision model to qwen3.5 0.8b.
```

The configured model must already exist on the selected backend before a spoken model switch succeeds.

The default backend after every application start is `local`.

---

## Installation on the Raspberry Pi 5

Use current 64-bit Raspberry Pi OS Trixie.

```bash
chmod +x setup.sh
./setup.sh
sudo reboot
```

After reboot:

```bash
cd be-more-agent
source venv/bin/activate
python hardware_check.py
python agent.py
```

`setup.sh` installs or configures:

- Raspberry Pi camera tools / Picamera2
- Hailo AI HAT+ packages (`hailo-all`)
- I2C tools
- GPIO Zero + lgpio for Pi 5 GPIO
- MAX98357A device-tree overlay
- Piper TTS
- whisper.cpp + `base.en`
- Ollama
- Qwen 3.5 2B
- OpenWakeWord
- Python dependencies

---

## Hardware diagnostics

Run:

```bash
source venv/bin/activate
python hardware_check.py
```

It checks:

- I2C bus and MPU-6050 `WHO_AM_I`
- Hailo device identification
- Raspberry Pi camera discovery
- ALSA playback devices
- ALSA capture devices
- local Ollama models
- Thor Ollama connectivity

For the GY-521, `i2cdetect -y 1` should normally show `68`.

For the AI HAT+, this command should identify a Hailo device:

```bash
hailortcli fw-control identify
```

---

## Configuration reference

The main sections in `config.json` are:

```text
ai.backends.local       Raspberry Pi Ollama/Qwen
ai.backends.thor        Jetson Thor LLM/VLM endpoint
hardware.mpu6050        GY-521 address, GPIO24 interrupt, thresholds
hardware.max98357a      documented I2S pin assignment
camera                  Pi Camera and Hailo detector settings
input_device            USB microphone selector
output_device           MAX98357A ALSA/PortAudio selector
voice_model             Piper ONNX voice
```

### Motion tuning

If normal vibration triggers movement too easily, raise:

```json
"accel_delta_threshold_g": 0.08,
"accel_magnitude_threshold_g": 0.12,
"gyro_threshold_dps": 18.0
```

If movement is not detected reliably, lower them gradually.

---

## Project structure additions

```text
be-more-agent/
├── agent.py                 Main GUI, speech loop, tool-calling agent
├── ai_backend.py            Local/Thor Ollama and OpenAI-compatible routing
├── hardware.py              MPU-6050 + GPIO24 movement monitor
├── vision.py                Pi Camera + Hailo-8 object perception
├── hardware_check.py        Raspberry Pi hardware diagnostics
├── setup_jetson_thor.sh     Optional Thor Ollama server setup
├── config.json              Hardware/model/backend configuration
├── assets/coco.txt          Hailo object labels
├── setup.sh                 Raspberry Pi 5 setup
└── start_agent.sh           Runtime launcher with Pi 5 GPIO backend
```

## License

The original software is MIT licensed. This modification retains that software basis and does not change the original project's third-party asset/license obligations.
