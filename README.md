# Be More Agent (BMO)

Voice-first embodied assistant for Raspberry Pi 5, with optional remote
inference on a Jetson AGX Thor.

The Pi owns all robot I/O (microphone, speaker, camera, IMU, Hailo). Thor is an
inference service only. See `ARCHITECTURE.md` for the full design.

## Quick start

```bash
./setup.sh                 # system packages, venv, whisper.cpp, piper, ollama
./install_wake_words.sh    # REQUIRED for wake phrases (see note below)
./start_agent.sh
```

Always launch via `start_agent.sh`. It `cd`s to the project directory, and the
agent resolves `config.json`, `faces/`, `sounds/`, `whisper.cpp/` and `piper/`
by relative path.

## Controls

| Key | In IDLE | While LISTENING | While THINKING / SPEAKING |
|---|---|---|---|
| Enter | Start listening | **Finish speaking and process** | Cancel and re-listen |
| Space | — | — | Interrupt speech |
| Escape | Exit | Exit | Exit |

Wake phrases: **Hey BMO**, **Hey Jarvis**, **Hey Mycroft**.

`runtime.keyboard_trigger_mode` in `config.json`:

- `vad` (default) — Enter starts; recording ends on trailing silence *or* a
  second Enter press.
- `toggle` / `ptt` — Enter starts, Enter stops. No silence detection.

## Troubleshooting

Diagnostics first:

```bash
./hardware_check.py
```

### BMO cannot hear me

At startup the agent runs a capture self-test and prints:

```
[MICTEST] device='...' rate=48000 peak=0.31 rms=0.04 (speech must exceed min_rms=0.006)
```

- `peak=0.00000` → the device returns digital silence. Either the wrong capture
  device is selected (`input_device` in `config.json`, matched by substring
  against the PortAudio device name) or the ALSA capture control is muted. Run
  `alsamixer -c <card>`, press **F4**, unmute and raise Mic.
- Very low `rms` → raise the capture gain, or lower `recording.min_rms`.

Set `runtime.debug_audio: true` to log live input peaks while waiting for a
wake word.

### The wake word never triggers

Every wake model depends on openWakeWord's **shared feature models**
(melspectrogram + embedding). Without them no phrase can ever match, even
though `wakeword.onnx` ships with this repo. The agent now says so explicitly:

```
[WAKE] openWakeWord's shared feature models (melspectrogram / embedding) are missing...
```

Fix: `./install_wake_words.sh`. Enter-only mode keeps working meanwhile.

### Enter / Space / Escape do nothing

Usually the fullscreen window never received keyboard focus from the window
manager. The agent now claims focus at startup and re-asserts it, but if the
keys are still dead, click the face once to focus the window.

### Nothing responds at all

If launched without a terminal (autostart, `systemd`, `nohup`), stdin is at EOF.
That is handled now, but confirm the launcher uses `start_agent.sh`.

The watchdog reports a wedged (not crashed) worker after
`runtime.stall_timeout_seconds` (default 180s, minimum 60s):

```
[WATCHDOG] No progress for 180s; forcing recovery.
```

### Thor is not used

`[AI ROUTE] requested=thor actual=local` means Thor failed and the agent fell
back. Verify from the Pi:

```bash
curl http://<THOR-IP>:11434/api/tags
```

Prefer a static IP over `jetson-thor.local` until mDNS is proven stable.

## Log prefixes

`[RUNTIME] [INIT] [WAKE] [MICTEST] [AUDIO] [INPUT] [STT] [AI] [AI ROUTE]`
`[AI PERF] [TOOL] [TTS] [MPU6050] [HAILO] [WATCHDOG] [RECOVERY] [UI] [STATE]`

## Notes

- Thor/Ollama setup notes: `docs-thor-setup-notes.md`.
- Port 11434 is unauthenticated. Keep it on a trusted LAN; never port-forward.
- Model/backend switches made by voice are session-only and are not written
  back to `config.json`.
- The MPU-6050 has no magnetometer, so absolute yaw is reported as unavailable
  by design.
