# =========================================================================
#  Be More Agent - Raspberry Pi 5 / Hailo / GY-521 / Jetson Thor edition
#
#  Based on the original Be More Agent project by brenpoly (MIT License).
# =========================================================================

from __future__ import annotations

import atexit
import datetime
import json
import os
import random
import re
import select
import queue
import subprocess
import sys
import threading
import time
import traceback
import warnings
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk

import numpy as np
import scipy.signal
import sounddevice as sd
from PIL import Image, ImageTk

import openwakeword  # noqa: F401 - retained so setup validates the dependency
from openwakeword.model import Model

try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS

from ai_backend import AIBackendError, AIBackendManager
from hardware import MPU6050Monitor
from vision import CameraVision

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================
# 1. CONFIGURATION
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": True,
    "system_prompt_extras": "",
    "input_device": "USB",
    "input_sample_rate": None,
    "output_device": "MAX98357A",
    "wake_word": {
        "enabled": True,
        "threshold": 0.5,
        "models": [
            {"phrase": "Hey BMO", "path": "wakewords/hey_bmo.onnx"},
            {"phrase": "Hey Jarvis", "model": "hey_jarvis"},
            {"phrase": "Hey Mycroft", "model": "hey_mycroft"}
        ],
        "legacy_model": "wakeword.onnx"
    },
    "recording": {
        "start_timeout_seconds": 6.0,
        "end_silence_seconds": 1.0,
        "max_record_seconds": 20.0,
        "min_speech_seconds": 0.25,
        "noise_calibration_seconds": 0.35,
        "pre_roll_seconds": 0.25,
        "chunk_seconds": 0.05,
        "min_rms": 0.006,
        "noise_multiplier": 2.4,
        "end_noise_multiplier": 1.7,
        "speech_start_chunks": 2
    },
    "memory": {
        "local_context_messages": 6,
        "thor_context_messages": 16,
        "saved_messages": 16
    },
    "thinking_audio": {
        "enabled": True,
        "delay_seconds": 0.25,
        "announce_once": True
    },
    "ai": {
        "default_backend": "local",
        "fallback_to_local": True,
        "backends": {
            "local": {
                "type": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "text_model": "qwen3.5:2b",
                "vision_model": "qwen3.5:2b",
                "text_models": ["qwen3.5:0.8b", "qwen3.5:2b", "qwen3:1.7b"],
                "vision_models": ["qwen3.5:0.8b", "qwen3.5:2b"],
                "aliases": ["local", "pi", "raspberry pi"],
                "keep_alive": -1,
                "think": False,
                "options": {
                    "num_ctx": 2048,
                    "num_predict": 96,
                    "temperature": 0.4
                },
            },
            "thor": {
                "type": "ollama",
                "base_url": "http://jetson-thor.local:11434",
                "text_model": "qwen3.6:35b",
                "vision_model": "qwen3.6:35b",
                "text_models": ["qwen3.6:35b", "qwen3.6:35b-a3b-nvfp4", "qwen3.8:27b", "qwen3.5:35b"],
                "vision_models": ["qwen3.6:35b", "qwen3.6:35b-a3b-nvfp4", "qwen3.8:27b"],
                "aliases": ["thor", "server", "jetson", "jetson thor"],
                "keep_alive": -1,
                "think": False,
                "options": {
                    "num_ctx": 8192,
                    "num_predict": 256,
                    "temperature": 0.5
                },
            },
        },
    },
    "hardware": {
        "mpu6050": {
            "enabled": True,
            "i2c_bus": 1,
            "address": "0x68",
            "sda_gpio": 2,
            "scl_gpio": 3,
            "interrupt_gpio": 24,
            "poll_hz": 20,
            "accel_delta_threshold_g": 0.08,
            "accel_magnitude_threshold_g": 0.12,
            "gyro_threshold_dps": 18.0,
            "motion_hold_seconds": 2.5,
            "interrupt_threshold_mg": 80,
            "interrupt_duration_ms": 20,
        },
        "max98357a": {
            "enabled": True,
            "din_gpio": 21,
            "bclk_gpio": 18,
            "lrclk_gpio": 19,
        },
    },
    "camera": {
        "image_path": "current_image.jpg",
        "width": 640,
        "height": 480,
        "rotation": 180,
        "camera_warmup_ms": 500,
        "hailo_enabled": True,
        "hailo_model_path": None,
        "hailo_threshold": 0.45,
        "max_detections": 12,
        "labels_path": "assets/coco.txt",
    },
}

OLLAMA_OPTIONS = {
    "num_thread": 4,
    "temperature": 0.5,
    "top_k": 20,
    "top_p": 0.9,
    # Backend-specific options in config.json override these defaults.
    "num_ctx": 4096,
    "num_predict": 160,
}

BASE_SYSTEM_PROMPT = """You are the conversational AI inside a small Raspberry Pi 5 robot.
Be concise, useful, and natural. You have live tools for the clock, web search, the camera, and the robot's motion sensor.

Rules:
- Use look_at_camera whenever the user asks what you see, asks about the surroundings, or asks a question that requires visual inspection. Never guess what the camera sees.
- Use get_motion_state when a question depends on whether the robot is moving or was moved.
- Use get_orientation whenever the user asks for angle, tilt, orientation, level, pose, or position relative to rest. Do not estimate these from words; use the live MPU-6050 reading.
- If the user asks to make the current physical pose the new zero/rest pose, use set_rest_orientation.
- The MPU-6050 can determine gravity-referenced roll/pitch and total tilt, but it cannot provide reliable absolute yaw without a magnetometer.
- Use get_time for the current local time.
- Use search_web only when fresh public information is needed.
- A live sensor-state system message is included on every turn. Treat it as current physical state.
- Backend/model switching is handled by the host application when the user explicitly says things like "switch to Thor", "use local", or "switch model to qwen...".
- Do not output hand-written JSON action commands; use the provided function tools.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the robot's current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the public web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_camera",
            "description": "Capture the Raspberry Pi camera and use the active VLM plus Hailo perception to inspect the scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What should be inspected or described in the camera image?",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_motion_state",
            "description": "Read whether the robot is moving or was recently moved, including accelerometer/gyro values and rest-relative tilt.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orientation",
            "description": "Get a fresh MPU-6050 orientation reading: roll, pitch, and total tilt in degrees relative to the calibrated rest pose. Yaw is unavailable without a magnetometer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_rest_orientation",
            "description": "Calibrate the robot's current stationary pose as zero/rest orientation for future roll, pitch, and tilt measurements.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
            return deep_merge(DEFAULT_CONFIG, json.load(handle))
    except Exception as exc:
        print(f"[CONFIG] Failed to read {CONFIG_FILE}: {exc}. Using defaults.", flush=True)
        return DEFAULT_CONFIG


CURRENT_CONFIG = load_config()
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n" + CURRENT_CONFIG.get("system_prompt_extras", "")


def resolve_audio_device(requested: Any, kind: str) -> Optional[int]:
    if requested in (None, "", "default"):
        return None
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"[AUDIO] Device query failed: {exc}", flush=True)
        return None

    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        return index if 0 <= index < len(devices) else None

    requested_lower = str(requested).lower()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"

    search_terms = [requested_lower]
    # The Raspberry Pi MAX98357A device-tree overlay can appear in PortAudio
    # with a generic ALSA simple-card name instead of the codec part number.
    if kind == "output" and "max98357" in requested_lower:
        search_terms.extend(["max98357", "snd_rpi_simple", "simple-card", "i2s"])

    for term in search_terms:
        for idx, device in enumerate(devices):
            if device.get(channel_key, 0) > 0 and term in device.get("name", "").lower():
                print(f"[AUDIO] Using {kind}: {device.get('name')} (index {idx})", flush=True)
                return idx
    print(f"[AUDIO] Requested {kind} device '{requested}' not found; using system default.", flush=True)
    return None


INPUT_DEVICE = resolve_audio_device(CURRENT_CONFIG.get("input_device"), "input")
OUTPUT_DEVICE = resolve_audio_device(CURRENT_CONFIG.get("output_device"), "output")


def choose_input_samplerate(device: Optional[int], preferred: Optional[int] = None) -> int:
    candidates: List[int] = []
    if preferred:
        candidates.append(int(preferred))
    try:
        info = sd.query_devices(device, "input")
        if info.get("default_samplerate"):
            candidates.append(int(info["default_samplerate"]))
    except Exception:
        pass
    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            pass
    return 16000


def piper_sample_rate(model_path: str) -> int:
    for candidate in (model_path + ".json", str(Path(model_path).with_suffix(".onnx.json"))):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return int(data.get("audio", {}).get("sample_rate") or data.get("sample_rate") or 22050)
        except Exception:
            pass
    return 22050


class BotStates:
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    CAPTURING = "capturing"
    WARMUP = "warmup"


greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"


# =========================================================================
# 2. GUI + AGENT
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Pi Assistant")
        master.attributes("-fullscreen", True)
        # bind_all keeps recovery keys working regardless of which widget owns focus.
        master.bind_all("<Escape>", self.exit_fullscreen)
        master.bind_all("<Return>", self.handle_ptt_toggle)
        master.bind_all("<KP_Enter>", self.handle_ptt_toggle)
        master.bind_all("<space>", self.handle_speaking_interrupt)

        self.current_state = BotStates.WARMUP
        self.current_volume = 0
        self.animations: Dict[str, List[ImageTk.PhotoImage]] = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        self.last_camera_path: Optional[str] = None

        self.history = self.load_chat_history()
        self.session_memory: List[Dict[str, str]] = []

        self.thinking_sound_active = threading.Event()
        self.ptt_event = threading.Event()
        self.recording_active = threading.Event()
        self.interrupted = threading.Event()
        self.tts_active = threading.Event()
        self.tts_queue: List[str] = []
        self.tts_queue_lock = threading.Lock()
        self.tts_thread: Optional[threading.Thread] = None
        self.main_thread: Optional[threading.Thread] = None
        self.watchdog_thread: Optional[threading.Thread] = None
        self.motion_init_thread: Optional[threading.Thread] = None
        self.warmup_thread: Optional[threading.Thread] = None
        self.wake_loader_thread: Optional[threading.Thread] = None
        self.wake_model_ready = threading.Event()
        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self.current_audio_process: Optional[subprocess.Popen] = None
        self.exiting = False
        self.last_ptt_time = 0.0
        self.last_progress_time = time.monotonic()
        self.warmed_up = False

        # Guards every "start this worker if it is not already running" check.
        # Without it the watchdog and agent-main can both observe a thread that
        # has been assigned but not yet started and each spawn their own copy.
        self._thread_lock = threading.RLock()
        # Set once stdin is known to be closed/EOF (autostart, systemd, nohup).
        # select() reports an EOF stdin as permanently readable, so the terminal
        # fallback has to be disabled instead of firing on every loop pass.
        self._stdin_usable = self._stdin_is_interactive()

        self.ai = AIBackendManager(CURRENT_CONFIG.get("ai", {}), OLLAMA_OPTIONS)
        self.motion = MPU6050Monitor(CURRENT_CONFIG.get("hardware", {}).get("mpu6050", {}))
        self.vision = CameraVision(CURRENT_CONFIG.get("camera", {}))

        # IMPORTANT: never initialize I2C hardware synchronously here. SMBus/I2C
        # transactions can stall when the sensor/bus is disconnected or wedged.
        # The previous build called motion.start() before the wake/main threads,
        # which could leave the GUI visible but with no active microphone listener.
        # Motion initialization is started asynchronously after input is online.

        # Wake-word model construction can take seconds on a Pi and must never
        # run on Tk's event thread. Configure metadata here; the ONNX models are
        # loaded by a daemon worker after the GUI event loop is alive.
        self.oww_model = None
        self.wake_word_threshold = float(CURRENT_CONFIG.get("wake_word", {}).get("threshold", 0.5))
        self.wake_word_phrases: List[str] = []
        self.wake_word_labels: Dict[str, str] = {}

        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind("<Button-1>", self.toggle_hud_visibility)

        self.overlay_label = tk.Label(master, bg="black")
        self.overlay_label.bind("<Button-1>", self.toggle_hud_visibility)

        self.response_text = tk.Text(
            master,
            height=6,
            width=60,
            wrap=tk.WORD,
            state=tk.DISABLED,
            bg="#ffffff",
            fg="#000000",
            font=("Arial", 12),
        )
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)

        self.load_animations()
        self.update_animation()
        self.master.after(50, self._drain_ui_queue)

        # Do not start worker threads until Tk's mainloop has had a chance to
        # start. Calling Tk methods from a worker before mainloop is running can
        # deadlock Tcl/Tk and make Escape/Enter appear completely frozen.
        self.master.after(100, self._start_runtime_workers)

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------
    def safe_exit(self):
        """Immediate non-blocking UI shutdown.

        Hardware/network cleanup must not run on Tk's event thread. A wedged
        PortAudio/Ollama/I2C call previously made Escape look broken because the
        key handler itself blocked. Worker threads are daemon threads, so closing
        the Tk window is enough to terminate the process cleanly.
        """
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN REQUESTED ---", flush=True)
        self.interrupted.set()
        self.ptt_event.set()
        self.recording_active.clear()
        self.thinking_sound_active.clear()
        try:
            self.save_chat_history()
        except Exception:
            pass
        # Best-effort cleanup occurs off the GUI thread. Never wait for it.
        threading.Thread(target=self._background_cleanup, daemon=True, name="shutdown-cleanup").start()
        try:
            self.master.destroy()
        except Exception:
            pass

    def _background_cleanup(self):
        try:
            with self.tts_queue_lock:
                self.tts_queue.clear()
        except Exception:
            pass
        process = self.current_audio_process
        if process:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass
        try:
            self.motion.stop()
        except Exception:
            pass
        try:
            self.vision.close()
        except Exception:
            pass
        # Do not call ai.unload() here. A dead Ollama endpoint can block during
        # shutdown; the server can manage model residency independently.

    def exit_fullscreen(self, event=None):
        self.safe_exit()
        return "break"

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError:
            pass

    def handle_ptt_toggle(self, event=None):
        """Queue a manual wake without blocking Tk's event thread."""
        now = time.time()
        if now - self.last_ptt_time < 0.35:
            return "break"
        self.last_ptt_time = now

        keyboard_mode = str(CURRENT_CONFIG.get("runtime", {}).get("keyboard_trigger_mode", "vad")).lower()
        if keyboard_mode in {"toggle", "hold", "ptt"}:
            if self.recording_active.is_set():
                self.recording_active.clear()
                return "break"
            self.recording_active.set()

        busy_states = {
            BotStates.LISTENING, BotStates.THINKING, BotStates.SPEAKING,
            BotStates.CAPTURING, BotStates.ERROR,
        }
        if self.current_state in busy_states:
            print(f"[RECOVERY] Enter pressed while {self.current_state}; cancellation queued.", flush=True)
            # Set the flags immediately, then perform potentially blocking audio/
            # HTTP cancellation in a daemon worker. Tk returns to its event loop
            # immediately so Enter and Escape stay responsive.
            self.interrupted.set()
            self.ptt_event.set()
            threading.Thread(
                target=self._interrupt_current_work,
                kwargs={"queue_ptt": True},
                daemon=True,
                name="manual-recovery",
            ).start()
            return "break"

        self.ptt_event.set()
        self._touch_progress("manual wake queued")
        return "break"

    def handle_speaking_interrupt(self, event=None):
        if self.current_state not in (BotStates.SPEAKING, BotStates.THINKING):
            return "break"
        self.interrupted.set()
        threading.Thread(
            target=self._interrupt_current_work,
            kwargs={"queue_ptt": False},
            daemon=True,
            name="speech-interrupt",
        ).start()
        return "break"

    def load_animations(self):
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"]
        for state in states:
            frames: List[ImageTk.PhotoImage] = []
            folder = os.path.join("faces", state)
            if os.path.exists(folder):
                for filename in sorted(f for f in os.listdir(folder) if f.lower().endswith(".png")):
                    try:
                        image = Image.open(os.path.join(folder, filename)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                        frames.append(ImageTk.PhotoImage(image))
                    except Exception:
                        pass
            self.animations[state] = frames

        if not self.animations.get("idle"):
            blank = Image.new("RGB", (self.BG_WIDTH, self.BG_HEIGHT), color="#0000FF")
            self.animations["idle"] = [ImageTk.PhotoImage(blank)]
        for state in states:
            if not self.animations[state]:
                self.animations[state] = self.animations["idle"]

    def update_animation(self):
        if self.exiting:
            return
        state = self.current_state
        try:
            frames = self.animations.get(state) or self.animations.get(BotStates.IDLE) or []
            # An empty frame list used to raise ZeroDivisionError on the modulo
            # below and permanently kill the animation loop.
            if frames:
                if state == BotStates.SPEAKING and len(frames) > 1:
                    self.current_frame_index = random.randint(1, len(frames) - 1)
                else:
                    self.current_frame_index = (self.current_frame_index + 1) % len(frames)
                self.background_label.config(image=frames[self.current_frame_index])
        except Exception as exc:
            if not self.exiting:
                print(f"[UI] Animation frame skipped: {exc}", flush=True)
        try:
            self.master.after(50 if state == BotStates.SPEAKING else 500, self.update_animation)
        except tk.TclError:
            pass

    def set_state(self, state: str, msg: str = "", cam_path: Optional[str] = None):
        # Logical state is thread-safe enough for event routing. Widget mutation
        # is marshalled through ui_queue and performed only by Tk's main thread.
        if self.current_state != state:
            self.current_state = state
            self.current_frame_index = 0
        self._touch_progress(f"state={state}")
        try:
            self.ui_queue.put_nowait(("state", state, msg, cam_path))
        except Exception:
            pass

    def append_to_text(self, text: str, newline: bool = True):
        try:
            self.ui_queue.put_nowait(("text", text, newline))
        except Exception:
            pass

    def _drain_ui_queue(self):
        if self.exiting:
            return
        try:
            while True:
                item = self.ui_queue.get_nowait()
                kind = item[0]
                if kind == "state":
                    _, state, msg, cam_path = item
                    if msg:
                        print(f"[STATE] {str(state).upper()}: {msg}", flush=True)
                        self.status_var.set(msg)
                    show_path = cam_path if cam_path and os.path.exists(cam_path) else None
                    if show_path and state in (BotStates.THINKING, BotStates.SPEAKING):
                        try:
                            image = Image.open(show_path)
                            image.thumbnail((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                            self.current_overlay_image = ImageTk.PhotoImage(image)
                            self.overlay_label.config(image=self.current_overlay_image)
                            self.overlay_label.place(relx=0.5, rely=0.40, anchor=tk.CENTER)
                            continue
                        except Exception:
                            pass
                    self.overlay_label.place_forget()
                elif kind == "text":
                    _, text, newline = item
                    self.response_text.config(state=tk.NORMAL)
                    self.response_text.insert(tk.END, text + ("\n" if newline else ""))
                    self.response_text.see(tk.END)
                    self.response_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        except Exception as exc:
            # This previously returned on TclError and let any other exception
            # escape, and in both cases the reschedule below was skipped. One
            # transient widget error therefore killed the UI updater for good:
            # the agent kept working but the screen never changed again.
            if not self.exiting:
                print(f"[UI] Drain error ignored: {exc}", flush=True)
        if self.exiting:
            return
        try:
            self.master.after(50, self._drain_ui_queue)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Main listening loop
    # ------------------------------------------------------------------
    @staticmethod
    def _stdin_is_interactive() -> bool:
        """Report whether the CLI Enter fallback can safely be polled.

        When the agent is launched from an autostart entry, a systemd unit, or
        with `nohup`, stdin is closed or attached to /dev/null. select() then
        reports it readable forever because EOF counts as readable, so polling
        it turns every loop iteration into a phantom Enter press.
        """
        try:
            return bool(sys.stdin) and sys.stdin.isatty()
        except Exception:
            return False

    def _touch_progress(self, reason: str = ""):
        self.last_progress_time = time.monotonic()
        if reason and CURRENT_CONFIG.get("runtime", {}).get("debug_progress", False):
            print(f"[WATCHDOG] progress: {reason}", flush=True)

    def _ensure_tts_worker(self):
        if self.exiting:
            return
        # agent-main, the watchdog, and error recovery all call this. Without the
        # lock they can each see a thread that was assigned but not yet started
        # and spawn a second worker; two workers then pop the same queue and
        # fight over the single audio output device.
        with self._thread_lock:
            if self.tts_thread is None or not self.tts_thread.is_alive():
                if self.tts_thread is not None:
                    print("[WATCHDOG] Restarting stopped TTS worker.", flush=True)
                self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True, name="tts-worker")
                self.tts_thread.start()

    def _start_runtime_workers(self):
        """Called by Tk after mainloop starts; never blocks the GUI."""
        if self.exiting:
            return
        print("[RUNTIME] Tk event loop online; starting voice/background workers.", flush=True)
        self._start_wake_loader_thread()
        self._start_main_thread()
        self.watchdog_thread = threading.Thread(
            target=self._runtime_watchdog, daemon=True, name="agent-watchdog"
        )
        self.watchdog_thread.start()
        self._start_motion_init_thread()
        # AI warmup is intentionally delayed/background-only. Voice input is
        # available regardless of Ollama/Thor state.
        self._start_background_warmup()

    def _start_wake_loader_thread(self):
        if self.exiting:
            return
        with self._thread_lock:
            if self.wake_loader_thread is not None and self.wake_loader_thread.is_alive():
                return
            self.wake_loader_thread = threading.Thread(
                target=self._wake_loader_worker, daemon=True, name="wake-model-loader"
            )
            self.wake_loader_thread.start()

    def _wake_loader_worker(self):
        print("[INIT] Loading wake words in background...", flush=True)
        wake_cfg = CURRENT_CONFIG.get("wake_word", {})
        try:
            if not wake_cfg.get("enabled", True):
                print("[WAKE] Wake-word detection disabled; Enter remains available.", flush=True)
                return
            wake_models: List[str] = []
            phrases: List[str] = []
            labels: Dict[str, str] = {}
            for entry in wake_cfg.get("models", []):
                if not isinstance(entry, dict):
                    continue
                phrase = str(entry.get("phrase", "Wake word")).strip()
                path = str(entry.get("path", "")).strip()
                model_name = str(entry.get("model", "")).strip()
                if path:
                    if os.path.exists(path):
                        wake_models.append(path)
                        phrases.append(phrase)
                        labels[Path(path).stem] = phrase
                    else:
                        print(f"[WAKE] Optional model missing for {phrase}: {path}", flush=True)
                elif model_name:
                    model_meta = getattr(openwakeword, "MODELS", {}).get(model_name, {})
                    model_path = str(model_meta.get("model_path", ""))
                    if model_path.endswith(".tflite"):
                        model_path = model_path[:-7] + ".onnx"
                    if model_path and os.path.exists(model_path):
                        wake_models.append(model_path)
                        phrases.append(phrase)
                        labels[Path(model_path).stem] = phrase
                    else:
                        print(f"[WAKE] Built-in model missing for {phrase} ({model_name}).", flush=True)
            if not wake_models:
                legacy_model = str(wake_cfg.get("legacy_model", "wakeword.onnx")).strip()
                if legacy_model and os.path.exists(legacy_model):
                    wake_models = [legacy_model]
                    phrases = ["Hey Jarvis"]
                    labels[Path(legacy_model).stem] = "Hey Jarvis"
            if not wake_models:
                print("[WAKE] No wake-word model available; Enter remains available.", flush=True)
                return
            try:
                model = Model(wakeword_models=wake_models, inference_framework="onnx")
            except TypeError:
                model = Model(wakeword_model_paths=wake_models)
            # Publish the labels before the model: agent-main waits on
            # oww_model, so assigning it last keeps the lookup tables consistent.
            self.wake_word_phrases = phrases
            self.wake_word_labels = labels
            self.oww_model = model
            print("[INIT] Wake words loaded: " + ", ".join(phrases), flush=True)
        except Exception as exc:
            print(f"[WAKE] Could not load wake-word models: {exc}; Enter remains available.", flush=True)
        finally:
            self.wake_model_ready.set()

    def _start_main_thread(self):
        if self.exiting:
            return
        with self._thread_lock:
            if self.main_thread is not None and self.main_thread.is_alive():
                return
            self.main_thread = threading.Thread(target=self.safe_main_execution, daemon=True, name="agent-main")
            self.main_thread.start()

    def _start_motion_init_thread(self):
        """Initialize/retry the MPU-6050 without ever blocking voice startup."""
        if self.exiting:
            return
        with self._thread_lock:
            if self.motion_init_thread is not None and self.motion_init_thread.is_alive():
                return
            self.motion_init_thread = threading.Thread(
                target=self._motion_init_worker,
                daemon=True,
                name="mpu6050-init",
            )
            self.motion_init_thread.start()

    def _motion_init_worker(self):
        retry_seconds = max(2.0, float(
            CURRENT_CONFIG.get("hardware", {}).get("mpu6050", {}).get("retry_seconds", 10.0)
        ))
        attempt = 0
        while not self.exiting:
            attempt += 1
            try:
                print(f"[MPU6050] Async initialization attempt {attempt}...", flush=True)
                if self.motion.start():
                    print("[MPU6050] Sensor online; motion/orientation context enabled.", flush=True)
                    return
                err = self.motion.snapshot().get("last_error", "unknown error")
                print(f"[MPU6050] Not ready: {err}. Retrying in {retry_seconds:.0f}s.", flush=True)
            except Exception as exc:
                print(f"[MPU6050] Async initialization failed: {exc}. Retrying in {retry_seconds:.0f}s.", flush=True)

            # Event-style sleep so shutdown does not have to wait the whole retry interval.
            deadline = time.monotonic() + retry_seconds
            while not self.exiting and time.monotonic() < deadline:
                time.sleep(0.25)

    def _runtime_watchdog(self):
        """Keep worker threads alive and recover from a dead main loop.

        Network/audio operations have their own finite timeouts.  This watchdog
        handles the other important failure mode: an unexpected exception that
        kills a worker while leaving the Tk GUI running.
        """
        runtime_cfg = CURRENT_CONFIG.get("runtime", {})
        interval = max(1.0, float(runtime_cfg.get("watchdog_interval_seconds", 2.0)))
        # A wedged-but-alive worker is the failure the progress timestamps were
        # collected for; until now nothing ever read them, so a stall was never
        # actually detected. Keep this comfortably above the longest legitimate
        # single step (STT/AI/TTS timeouts) to avoid interrupting real work.
        stall_limit = max(60.0, float(runtime_cfg.get("stall_timeout_seconds", 180.0)))
        while not self.exiting:
            try:
                self._ensure_tts_worker()
                if self.main_thread is not None and not self.main_thread.is_alive():
                    print("[WATCHDOG] agent-main stopped unexpectedly; restarting it.", flush=True)
                    self._start_main_thread()
                elif time.monotonic() - self.last_progress_time > stall_limit:
                    print(
                        f"[WATCHDOG] No progress for {stall_limit:.0f}s; forcing recovery.",
                        flush=True,
                    )
                    # Reset the clock first so a slow recovery is not treated as
                    # a fresh stall on the very next pass.
                    self._touch_progress("watchdog recovery")
                    self._interrupt_current_work(queue_ptt=False)
                    self.set_state(BotStates.IDLE, "Recovered from a stall - ready.")
            except Exception as exc:
                print(f"[WATCHDOG] recovery check failed: {exc}", flush=True)
            time.sleep(interval)

    def _interrupt_current_work(self, *, queue_ptt: bool = False):
        """Best-effort cancellation used by Enter and error recovery."""
        self.interrupted.set()
        self.thinking_sound_active.clear()
        self.recording_active.clear()
        with self.tts_queue_lock:
            self.tts_queue.clear()
        process = self.current_audio_process
        if process:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass
        try:
            self.ai.cancel_pending_requests()
        except Exception:
            pass
        if queue_ptt:
            self.ptt_event.set()
        self._touch_progress("interrupt requested")

    def _reset_after_cycle_error(self):
        self.thinking_sound_active.clear()
        self.recording_active.clear()
        self.interrupted.clear()
        try:
            sd.stop()
        except Exception:
            pass
        self._ensure_tts_worker()

    def safe_main_execution(self):
        # Voice input is the primary control path, so it must never wait for an
        # AI/network warmup.  Start warmup in the background and enter the wake
        # loop immediately.  The first request may be slower if the model is not
        # loaded yet, but Hey BMO and Enter remain responsive.
        if not self.warmed_up:
            self.warmed_up = True
            self.set_state(BotStates.IDLE, "Ready - listening for wake or Enter.")
            # _start_runtime_workers already kicked warmup off; _start_background_warmup
            # is idempotent now, so this no longer spawns a second warmup thread.

        self._ensure_tts_worker()

        while not self.exiting:
            try:
                # Clear a previous cancellation before beginning the next cycle.
                # A queued Enter/PTT event is deliberately preserved.
                self.interrupted.clear()
                self._touch_progress("waiting for wake")
                trigger_source = self.detect_wake_word_or_ptt()
                if self.exiting:
                    break

                self._touch_progress(f"trigger={trigger_source}")
                self.set_state(BotStates.LISTENING, "I'm listening!")
                keyboard_mode = str(CURRENT_CONFIG.get("runtime", {}).get("keyboard_trigger_mode", "vad")).lower()
                if trigger_source == "PTT" and keyboard_mode in {"toggle", "hold", "ptt"}:
                    audio_file = self.record_voice_ptt()
                else:
                    # Wake word and the default one-press Enter trigger both use
                    # silence-ending adaptive VAD.
                    audio_file = self.record_voice_adaptive()

                if self.interrupted.is_set():
                    self._reset_after_cycle_error()
                    self.set_state(BotStates.IDLE, "Interrupted; ready again.")
                    continue
                if not audio_file:
                    self.set_state(BotStates.IDLE, "Heard nothing.")
                    continue

                self._touch_progress("transcribing")
                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue

                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self._touch_progress("AI request")
                self.chat_and_respond(user_text)
                self._touch_progress("cycle complete")

            except Exception as exc:
                # This used to sit outside the whole while-loop. One transient
                # device/network exception therefore killed agent-main forever.
                # Keep the process alive and return to wake/PTT instead.
                traceback.print_exc()
                print(f"[RECOVERY] Interaction failed: {exc}", flush=True)
                self._reset_after_cycle_error()
                self.set_state(BotStates.ERROR, f"Recovered from error: {str(exc)[:60]}")
                time.sleep(0.35)
                self.set_state(BotStates.IDLE, "Recovered - waiting for wake or Enter.")

    def _start_background_warmup(self):
        if self.exiting:
            return
        with self._thread_lock:
            if self.warmup_thread is not None and self.warmup_thread.is_alive():
                return
            self.warmup_thread = threading.Thread(
                target=self._background_warmup_worker,
                daemon=True,
                name="ai-warmup",
            )
            self.warmup_thread.start()

    def _background_warmup_worker(self):
        try:
            print("[AI] Background warmup started; voice input remains active.", flush=True)
            self.ai.warmup()
            print(f"[AI] Background warmup complete: {self.ai.status_text()}", flush=True)
        except Exception as exc:
            print(f"[AI] Background warmup warning: {exc}", flush=True)

    def warm_up_logic(self):
        """Compatibility wrapper; warmup is intentionally non-blocking now."""
        self._start_background_warmup()

    # ------------------------------------------------------------------
    # Wake word and recording
    # ------------------------------------------------------------------
    def detect_wake_word_or_ptt(self) -> str:
        self.set_state(BotStates.IDLE, "Waiting...")
        # Never clear here: Enter may have been pressed while the previous
        # request was finishing. Preserve that queued manual wake.
        if self.ptt_event.is_set():
            self.ptt_event.clear()
            return "PTT"
        # Wake models load asynchronously so the GUI never blocks. Enter must
        # work immediately even while ONNX initialization is still happening.
        while self.oww_model is None and not self.exiting:
            if self.ptt_event.wait(timeout=0.10):
                self.ptt_event.clear()
                return "PTT"
            if self.wake_model_ready.is_set() and self.oww_model is None:
                # Loading failed/disabled. Continue polling for Enter without
                # blocking forever in Event.wait(), which also aids shutdown.
                continue
        if self.exiting:
            return "PTT"
        if self.oww_model:
            self.oww_model.reset()

        target_size = 1280
        target_rate = 16000
        input_rate = choose_input_samplerate(INPUT_DEVICE, CURRENT_CONFIG.get("input_sample_rate"))
        input_size = max(1, int(round(target_size * input_rate / target_rate)))

        try:
            with sd.InputStream(
                samplerate=input_rate,
                channels=1,
                dtype="int16",
                blocksize=input_size,
                device=INPUT_DEVICE,
                latency="high",
            ) as stream:
                print(f"[AUDIO] Wake listening at {input_rate} Hz", flush=True)
                while not self.exiting:
                    if self.ptt_event.is_set():
                        self.ptt_event.clear()
                        return "PTT"

                    # Keep the original CLI-enter fallback, but only when stdin is
                    # a real terminal. A closed/redirected stdin is always
                    # "readable" at EOF, which previously fired a phantom Enter
                    # on every pass and made the agent loop on itself forever.
                    if self._stdin_usable:
                        try:
                            rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                            if rlist:
                                line = sys.stdin.readline()
                                if line == "":
                                    # EOF: stdin will never block again. Stop
                                    # polling it instead of spinning on it.
                                    print("[INPUT] stdin reached EOF; disabling the terminal Enter fallback.", flush=True)
                                    self._stdin_usable = False
                                else:
                                    return "PTT"
                        except Exception:
                            self._stdin_usable = False

                    data, overflow = stream.read(input_size)
                    if overflow:
                        print("[AUDIO] Wake input overflow", flush=True)
                    audio = np.asarray(data, dtype=np.int16).reshape(-1)
                    if input_rate != target_rate:
                        gcd = np.gcd(input_rate, target_rate)
                        audio = scipy.signal.resample_poly(audio, target_rate // gcd, input_rate // gcd)
                        audio = np.asarray(audio[:target_size], dtype=np.int16)
                    if len(audio) < target_size:
                        audio = np.pad(audio, (0, target_size - len(audio)))
                    elif len(audio) > target_size:
                        audio = audio[:target_size]

                    # openWakeWord is a streaming model: it keeps internal audio
                    # feature state across calls. Skipping quiet frames breaks
                    # that continuity and wrecks detection accuracy, so every
                    # frame is fed. The amplitude check only skips the cheap
                    # score scan below.
                    self.oww_model.predict(audio)
                    current_max = int(np.max(np.abs(audio))) if len(audio) else 0
                    if current_max <= 200:
                        continue
                    for model_name, scores in self.oww_model.prediction_buffer.items():
                        if not scores:
                            continue
                        score = float(list(scores)[-1])
                        if score > self.wake_word_threshold:
                            phrase = self.wake_word_labels.get(model_name, model_name)
                            print(f"[WAKE] {phrase}: {score:.2f}", flush=True)
                            self.oww_model.reset()
                            return "WAKE"
        except Exception as exc:
            print(f"[WAKE] Stream failed: {exc}; falling back to PTT.", flush=True)
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"
        return "WAKE"

    def record_voice_adaptive(self, filename: str = "input.wav") -> Optional[str]:
        """Record one wake-word utterance and stop after real post-speech silence.

        USB microphones often have a noise floor above the old fixed 0.006 RMS
        threshold.  This recorder measures the room/mic noise floor for a short
        window, derives start/end thresholds from it, waits for speech to begin,
        and only then counts trailing silence.
        """
        print("[AUDIO] Recording adaptive VAD...", flush=True)
        samplerate = choose_input_samplerate(INPUT_DEVICE, CURRENT_CONFIG.get("input_sample_rate"))

        cfg = CURRENT_CONFIG.get("recording", {})
        legacy_min = float(CURRENT_CONFIG.get("silence_threshold", 0.006))
        start_timeout = max(1.0, float(cfg.get("start_timeout_seconds", 6.0)))
        end_silence = max(0.35, float(cfg.get("end_silence_seconds", 1.0)))
        max_record_time = max(end_silence + 1.0, float(cfg.get("max_record_seconds", 20.0)))
        min_speech_time = max(0.10, float(cfg.get("min_speech_seconds", 0.25)))
        calibration_time = max(0.10, float(cfg.get("noise_calibration_seconds", 0.35)))
        pre_roll_time = max(0.0, float(cfg.get("pre_roll_seconds", 0.25)))
        chunk_duration = min(0.10, max(0.02, float(cfg.get("chunk_seconds", 0.05))))
        min_rms = max(0.0005, float(cfg.get("min_rms", legacy_min)))
        noise_multiplier = max(1.1, float(cfg.get("noise_multiplier", 2.4)))
        end_noise_multiplier = max(1.05, float(cfg.get("end_noise_multiplier", 1.7)))
        speech_start_chunks = max(1, int(cfg.get("speech_start_chunks", 2)))

        chunk_size = max(1, int(samplerate * chunk_duration))
        calibration_chunks = max(2, int(calibration_time / chunk_duration))
        required_silent = max(1, int(end_silence / chunk_duration))
        min_speech_chunks = max(1, int(min_speech_time / chunk_duration))
        max_chunks = max(1, int(max_record_time / chunk_duration))
        start_timeout_chunks = max(1, int(start_timeout / chunk_duration))
        pre_roll_limit = max(1, int(pre_roll_time / chunk_duration))

        buffer: List[np.ndarray] = []
        pre_roll: List[np.ndarray] = []
        noise_samples: List[float] = []
        total_chunks = 0
        speech_chunks = 0
        speech_run = 0
        silent_chunks = 0
        speech_started = False
        stop_reason = ""
        stop_event = threading.Event()
        noise_floor = min_rms / noise_multiplier
        start_threshold = min_rms
        end_threshold = min_rms * 0.8

        def rms_level(chunk: np.ndarray) -> float:
            values = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if values.size == 0:
                return 0.0
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.sqrt(np.mean(values * values)))

        def callback(indata, frames, time_info, status):
            nonlocal total_chunks, speech_chunks, speech_run, silent_chunks
            nonlocal speech_started, stop_reason, noise_floor, start_threshold, end_threshold

            if self.interrupted.is_set():
                stop_reason = "interrupted"
                stop_event.set()
                return

            chunk = indata.copy()
            level = rms_level(chunk)
            total_chunks += 1

            if not speech_started:
                pre_roll.append(chunk)
                if len(pre_roll) > pre_roll_limit:
                    del pre_roll[0]

                # Learn the lower-energy room/microphone floor. Median is much
                # less sensitive to a cough, click, or the tail of the wake word.
                if len(noise_samples) < calibration_chunks:
                    noise_samples.append(level)
                    sorted_noise = sorted(noise_samples)
                    lower_half = sorted_noise[: max(1, (len(sorted_noise) + 1) // 2)]
                    noise_floor = float(np.median(lower_half))
                    start_threshold = max(min_rms, noise_floor * noise_multiplier)
                    end_threshold = max(min_rms, noise_floor * end_noise_multiplier)

                # Do not declare speech until the short noise calibration has
                # completed. Requiring consecutive chunks rejects clicks/noise.
                if total_chunks >= calibration_chunks:
                    if level >= start_threshold:
                        speech_run += 1
                    else:
                        speech_run = 0

                    if speech_run >= speech_start_chunks:
                        speech_started = True
                        speech_chunks = speech_run
                        buffer.extend(pre_roll)
                        pre_roll.clear()
                        print(
                            f"[AUDIO] Speech started: noise={noise_floor:.4f}, "
                            f"start={start_threshold:.4f}, end={end_threshold:.4f}",
                            flush=True,
                        )
                        return

                if total_chunks >= start_timeout_chunks:
                    stop_reason = "no speech detected"
                    stop_event.set()
                return

            buffer.append(chunk)
            speech_chunks += 1

            if level < end_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if speech_chunks >= min_speech_chunks and silent_chunks >= required_silent:
                stop_reason = f"{end_silence:.1f}s trailing silence"
                stop_event.set()
                return

            if total_chunks >= max_chunks:
                stop_reason = f"{max_record_time:.1f}s safety limit"
                stop_event.set()

        try:
            sd.stop()
            time.sleep(0.05)
            with sd.InputStream(
                samplerate=samplerate,
                channels=1,
                callback=callback,
                device=INPUT_DEVICE,
                blocksize=chunk_size,
                dtype="float32",
            ):
                # The chunk limits above are enforced inside the callback, so a
                # stream that stops delivering callbacks would otherwise wedge
                # agent-main here forever with the GUI stuck on "I'm listening".
                # This wall-clock deadline bounds the wait no matter what the
                # audio device does.
                hard_deadline = time.monotonic() + max_record_time + calibration_time + 5.0
                while not stop_event.is_set() and not self.exiting and not self.interrupted.is_set():
                    if time.monotonic() >= hard_deadline:
                        print("[AUDIO] Input stream stalled; abandoning this recording.", flush=True)
                        stop_reason = "input stream stalled"
                        break
                    sd.sleep(25)
        except Exception as exc:
            print(f"[AUDIO] Adaptive recording failed: {exc}", flush=True)
            return None

        if self.interrupted.is_set():
            print("[AUDIO] Recording interrupted.", flush=True)
            return None

        if not speech_started:
            print(
                f"[AUDIO] Stopped: {stop_reason or 'no speech'} "
                f"(noise={noise_floor:.4f}, start={start_threshold:.4f})",
                flush=True,
            )
            return None

        print(f"[AUDIO] Stopped: {stop_reason or 'finished'}", flush=True)

        # Do not send the counted trailing silence to Whisper.
        if silent_chunks > 0 and len(buffer) > silent_chunks:
            del buffer[-silent_chunks:]

        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename: str = "input.wav") -> Optional[str]:
        print("[AUDIO] Recording PTT...", flush=True)
        time.sleep(0.2)
        samplerate = choose_input_samplerate(INPUT_DEVICE, CURRENT_CONFIG.get("input_sample_rate"))
        buffer: List[np.ndarray] = []

        def callback(indata, frames, time_info, status):
            buffer.append(indata.copy())

        try:
            sd.stop()
            time.sleep(0.1)
            max_ptt_seconds = max(5.0, float(CURRENT_CONFIG.get("recording", {}).get("max_record_seconds", 20.0)))
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE):
                # A held key or a wedged stream must not hold agent-main forever.
                hard_deadline = time.monotonic() + max_ptt_seconds
                while self.recording_active.is_set() and not self.exiting and not self.interrupted.is_set():
                    if time.monotonic() >= hard_deadline:
                        print(f"[AUDIO] PTT hit the {max_ptt_seconds:.0f}s safety limit.", flush=True)
                        self.recording_active.clear()
                        break
                    sd.sleep(50)
        except Exception as exc:
            print(f"[AUDIO] PTT recording failed: {exc}", flush=True)
            return None
        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer: List[np.ndarray], filename: str, samplerate: int) -> Optional[str]:
        if not buffer:
            return None
        audio = np.concatenate(buffer, axis=0).reshape(-1)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if np.issubdtype(audio.dtype, np.floating):
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)
        with wave.open(filename, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(samplerate)
            handle.writeframes(audio.tobytes())
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename: str) -> str:
        print("[STT] Transcribing...", flush=True)
        executable = "./whisper.cpp/build/bin/whisper-cli"
        model = "./whisper.cpp/models/ggml-base.en.bin"
        if not os.path.exists(executable) or not os.path.exists(model):
            print("[STT] whisper.cpp is not installed. Run ./setup.sh.", flush=True)
            return ""
        try:
            timeout = max(10.0, float(CURRENT_CONFIG.get("runtime", {}).get("stt_timeout_seconds", 60.0)))
            proc = subprocess.Popen(
                [executable, "-m", model, "-l", "en", "-t", "4", "-f", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if self.interrupted.is_set() or self.exiting:
                    print("[STT] Transcription interrupted.", flush=True)
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return ""
                if time.monotonic() >= deadline:
                    print(f"[STT] Timed out after {timeout:.1f}s.", flush=True)
                    proc.kill()
                    proc.wait(timeout=2.0)
                    return ""
                time.sleep(0.05)
            stdout, _stderr = proc.communicate()
            segments = []
            for line in stdout.splitlines():
                if "]" in line and "[" in line:
                    candidate = line.split("]", 1)[1].strip()
                    if candidate and candidate not in {"[BLANK_AUDIO]", "[ Silence ]"}:
                        segments.append(candidate)
            transcription = " ".join(segments).strip()
            print(f"[STT] Heard: {transcription!r}", flush=True)
            return transcription
        except Exception as exc:
            print(f"[STT] Error: {exc}", flush=True)
            return ""

    # ------------------------------------------------------------------
    # Tool execution and AI
    # ------------------------------------------------------------------
    def execute_tool(self, name: str, arguments: Dict[str, Any], user_text: str) -> str:
        print(f"[TOOL] {name} {arguments}", flush=True)
        if name == "get_time":
            now = datetime.datetime.now().astimezone()
            return now.strftime("It is %A, %B %d, %Y at %I:%M %p %Z.")

        if name == "get_motion_state":
            return self.motion.context_for_llm()

        if name == "get_orientation":
            return self.motion.orientation_context_for_llm()

        if name == "set_rest_orientation":
            result = self.motion.calibrate_rest()
            if result.get("ok"):
                return (
                    "Rest orientation recalibrated successfully. "
                    f"Baseline roll={result.get('rest_roll_deg')} degrees and "
                    f"pitch={result.get('rest_pitch_deg')} degrees. "
                    "Future roll/pitch/tilt readings are relative to this pose."
                )
            return f"Could not calibrate the MPU-6050 rest orientation: {result.get('error', 'unknown error')}"

        if name == "search_web":
            query = str(arguments.get("query") or user_text).strip()
            if not query:
                return "No search query was provided."
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, region="us-en", max_results=3))
                if not results:
                    return f"No web results were found for {query!r}."
                snippets = []
                for item in results[:3]:
                    title = item.get("title", "Untitled")
                    body = item.get("body", item.get("snippet", ""))
                    href = item.get("href", item.get("url", ""))
                    snippets.append(f"{title}: {body[:350]} ({href})")
                return "\n".join(snippets)
            except Exception as exc:
                return f"Web search failed: {exc}"

        if name == "look_at_camera":
            prompt = str(arguments.get("prompt") or user_text or "Describe what you see.")
            self.set_state(BotStates.CAPTURING, "Looking through the camera...")
            try:
                scene = self.vision.capture_and_perceive()
                self.last_camera_path = scene["image_path"]
                self.set_state(BotStates.THINKING, "Analyzing the camera...", cam_path=self.last_camera_path)
                description = self.ai.vision_describe(
                    prompt,
                    self.last_camera_path,
                    perception_context=scene.get("summary", ""),
                )
                return (
                    f"Camera/VLM description: {description}\n"
                    f"Hailo perception: {scene.get('summary', '')}"
                )
            except Exception as exc:
                return f"Camera analysis failed: {exc}"

        return f"Unknown tool: {name}"

    def chat_and_respond(self, text: str):
        # Do not keep showing an image from a previous camera request.
        self.last_camera_path = None
        lower = text.lower()
        if "forget everything" in lower or "reset memory" in lower:
            self.history = []
            self.session_memory = []
            self.save_chat_history()
            self.respond_plain("Okay. Conversation memory cleared.")
            return

        # Physical calibration commands should not depend on the LLM choosing a tool.
        # The user must keep the robot stationary while this short calibration runs.
        if re.search(r"\b(?:set|make|calibrate)\b.*\b(?:rest|zero|home)\b", lower) or re.search(
            r"\b(?:set|make) (?:this|current) (?:position|pose|orientation) (?:as|the) (?:rest|zero)\b", lower
        ):
            result = self.motion.calibrate_rest()
            if result.get("ok"):
                self.respond_plain(
                    "Rest position calibrated. Roll and pitch are now zeroed to this pose."
                )
            else:
                self.respond_plain(f"I could not calibrate the accelerometer: {result.get('error', 'unknown error')}")
            return

        # Speech-level routing is handled before asking the LLM so even a small
        # local model cannot accidentally ignore a backend switch command.
        switch_reply = self.ai.parse_speech_command(text)
        if switch_reply:
            self.respond_plain(switch_reply)
            return

        sensor_context = self.motion.context_for_llm()
        backend_context = self.ai.status_text()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": f"LIVE ROBOT STATE: {sensor_context}\nAI ROUTING STATE: {backend_context}",
            },
        ]
        # Angle/orientation questions always receive a fresh dedicated sensor read,
        # even if the model chooses not to emit a function call.
        if re.search(r"\b(angle|tilt|orientation|level|lean|roll|pitch|accelerometer|gyro|gyroscope|moved|moving)\b", lower):
            messages.append({
                "role": "system",
                "content": "FRESH MPU-6050 READ: " + self.motion.orientation_context_for_llm(),
            })
        # Keep the local prompt small. Re-evaluating a long conversation on the
        # Pi CPU adds substantial first-token latency. Thor retains a larger window.
        memory_cfg = CURRENT_CONFIG.get("memory", {})
        if self.ai.current_backend == "local":
            max_context_messages = int(memory_cfg.get("local_context_messages", 6))
        else:
            max_context_messages = int(memory_cfg.get("thor_context_messages", 16))
        conversation_memory = self.history + self.session_memory
        if max_context_messages > 0:
            conversation_memory = conversation_memory[-max_context_messages:]
        else:
            conversation_memory = []
        messages.extend(conversation_memory)
        messages.append({"role": "user", "content": text})
        self.session_memory.append({"role": "user", "content": text})

        self.set_state(BotStates.THINKING, f"Thinking on {self.ai.current_backend}...")
        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        final_text = ""
        fallback_notice = ""
        try:
            # try/finally below guarantees the thinking-sound flag is cleared;
            # the interrupted early-returns in this loop used to leave it set.
            for _ in range(5):
                if self.interrupted.is_set():
                    return
                response = self.ai.chat(messages, tools=TOOL_SCHEMAS)
                actual_backend = response.get("actual_backend", self.ai.current_backend)
                print(
                    f"[AI ROUTE] requested={response.get('requested_backend', actual_backend)} "
                    f"actual={actual_backend} model={response.get('model', self.ai.get_model())}",
                    flush=True,
                )
                if response.get("fallback_used"):
                    fallback_from = response.get("fallback_from", "remote backend")
                    fallback_notice = f"{fallback_from} became unavailable, so I switched back to local. "
                if self.interrupted.is_set():
                    print("[AI] Response discarded because the request was interrupted.", flush=True)
                    return
                metrics = response.get("metrics") or {}
                if metrics:
                    total_ns = metrics.get("total_duration")
                    eval_count = metrics.get("eval_count")
                    eval_ns = metrics.get("eval_duration")
                    total_s = (float(total_ns) / 1e9) if total_ns else None
                    tok_s = (float(eval_count) / (float(eval_ns) / 1e9)) if eval_count and eval_ns else None
                    parts = []
                    if total_s is not None:
                        parts.append(f"total={total_s:.2f}s")
                    if tok_s is not None:
                        parts.append(f"generation={tok_s:.1f} tok/s")
                    if metrics.get("prompt_eval_count") is not None:
                        parts.append(f"prompt={metrics.get('prompt_eval_count')} tok")
                    if parts:
                        print("[AI PERF] " + ", ".join(parts), flush=True)
                tool_calls = response.get("tool_calls") or []
                content = str(response.get("content", "")).strip()
                if not tool_calls:
                    final_text = content
                    break

                assistant_tool_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_tool_message)

                for call in tool_calls:
                    function = call.get("function", {})
                    name = str(function.get("name", ""))
                    arguments = function.get("arguments") or {}
                    if not isinstance(arguments, dict):
                        arguments = {}
                    result = self.execute_tool(name, arguments, text)
                    tool_message: Dict[str, Any] = {
                        "role": "tool",
                        "tool_name": name,
                        "content": result,
                    }
                    if call.get("id"):
                        tool_message["tool_call_id"] = call.get("id")
                    messages.append(tool_message)
            if not final_text:
                final_text = "I could not complete that request with the available tools."
            if fallback_notice and not final_text.startswith(fallback_notice):
                final_text = fallback_notice + final_text

            self.thinking_sound_active.clear()
            self.session_memory.append({"role": "assistant", "content": final_text})
            self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=self.last_camera_path)
            self.append_to_text(f"BOT: {final_text}")
            self.queue_tts(final_text)
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, f"Ready - {self.ai.current_backend}")
        except AIBackendError as exc:
            self.thinking_sound_active.clear()
            print(f"[AI] Error: {exc}", flush=True)
            self.respond_plain(f"My AI backend is unavailable: {str(exc)[:160]}")
        except Exception as exc:
            self.thinking_sound_active.clear()
            traceback.print_exc()
            self.respond_plain(f"I hit an internal error: {str(exc)[:120]}")
        finally:
            self.thinking_sound_active.clear()

    def respond_plain(self, text: str):
        self.thinking_sound_active.clear()
        self.set_state(BotStates.SPEAKING, "Speaking...", cam_path=self.last_camera_path)
        self.append_to_text(f"BOT: {text}")
        self.queue_tts(text)
        self.wait_for_tts()
        self.set_state(BotStates.IDLE, f"Ready - {self.ai.current_backend}")

    # ------------------------------------------------------------------
    # TTS / audio output
    # ------------------------------------------------------------------
    def queue_tts(self, text: str):
        clean = text.strip()
        if not clean:
            return
        with self.tts_queue_lock:
            self.tts_queue.append(clean)

    def wait_for_tts(self):
        timeout = max(5.0, float(CURRENT_CONFIG.get("runtime", {}).get("tts_timeout_seconds", 35.0)))
        deadline = time.monotonic() + timeout
        while not self.exiting:
            with self.tts_queue_lock:
                queue_empty = not self.tts_queue
            if queue_empty and not self.tts_active.is_set():
                return
            if self.interrupted.is_set():
                return
            if time.monotonic() >= deadline:
                print(f"[TTS] Timed out after {timeout:.1f}s; recovering audio worker.", flush=True)
                with self.tts_queue_lock:
                    self.tts_queue.clear()
                if self.current_audio_process:
                    try:
                        self.current_audio_process.terminate()
                    except Exception:
                        pass
                try:
                    sd.stop()
                except Exception:
                    pass
                return
            time.sleep(0.05)

    def _tts_worker(self):
        while not self.exiting:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
                    # Mark the worker busy while still holding the lock. If this
                    # happened after the lock was released, wait_for_tts could
                    # observe an empty queue and an inactive worker in the gap,
                    # return early, and let the next cycle's sd.stop() cut off
                    # the reply the robot had only just started speaking.
                    self.tts_active.set()
            if text:
                try:
                    self.speak(text)
                finally:
                    self.tts_active.clear()
            else:
                time.sleep(0.05)

    def speak(self, text: str):
        clean = re.sub(r"[^\w\s,.!?:;'\-]", "", text)
        if not clean.strip():
            return
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")
        if not os.path.exists("./piper/piper") or not os.path.exists(voice_model):
            print("[TTS] Piper or voice model missing.", flush=True)
            return
        source_rate = piper_sample_rate(voice_model)
        print(f"[TTS] {clean!r}", flush=True)

        kill_timer = None
        try:
            self.current_audio_process = subprocess.Popen(
                ["./piper/piper", "--model", voice_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            active_process = self.current_audio_process
            piper_timeout = max(5.0, float(CURRENT_CONFIG.get("runtime", {}).get("piper_process_timeout_seconds", 30.0)))

            def _kill_stuck_piper():
                if active_process.poll() is None:
                    print(f"[TTS] Piper exceeded {piper_timeout:.1f}s; terminating it.", flush=True)
                    try:
                        active_process.terminate()
                    except Exception:
                        pass

            kill_timer = threading.Timer(piper_timeout, _kill_stuck_piper)
            kill_timer.daemon = True
            kill_timer.start()

            self.current_audio_process.stdin.write(clean.encode("utf-8") + b"\n")
            self.current_audio_process.stdin.close()

            try:
                info = sd.query_devices(OUTPUT_DEVICE, "output")
                native_rate = int(info["default_samplerate"])
            except Exception:
                native_rate = 48000

            playback_rate = source_rate
            resample = False
            try:
                sd.check_output_settings(
                    device=OUTPUT_DEVICE,
                    samplerate=source_rate,
                    channels=1,
                    dtype="int16",
                )
            except Exception:
                playback_rate = native_rate
                resample = True

            with sd.RawOutputStream(
                samplerate=playback_rate,
                channels=1,
                dtype="int16",
                device=OUTPUT_DEVICE,
                latency="low",
                blocksize=2048,
            ) as stream:
                while not self.exiting and not self.interrupted.is_set():
                    data = self.current_audio_process.stdout.read(4096)
                    if not data:
                        break
                    chunk = np.frombuffer(data, dtype=np.int16)
                    if not len(chunk):
                        continue
                    self.current_volume = int(np.max(np.abs(chunk)))
                    if resample:
                        target_samples = max(1, int(round(len(chunk) * playback_rate / source_rate)))
                        chunk = scipy.signal.resample(chunk, target_samples).astype(np.int16)
                    stream.write(chunk.tobytes())
        except Exception as exc:
            print(f"[TTS] Audio error: {exc}", flush=True)
        finally:
            if kill_timer is not None:
                kill_timer.cancel()
            self.current_volume = 0
            if self.current_audio_process:
                try:
                    if self.current_audio_process.stdout:
                        self.current_audio_process.stdout.close()
                    if self.current_audio_process.poll() is None:
                        self.current_audio_process.terminate()
                except Exception:
                    pass
                self.current_audio_process = None

    def _run_thinking_sound_loop(self):
        """Play at most one thinking acknowledgement for each user request."""
        cfg = CURRENT_CONFIG.get("thinking_audio", {})
        if not cfg.get("enabled", True):
            return
        delay = max(0.0, float(cfg.get("delay_seconds", 0.25)))
        time.sleep(delay)
        if not self.thinking_sound_active.is_set() or self.exiting:
            return
        sound = self.get_random_sound(thinking_sounds_dir)
        if sound:
            self.play_sound(sound)
        # Deliberately do not loop. The event remains useful for cancellation,
        # but a slow backend no longer repeats "thinking" announcements.

    @staticmethod
    def get_random_sound(directory: str) -> Optional[str]:
        if not os.path.exists(directory):
            return None
        files = [f for f in os.listdir(directory) if f.lower().endswith(".wav")]
        return os.path.join(directory, random.choice(files)) if files else None

    def play_sound(self, file_path: Optional[str]):
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with wave.open(file_path, "rb") as handle:
                file_rate = handle.getframerate()
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                if width != 2:
                    return
                audio = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
            try:
                sd.check_output_settings(device=OUTPUT_DEVICE, samplerate=file_rate, channels=1, dtype="int16")
                rate = file_rate
            except Exception:
                try:
                    rate = int(sd.query_devices(OUTPUT_DEVICE, "output")["default_samplerate"])
                except Exception:
                    rate = 48000
                audio = scipy.signal.resample(audio, int(len(audio) * rate / file_rate)).astype(np.int16)
            sd.play(audio, rate, device=OUTPUT_DEVICE)
            sd.wait()
        except Exception as exc:
            print(f"[AUDIO] Sound effect failed: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------
    @staticmethod
    def load_chat_history() -> List[Dict[str, str]]:
        if not os.path.exists(MEMORY_FILE):
            return []
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as handle:
                messages = json.load(handle)
            if not isinstance(messages, list):
                return []
            # Drop stale system prompts from older versions of the project.
            return [m for m in messages if isinstance(m, dict) and m.get("role") in {"user", "assistant"}]
        except Exception:
            return []

    def save_chat_history(self):
        if not CURRENT_CONFIG.get("chat_memory", True):
            return
        saved_messages = max(0, int(CURRENT_CONFIG.get("memory", {}).get("saved_messages", 16)))
        combined = (self.history + self.session_memory)[-saved_messages:] if saved_messages else []
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as handle:
                json.dump(combined, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[MEMORY] Save failed: {exc}", flush=True)


if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    # atexit was imported but never used, so a kill/crash lost the conversation.
    atexit.register(app.save_chat_history)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.safe_exit()
