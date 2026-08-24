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
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

DEFAULT_CONFIG: Dict[str, Any] = {
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": True,
    "system_prompt_extras": "",
    "input_device": "USB",
    "input_sample_rate": None,
    "output_device": "MAX98357A",
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
                "keep_alive": "-1",
            },
            "thor": {
                "type": "ollama",
                "base_url": "http://jetson-thor.local:11434",
                "text_model": "qwen3.5:27b",
                "vision_model": "qwen3.5:27b",
                "text_models": ["qwen3.5:9b", "qwen3.5:27b", "qwen3.5:35b"],
                "vision_models": ["qwen3.5:9b", "qwen3.5:27b"],
                "aliases": ["thor", "server", "jetson", "jetson thor"],
                "keep_alive": "-1",
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
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.9,
    # Keep the Pi responsive. The application only needs a modest context window.
    "num_ctx": 8192,
}

BASE_SYSTEM_PROMPT = """You are the conversational AI inside a small Raspberry Pi 5 robot.
Be concise, useful, and natural. You have live tools for the clock, web search, the camera, and the robot's motion sensor.

Rules:
- Use look_at_camera whenever the user asks what you see, asks about the surroundings, or asks a question that requires visual inspection. Never guess what the camera sees.
- Use get_motion_state when a question depends on whether the robot is moving or was moved.
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
            "description": "Read whether the robot is moving or was recently moved, including accelerometer/gyro values.",
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
        master.bind("<Escape>", self.exit_fullscreen)
        master.bind("<Return>", self.handle_ptt_toggle)
        master.bind("<space>", self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)

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
        self.current_audio_process: Optional[subprocess.Popen] = None
        self.exiting = False
        self.last_ptt_time = 0.0

        self.ai = AIBackendManager(CURRENT_CONFIG.get("ai", {}), OLLAMA_OPTIONS)
        self.motion = MPU6050Monitor(CURRENT_CONFIG.get("hardware", {}).get("mpu6050", {}))
        self.vision = CameraVision(CURRENT_CONFIG.get("camera", {}))
        self.motion.start()

        print("[INIT] Loading wake word...", flush=True)
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                try:
                    self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                except TypeError:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                print("[INIT] Wake word loaded.", flush=True)
            except Exception as exc:
                print(f"[WAKE] Could not load model: {exc}", flush=True)
        else:
            print(f"[WAKE] Model not found: {WAKE_WORD_MODEL}; PTT only.", flush=True)

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
        threading.Thread(target=self.safe_main_execution, daemon=True, name="agent-main").start()

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------
    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)
        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.interrupted.set()
        with self.tts_queue_lock:
            self.tts_queue.clear()
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
            except Exception:
                pass
        self.save_chat_history()
        self.motion.stop()
        self.vision.close()
        self.ai.unload()
        try:
            sd.stop()
        except Exception:
            pass
        try:
            self.master.quit()
        except Exception:
            pass

    def exit_fullscreen(self, event=None):
        self.master.attributes("-fullscreen", False)
        self.safe_exit()

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
        now = time.time()
        if now - self.last_ptt_time < 0.5:
            return
        self.last_ptt_time = now
        if self.recording_active.is_set():
            self.recording_active.clear()
        elif self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
            self.recording_active.set()
            self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state not in (BotStates.SPEAKING, BotStates.THINKING):
            return
        self.interrupted.set()
        self.thinking_sound_active.clear()
        with self.tts_queue_lock:
            self.tts_queue.clear()
        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
            except Exception:
                pass
        self.set_state(BotStates.IDLE, "Interrupted.")

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
        frames = self.animations.get(self.current_state) or self.animations[BotStates.IDLE]
        if self.current_state == BotStates.SPEAKING and len(frames) > 1:
            self.current_frame_index = random.randint(1, len(frames) - 1)
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)
        self.background_label.config(image=frames[self.current_frame_index])
        self.master.after(50 if self.current_state == BotStates.SPEAKING else 500, self.update_animation)

    def set_state(self, state: str, msg: str = "", cam_path: Optional[str] = None):
        def _update():
            if msg:
                print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg:
                self.status_var.set(msg)
            show_path = cam_path if cam_path and os.path.exists(cam_path) else None
            if show_path and state in (BotStates.THINKING, BotStates.SPEAKING):
                try:
                    image = Image.open(show_path)
                    image.thumbnail((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(image)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(relx=0.5, rely=0.40, anchor=tk.CENTER)
                    return
                except Exception:
                    pass
            self.overlay_label.place_forget()

        try:
            self.master.after(0, _update)
        except tk.TclError:
            pass

    def append_to_text(self, text: str, newline: bool = True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, text + ("\n" if newline else ""))
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
        try:
            self.master.after(0, _update)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # Main listening loop
    # ------------------------------------------------------------------
    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True, name="tts-worker")
            self.tts_thread.start()

            while not self.exiting:
                trigger_source = self.detect_wake_word_or_ptt()
                if self.exiting:
                    break
                if self.interrupted.is_set():
                    self.interrupted.clear()
                    self.set_state(BotStates.IDLE, "Resetting...")
                    continue

                self.set_state(BotStates.LISTENING, "I'm listening!")
                audio_file = self.record_voice_ptt() if trigger_source == "PTT" else self.record_voice_adaptive()
                if not audio_file:
                    self.set_state(BotStates.IDLE, "Heard nothing.")
                    continue

                user_text = self.transcribe_audio(audio_file)
                if not user_text:
                    self.set_state(BotStates.IDLE, "Transcription empty.")
                    continue

                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.chat_and_respond(user_text)
        except Exception as exc:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Fatal Error: {str(exc)[:60]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Warming up local AI...")
        try:
            self.ai.warmup()
            print(f"[AI] {self.ai.status_text()}", flush=True)
        except Exception as exc:
            print(f"[AI] Warmup warning: {exc}", flush=True)
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        self.set_state(BotStates.IDLE, "Ready")

    # ------------------------------------------------------------------
    # Wake word and recording
    # ------------------------------------------------------------------
    def detect_wake_word_or_ptt(self) -> str:
        self.set_state(BotStates.IDLE, "Waiting...")
        self.ptt_event.clear()
        if self.oww_model:
            self.oww_model.reset()
        if self.oww_model is None:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

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

                    # Keep the original CLI-enter fallback when running in a terminal.
                    try:
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                        if rlist:
                            sys.stdin.readline()
                            return "PTT"
                    except Exception:
                        pass

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

                    current_max = int(np.max(np.abs(audio))) if len(audio) else 0
                    if current_max <= 200:
                        continue
                    self.oww_model.predict(audio)
                    for model_name, scores in self.oww_model.prediction_buffer.items():
                        if not scores:
                            continue
                        score = float(list(scores)[-1])
                        if score > WAKE_WORD_THRESHOLD:
                            print(f"[WAKE] {model_name}: {score:.2f}", flush=True)
                            self.oww_model.reset()
                            return "WAKE"
        except Exception as exc:
            print(f"[WAKE] Stream failed: {exc}; falling back to PTT.", flush=True)
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"
        return "WAKE"

    def record_voice_adaptive(self, filename: str = "input.wav") -> Optional[str]:
        print("[AUDIO] Recording adaptive...", flush=True)
        time.sleep(0.35)
        samplerate = choose_input_samplerate(INPUT_DEVICE, CURRENT_CONFIG.get("input_sample_rate"))
        silence_threshold = float(CURRENT_CONFIG.get("silence_threshold", 0.006))
        silence_duration = 1.5
        max_record_time = 30.0
        chunk_duration = 0.05
        chunk_size = int(samplerate * chunk_duration)
        buffer: List[np.ndarray] = []
        silent_chunks = 0
        recorded_chunks = 0
        silence_started = False
        required_silent = int(silence_duration / chunk_duration)
        max_chunks = int(max_record_time / chunk_duration)

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started
            buffer.append(indata.copy())
            recorded_chunks += 1
            if recorded_chunks < 5:
                return
            volume = np.linalg.norm(indata) / np.sqrt(max(1, len(indata)))
            if volume < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= required_silent:
                    silence_started = True
            else:
                silent_chunks = 0

        try:
            sd.stop()
            time.sleep(0.1)
            with sd.InputStream(
                samplerate=samplerate,
                channels=1,
                callback=callback,
                device=INPUT_DEVICE,
                blocksize=chunk_size,
            ):
                while not silence_started and recorded_chunks < max_chunks and not self.exiting:
                    sd.sleep(int(chunk_duration * 1000))
        except Exception as exc:
            print(f"[AUDIO] Adaptive recording failed: {exc}", flush=True)
            return None
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
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE):
                while self.recording_active.is_set() and not self.exiting:
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
            result = subprocess.run(
                [executable, "-m", model, "-l", "en", "-t", "4", "-f", filename],
                capture_output=True,
                text=True,
                timeout=120,
            )
            segments = []
            for line in result.stdout.splitlines():
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
        messages.extend(self.history)
        messages.extend(self.session_memory)
        messages.append({"role": "user", "content": text})
        self.session_memory.append({"role": "user", "content": text})

        self.set_state(BotStates.THINKING, f"Thinking on {self.ai.current_backend}...")
        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        final_text = ""
        try:
            for _ in range(5):
                if self.interrupted.is_set():
                    return
                response = self.ai.chat(messages, tools=TOOL_SCHEMAS)
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
        while not self.exiting:
            with self.tts_queue_lock:
                queue_empty = not self.tts_queue
            if queue_empty and not self.tts_active.is_set():
                return
            if self.interrupted.is_set():
                return
            time.sleep(0.05)

    def _tts_worker(self):
        while not self.exiting:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
            if text:
                self.tts_active.set()
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

        try:
            self.current_audio_process = subprocess.Popen(
                ["./piper/piper", "--model", voice_model, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
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
        time.sleep(0.4)
        while self.thinking_sound_active.is_set() and not self.exiting:
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound:
                self.play_sound(sound)
            for _ in range(40):
                if not self.thinking_sound_active.is_set() or self.exiting:
                    return
                time.sleep(0.1)

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
        combined = (self.history + self.session_memory)[-16:]
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as handle:
                json.dump(combined, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[MEMORY] Save failed: {exc}", flush=True)


if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
