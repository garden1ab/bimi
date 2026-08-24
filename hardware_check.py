#!/usr/bin/env python3
"""Target diagnostics for Raspberry Pi 5 + AI HAT+ + GY-521 + audio.

This file intentionally uses the Python standard library for its top-level
checks so it can still explain a broken/missing project venv.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command(name: str) -> str | None:
    return shutil.which(name)


def run(label: str, argv: list[str], timeout: int = 15) -> tuple[bool, str]:
    print(f"\n[{label}] {' '.join(argv)}")
    if command(argv[0]) is None:
        msg = f"MISSING COMMAND: {argv[0]}"
        print(msg)
        return False, msg
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout + result.stderr).strip()
        print(output if output else f"exit={result.returncode}")
        return result.returncode == 0, output
    except Exception as exc:
        print(f"ERROR: {exc}")
        return False, str(exc)


def os_info() -> dict[str, str]:
    data: dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value.strip().strip('"')
    return data


def check_venv() -> bool:
    print("\n[Project virtual environment]")
    activate = ROOT / "venv" / "bin" / "activate"
    python = ROOT / "venv" / "bin" / "python"
    if activate.exists() and python.exists():
        print(f"OK: {ROOT / 'venv'}")
        return True
    print("MISSING: ./venv")
    print("Fix: run ./setup.sh from the project directory.")
    print("The updated setup creates the venv before optional Hailo/Piper/Ollama steps.")
    return False


def check_i2c() -> bool:
    print("\n[I2C / GY-521]")
    dev = Path("/dev/i2c-1")
    if not dev.exists():
        print("FAIL: /dev/i2c-1 does not exist.")
        print("Run: sudo raspi-config nonint do_i2c 0")
        print("Then reboot: sudo reboot")
        return False

    print("OK: /dev/i2c-1 exists")
    ok, output = run("I2C scan", ["i2cdetect", "-y", "1"])
    if not ok:
        print("Install the scan utility with: sudo apt install i2c-tools")
        return False

    # i2cdetect may show 68 or UU. UU means a kernel driver owns the address.
    detected = any(token.lower() in {"68", "uu"} for token in output.split())
    if detected:
        print("GY-521 candidate detected on I2C bus 1.")
    else:
        print("FAIL: address 0x68 was not visible in the scan.")
        print("Check GY-521 wiring: VCC->3.3V, GND->GND, SDA->GPIO2/pin3, SCL->GPIO3/pin5.")
        print("If AD0 is tied high, the MPU-6050 address is 0x69 instead of 0x68.")
        return False

    # Try WHO_AM_I without requiring smbus2.
    if command("i2cget"):
        who_ok, who_output = run("MPU-6050 WHO_AM_I", ["i2cget", "-y", "1", "0x68", "0x75"])
        if who_ok:
            value = who_output.splitlines()[-1].strip().lower()
            if value == "0x68":
                print("OK: MPU-6050 WHO_AM_I is 0x68")
                return True
            print(f"Unexpected WHO_AM_I value: {value}")
            return False
    return detected


def check_hailo() -> bool:
    print("\n[Hailo AI HAT+]")
    cli = command("hailortcli")
    if cli:
        print(f"hailortcli: {cli}")
        ok, _ = run("Hailo identify", ["hailortcli", "fw-control", "identify"])
        if ok:
            return True
        print("hailortcli is installed but the device is not ready. Reboot after hailo-all/DKMS installation.")
    else:
        print("FAIL: hailortcli is not installed.")

    if command("dpkg-query"):
        run("hailo-all package", ["dpkg-query", "-W", "-f=${Status} ${Version}\\n", "hailo-all"])
    if command("apt-cache"):
        available, _ = run("hailo-all repository", ["apt-cache", "policy", "hailo-all"])
        if not available:
            print("The Raspberry Pi repository may not provide hailo-all for this OS/repository configuration.")

    if Path("/dev/hailo0").exists():
        print("Kernel device exists: /dev/hailo0")
    else:
        print("Kernel device missing: /dev/hailo0")

    if command("lspci"):
        _, pci = run("PCIe devices", ["lspci", "-nn"])
        lines = [line for line in pci.splitlines() if "hailo" in line.lower()]
        if lines:
            print("Hailo PCIe hardware is visible:")
            for line in lines:
                print("  " + line)
        else:
            print("No Hailo device was identified by name in lspci output.")

    print("Fix on current Raspberry Pi OS: sudo apt update && sudo apt install dkms hailo-all && sudo reboot")
    return False


def check_camera() -> bool:
    binary = "rpicam-still" if command("rpicam-still") else "rpicam-hello"
    if not command(binary):
        print("\n[Camera]\nMISSING: rpicam-apps. Install with: sudo apt install rpicam-apps")
        return False
    argv = [binary, "--list-cameras"] if binary == "rpicam-still" else [binary, "--list-cameras"]
    ok, _ = run("Camera", argv)
    return ok


def check_ollama(name: str, base_url: str) -> bool:
    print(f"\n[{name} AI] {base_url}")
    if not base_url:
        print("No base_url configured")
        return False
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = [m.get("name") or m.get("model") for m in payload.get("models", [])]
        print("Models:", ", ".join(x for x in models if x) or "none")
        return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"Unavailable: {exc}")
        return False


def main() -> int:
    print("Be More Agent hardware check")
    info = os_info()
    try:
        model = Path("/proc/device-tree/model").read_bytes().replace(b"\x00", b"").decode(errors="ignore")
    except Exception:
        model = "unknown"
    print(f"Board: {model}")
    print(f"Platform: {platform.platform()} | machine={platform.machine()}")
    print(f"OS: {info.get('PRETTY_NAME', 'unknown')} | codename={info.get('VERSION_CODENAME', 'unknown')}")
    print(f"Python: {sys.executable}")

    config_path = ROOT / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    print("\n[Configured wiring]")
    print("GY-521: SDA GPIO2/pin3, SCL GPIO3/pin5, INT GPIO24/pin18")
    print("MAX98357A: DIN GPIO21/pin40, BCLK GPIO18/pin12, LRCLK GPIO19/pin35")

    results = {
        "venv": check_venv(),
        "i2c": check_i2c(),
        "hailo": check_hailo(),
        "camera": check_camera(),
    }

    audio_play, _ = run("ALSA playback", ["aplay", "-l"])
    audio_cap, _ = run("ALSA capture / USB microphone", ["arecord", "-l"])
    results["audio_playback"] = audio_play
    results["audio_capture"] = audio_cap

    backends = config.get("ai", {}).get("backends", {})
    for name in ("local", "thor"):
        backend = backends.get(name, {})
        if backend.get("type", "ollama") == "ollama":
            results[f"ai_{name}"] = check_ollama(name.upper(), backend.get("base_url", ""))

    print("\n[Summary]")
    for key, value in results.items():
        print(f"  {'PASS' if value else 'FAIL'}  {key}")

    core_ok = results.get("venv", False) and results.get("i2c", False)
    if not core_ok:
        print("\nRun ./setup.sh, reboot, then rerun this check before starting the agent.")
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
