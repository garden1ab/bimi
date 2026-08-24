"""Raspberry Pi hardware integration for the GY-521 / MPU-6050 motion sensor."""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, Optional

try:
    from smbus2 import SMBus
except Exception:  # pragma: no cover - target hardware dependency
    SMBus = None

try:
    from gpiozero import DigitalInputDevice
except Exception:  # pragma: no cover - target hardware dependency
    DigitalInputDevice = None


class MPU6050Monitor:
    MPU6050_ADDR = 0x68

    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    MOT_THR = 0x1F
    MOT_DUR = 0x20
    INT_PIN_CFG = 0x37
    INT_ENABLE = 0x38
    INT_STATUS = 0x3A
    ACCEL_XOUT_H = 0x3B

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.bus_id = int(self.config.get("i2c_bus", 1))
        self.address = int(str(self.config.get("address", "0x68")), 0)
        self.interrupt_gpio = int(self.config.get("interrupt_gpio", 24))
        self.poll_hz = float(self.config.get("poll_hz", 20.0))
        self.accel_delta_threshold_g = float(self.config.get("accel_delta_threshold_g", 0.08))
        self.accel_magnitude_threshold_g = float(self.config.get("accel_magnitude_threshold_g", 0.12))
        self.gyro_threshold_dps = float(self.config.get("gyro_threshold_dps", 18.0))
        self.motion_hold_seconds = float(self.config.get("motion_hold_seconds", 2.5))

        self._bus = None
        self._gpio = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._available = False
        self._last_error = ""
        self._last_vector = None
        self._last_motion_time = 0.0
        self._last_interrupt_time = 0.0
        self._motion_count = 0
        self._accel = (0.0, 0.0, 0.0)
        self._gyro = (0.0, 0.0, 0.0)
        self._temperature_c = 0.0

    def start(self) -> bool:
        if not self.enabled:
            return False
        if SMBus is None:
            self._last_error = "smbus2 is not installed"
            return False
        try:
            self._bus = SMBus(self.bus_id)
            # Wake the MPU-6050 and use conservative +/-2g, +/-250 dps ranges.
            self._write(self.PWR_MGMT_1, 0x00)
            time.sleep(0.05)
            self._write(self.CONFIG, 0x03)
            self._write(self.GYRO_CONFIG, 0x00)
            # DHPF=5 Hz improves the hardware motion detector while retaining +/-2g.
            self._write(self.ACCEL_CONFIG, 0x01)

            # Motion interrupt: threshold register is ~2 mg/LSB, duration ~1 ms/LSB.
            threshold_mg = int(self.config.get("interrupt_threshold_mg", 80))
            duration_ms = int(self.config.get("interrupt_duration_ms", 20))
            self._write(self.MOT_THR, max(1, min(255, round(threshold_mg / 2))))
            self._write(self.MOT_DUR, max(1, min(255, duration_ms)))
            # Latch interrupt until INT_STATUS is read. Active high, push-pull.
            self._write(self.INT_PIN_CFG, 0x20)
            self._write(self.INT_ENABLE, 0x40)

            # Confirm the device answers by reading sensor data once.
            self._read_sample()
            self._available = True

            if DigitalInputDevice is not None:
                try:
                    self._gpio = DigitalInputDevice(
                        self.interrupt_gpio,
                        pull_up=False,
                        bounce_time=0.01,
                    )
                    self._gpio.when_activated = self._on_motion_interrupt
                except Exception as exc:
                    print(f"[MPU6050] GPIO{self.interrupt_gpio} interrupt unavailable: {exc}", flush=True)
                    self._gpio = None

            self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="mpu6050-monitor")
            self._thread.start()
            print(
                f"[MPU6050] Ready on I2C-{self.bus_id} address 0x{self.address:02X}; "
                f"INT GPIO{self.interrupt_gpio}",
                flush=True,
            )
            return True
        except Exception as exc:
            self._last_error = str(exc)
            print(f"[MPU6050] Initialization failed: {exc}", flush=True)
            self.stop()
            return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._gpio is not None:
            try:
                self._gpio.close()
            except Exception:
                pass
            self._gpio = None
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None

    def _write(self, register: int, value: int) -> None:
        self._bus.write_byte_data(self.address, register, value)

    @staticmethod
    def _signed16(high: int, low: int) -> int:
        value = (high << 8) | low
        return value - 65536 if value & 0x8000 else value

    def _read_sample(self) -> None:
        block = self._bus.read_i2c_block_data(self.address, self.ACCEL_XOUT_H, 14)
        ax = self._signed16(block[0], block[1]) / 16384.0
        ay = self._signed16(block[2], block[3]) / 16384.0
        az = self._signed16(block[4], block[5]) / 16384.0
        temp_raw = self._signed16(block[6], block[7])
        gx = self._signed16(block[8], block[9]) / 131.0
        gy = self._signed16(block[10], block[11]) / 131.0
        gz = self._signed16(block[12], block[13]) / 131.0

        accel = (ax, ay, az)
        gyro = (gx, gy, gz)
        now = time.monotonic()

        accel_mag = math.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)
        delta = 0.0
        if self._last_vector is not None:
            delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(accel, self._last_vector)))
        self._last_vector = accel

        motion = (
            abs(accel_mag - 1.0) >= self.accel_magnitude_threshold_g
            or delta >= self.accel_delta_threshold_g
            or gyro_mag >= self.gyro_threshold_dps
        )

        with self._lock:
            self._accel = accel
            self._gyro = gyro
            self._temperature_c = temp_raw / 340.0 + 36.53
        if motion:
            self._mark_motion(now)

    def _on_motion_interrupt(self, _device=None) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_interrupt_time = now
        self._mark_motion(now)
        # Reading INT_STATUS clears the latched interrupt.
        try:
            self._bus.read_byte_data(self.address, self.INT_STATUS)
        except Exception:
            pass

    def _mark_motion(self, when: Optional[float] = None) -> None:
        now = when or time.monotonic()
        with self._lock:
            if now - self._last_motion_time > 0.20:
                self._motion_count += 1
            self._last_motion_time = now

    def _poll_loop(self) -> None:
        delay = 1.0 / max(1.0, self.poll_hz)
        while not self._stop_event.is_set():
            try:
                self._read_sample()
                # Clear any latched interrupt even if gpiozero is unavailable.
                status = self._bus.read_byte_data(self.address, self.INT_STATUS)
                if status & 0x40:
                    self._mark_motion()
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
            self._stop_event.wait(delay)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            age = None if self._last_motion_time <= 0 else max(0.0, now - self._last_motion_time)
            moving = age is not None and age <= self.motion_hold_seconds
            return {
                "available": self._available,
                "moving": moving,
                "seconds_since_motion": age,
                "motion_count": self._motion_count,
                "acceleration_g": tuple(round(x, 3) for x in self._accel),
                "gyro_dps": tuple(round(x, 1) for x in self._gyro),
                "temperature_c": round(self._temperature_c, 1),
                "interrupt_gpio": self.interrupt_gpio,
                "last_error": self._last_error,
            }

    def context_for_llm(self) -> str:
        state = self.snapshot()
        if not state["available"]:
            return f"Motion sensor unavailable. {state.get('last_error', '')}".strip()
        if state["moving"]:
            age = state["seconds_since_motion"] or 0.0
            movement = f"The robot is being moved or was moved {age:.1f} seconds ago."
        else:
            age = state["seconds_since_motion"]
            movement = "The robot is currently stationary."
            if age is not None:
                movement += f" The last detected movement was {age:.1f} seconds ago."
        return (
            f"{movement} Accelerometer XYZ={state['acceleration_g']} g; "
            f"gyro XYZ={state['gyro_dps']} deg/s."
        )
