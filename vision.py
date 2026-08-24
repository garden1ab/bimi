"""Pi Camera capture plus optional Hailo-8 object perception."""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from picamera2.devices import Hailo, hailo_architecture
except Exception:  # pragma: no cover - only present on configured Raspberry Pi
    Hailo = None
    hailo_architecture = None


class CameraVision:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.image_path = str(self.config.get("image_path", "current_image.jpg"))
        self.width = int(self.config.get("width", 640))
        self.height = int(self.config.get("height", 480))
        self.rotation = int(self.config.get("rotation", 0))
        self.hailo_enabled = bool(self.config.get("hailo_enabled", True))
        self.hailo_threshold = float(self.config.get("hailo_threshold", 0.45))
        self.max_detections = int(self.config.get("max_detections", 12))
        self.labels_path = str(self.config.get("labels_path", "assets/coco.txt"))
        self.model_path = self.config.get("hailo_model_path")
        self._hailo = None
        self._hailo_model_path = None
        self._labels: List[str] = []
        self._hailo_error = ""

    def close(self) -> None:
        if self._hailo is not None:
            try:
                self._hailo.__exit__(None, None, None)
            except Exception:
                pass
            self._hailo = None

    def capture(self, image_path: Optional[str] = None) -> str:
        path = image_path or self.image_path
        command = [
            "rpicam-still",
            "-t", str(int(self.config.get("camera_warmup_ms", 500))),
            "-n",
            "--width", str(self.width),
            "--height", str(self.height),
            "-o", path,
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=max(3.0, float(self.config.get("capture_timeout_seconds", 12.0))),
        )
        if self.rotation:
            image = Image.open(path)
            image = image.rotate(self.rotation, expand=True)
            image.save(path)
        return path

    def capture_and_perceive(self, image_path: Optional[str] = None) -> Dict[str, Any]:
        path = self.capture(image_path)
        detections = self.detect(path)
        return {
            "image_path": path,
            "detections": detections,
            "hailo_available": self._hailo is not None,
            "hailo_model": self._hailo_model_path,
            "hailo_error": self._hailo_error,
            "summary": self.format_detections(detections),
        }

    def _resolve_hailo_model(self) -> Optional[str]:
        if self.model_path and os.path.exists(str(self.model_path)):
            return str(self.model_path)
        candidates = [
            "/usr/share/hailo-models/yolov8s_h8l.hef",
            "/usr/share/hailo-models/yolov8s_h8.hef",
            "/usr/share/hailo-models/yolov6n_h8l.hef",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_hailo(self) -> bool:
        if self._hailo is not None:
            return True
        if not self.hailo_enabled:
            return False
        if Hailo is None:
            self._hailo_error = "Picamera2 Hailo Python support is not installed"
            return False
        model = self._resolve_hailo_model()
        if not model:
            self._hailo_error = "No compatible Hailo HEF model found under /usr/share/hailo-models"
            return False
        try:
            self._hailo = Hailo(model)
            self._hailo.__enter__()
            self._hailo_model_path = model
            self._labels = self._load_labels()
            arch = hailo_architecture() if hailo_architecture else "unknown"
            print(f"[HAILO] Vision perception ready: {arch}, {model}", flush=True)
            return True
        except Exception as exc:
            self._hailo_error = str(exc)
            self._hailo = None
            print(f"[HAILO] Vision perception unavailable: {exc}", flush=True)
            return False

    def _load_labels(self) -> List[str]:
        path = Path(self.labels_path)
        if path.exists():
            return path.read_text(encoding="utf-8").splitlines()
        return []

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        if not self._load_hailo():
            return []
        try:
            model_h, model_w, _ = self._hailo.get_input_shape()
            image = Image.open(image_path).convert("RGB").resize((model_w, model_h))
            frame = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            output = self._hailo.run(frame)
            detections = self._extract_detections(output, model_w, model_h)
            self._hailo_error = ""
            return detections[: self.max_detections]
        except Exception as exc:
            self._hailo_error = str(exc)
            print(f"[HAILO] Detection failed: {exc}", flush=True)
            return []

    def _extract_detections(self, hailo_output: Any, width: int, height: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if hailo_output is None:
            return results
        # HailoRT-postprocess NMS output follows the same layout used by the
        # official Picamera2 Hailo detect.py example: one list per class.
        for class_id, class_detections in enumerate(hailo_output):
            for detection in class_detections:
                if len(detection) < 5:
                    continue
                score = float(detection[4])
                if score < self.hailo_threshold:
                    continue
                y0, x0, y1, x1 = [float(x) for x in detection[:4]]
                label = self._labels[class_id] if class_id < len(self._labels) else f"class_{class_id}"
                results.append(
                    {
                        "label": label,
                        "confidence": round(score, 3),
                        "bbox": [
                            int(x0 * width),
                            int(y0 * height),
                            int(x1 * width),
                            int(y1 * height),
                        ],
                    }
                )
        results.sort(key=lambda item: item["confidence"], reverse=True)
        return results

    @staticmethod
    def format_detections(detections: List[Dict[str, Any]]) -> str:
        if not detections:
            return "Hailo object detector found no confident objects."
        counts = Counter(d["label"] for d in detections)
        parts = [f"{count} {label}" for label, count in counts.most_common()]
        return "Hailo object detector saw: " + ", ".join(parts) + "."
