"""Shared inference, serialization, and drawing helpers."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from src.config import PROJECT_ROOT, get_confidence_tier


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def require_ultralytics():
    # Keep settings/cache writes inside the project in restricted environments.
    config_root = PROJECT_ROOT / ".ultralytics"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return YOLO


def select_device(requested: str = "auto") -> str | int:
    if requested != "auto":
        return requested
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_is_grape_detector(model: Any) -> bool:
    names = model.names
    values = names.values() if isinstance(names, dict) else names
    return any(str(name).lower() == "grape_cluster" for name in values)


def result_to_detections(result: Any, image_id: str, frame_id: int | None = None) -> list[dict]:
    detections: list[dict] = []
    if result.boxes is None:
        return detections
    names = result.names
    for index, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = names[class_id] if isinstance(names, dict) else names[class_id]
        detections.append(
            {
                "image_id": image_id,
                "frame_id": frame_id,
                "detection_id": f"G{index:03d}",
                "class_id": class_id,
                "class": str(class_name),
                "confidence": round(confidence, 6),
                "confidence_tier": get_confidence_tier(confidence),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": round((x1 + x2) / 2.0, 2),
                "center_y": round((y1 + y2) / 2.0, 2),
                "review_status": "REVIEW_REQUIRED",
            }
        )
    return detections


def predictions_to_detections(
    predictions: Iterable[dict], image_id: str, frame_id: int | None = None,
) -> list[dict]:
    """Convert generic ``box``/``confidence`` predictions to the export schema."""
    detections = []
    for index, prediction in enumerate(predictions, start=1):
        x1, y1, x2, y2 = (float(value) for value in prediction["box"])
        confidence = float(prediction["confidence"])
        detections.append(
            {
                "image_id": image_id,
                "frame_id": frame_id,
                "detection_id": f"G{index:03d}",
                "class_id": 0,
                "class": "grape_cluster",
                "confidence": round(confidence, 6),
                "confidence_tier": get_confidence_tier(confidence),
                "x1": round(x1, 2), "y1": round(y1, 2),
                "x2": round(x2, 2), "y2": round(y2, 2),
                "center_x": round((x1 + x2) / 2.0, 2),
                "center_y": round((y1 + y2) / 2.0, 2),
                "review_status": "REVIEW_REQUIRED",
            }
        )
    return detections


def draw_detections(image: np.ndarray, detections: Iterable[dict]) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    colours = {
        "HIGH_CONFIDENCE": (35, 180, 35),
        "REVIEW": (0, 180, 255),
        "LOW_CONFIDENCE": (40, 40, 230),
    }
    for item in detections:
        x1, y1, x2, y2 = (int(round(item[k])) for k in ("x1", "y1", "x2", "y2"))
        colour = colours.get(item.get("confidence_tier"), (35, 180, 35))
        label = f"{item['detection_id']} {item['class']} {item['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        # A compact font keeps the required class/confidence label readable in
        # dense scenes without covering as much of the vineyard image.
        font_scale, text_thickness = 0.42, 1
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        label_x = max(0, min(x1, width - tw - 8))
        if y1 >= th + 10:
            top, bottom, text_y = y1 - th - 10, y1, y1 - 5
        else:
            top = max(0, y1)
            bottom = min(height - 1, y1 + th + 10)
            text_y = min(height - 5, y1 + th + 5)
        cv2.rectangle(canvas, (label_x, top), (label_x + tw + 8, bottom), colour, -1)
        cv2.putText(canvas, label, (label_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return canvas


def write_detection_exports(detections: list[dict], output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    csv_path = output_stem.with_suffix(".csv")
    json_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    fieldnames = [
        "image_id", "frame_id", "detection_id", "class_id", "class", "confidence",
        "confidence_tier", "x1", "y1", "x2", "y2", "center_x", "center_y",
        "review_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections)
    return json_path, csv_path


def timed_predict(model: Any, source: Any, **kwargs: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = model.predict(source=source, verbose=False, **kwargs)[0]
    return result, (time.perf_counter() - started) * 1000.0


def write_resolved_dataset_yaml(source: Path, destination: Path) -> Path:
    """Create a runtime YAML with an absolute root while keeping source YAML portable."""
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    configured_root = Path(config.get("path", "."))
    if not configured_root.is_absolute():
        configured_root = (source.parent / configured_root).resolve()
    config["path"] = configured_root.as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination
