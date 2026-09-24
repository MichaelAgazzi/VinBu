"""Sliding-window inference helpers for small objects in high-resolution images."""

from __future__ import annotations

from typing import Any

import numpy as np


def tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return origins that cover an axis, including an edge-aligned final tile."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1 - overlap))))
    origins = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def classless_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Apply confidence-ordered NMS to dictionaries containing ``box`` and ``confidence``."""
    if not detections:
        return []
    boxes = np.asarray([item["box"] for item in detections], dtype=np.float32)
    scores = np.asarray([item["confidence"] for item in detections], dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        overlaps = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[overlaps <= iou_threshold]
    return [detections[index] for index in keep]


def predict_tiled(
    model: Any,
    image: np.ndarray,
    *,
    conf: float,
    imgsz: int,
    device: str | int,
    tile_size: int = 640,
    overlap: float = 0.25,
    nms_iou: float = 0.5,
) -> list[dict]:
    """Predict on overlapping native-resolution tiles and merge boxes globally."""
    height, width = image.shape[:2]
    detections: list[dict] = []
    for y0 in tile_origins(height, tile_size, overlap):
        for x0 in tile_origins(width, tile_size, overlap):
            tile = image[y0:min(y0 + tile_size, height), x0:min(x0 + tile_size, width)]
            result = model.predict(
                tile, conf=conf, imgsz=imgsz, device=device, verbose=False
            )[0]
            if result.boxes is None:
                continue
            for box in result.boxes:
                left, top, right, bottom = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    {
                        "box": [left + x0, top + y0, right + x0, bottom + y0],
                        "confidence": float(box.conf[0]),
                    }
                )
    return classless_nms(detections, nms_iou)
