"""Direct two-class grape-cluster and peduncle segmentation pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.common import require_ultralytics, select_device
from src.config import (
    DEFAULT_DIRECT_CLUSTER_CONF_THRESHOLD,
    DEFAULT_DIRECT_IMGSZ,
    DEFAULT_DIRECT_PEDUNCLE_CONF_THRESHOLD,
    DIRECT_PEDUNCLE_SEGMENTATION_MODEL,
    RESULTS_DIR,
)
from src.peduncle_pipeline import (
    draw_pipeline,
    estimate_cut_point,
    input_images,
    mask_anchor,
    mask_polygon,
    serializable,
)


DEFAULT_MODEL = DIRECT_PEDUNCLE_SEGMENTATION_MODEL
DEFAULT_OUTPUT = RESULTS_DIR / "peduncle_pipeline" / "direct"


def result_mask(result: Any, index: int, shape: tuple[int, int]) -> np.ndarray:
    mask = result.masks.data[index].cpu().numpy()
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def association_score(cluster_box: list[float], peduncle_mask: np.ndarray, confidence: float) -> float | None:
    """Score a plausible stem-to-bunch attachment; reject impossible geometry."""
    anchor = mask_anchor(peduncle_mask)
    if anchor is None:
        return None
    x1, y1, x2, y2 = cluster_box
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    centre_x = (x1 + x2) / 2.0
    if not x1 - 0.45 * width <= anchor[0] <= x2 + 0.45 * width:
        return None
    if not y1 - 1.50 * height <= anchor[1] <= y1 + 0.35 * height:
        return None
    distance = float(np.hypot(anchor[0] - centre_x, anchor[1] - y1)) / max(width, height)
    return confidence - 0.30 * distance


def associate_instances(clusters: list[dict], peduncles: list[dict]) -> dict[int, int]:
    """Greedily form a one-to-one assignment from cluster to peduncle."""
    candidates = []
    for cluster_index, cluster in enumerate(clusters):
        for peduncle_index, peduncle in enumerate(peduncles):
            score = association_score(cluster["box"], peduncle["mask"], peduncle["confidence"])
            if score is not None:
                candidates.append((score, cluster_index, peduncle_index))
    assignments: dict[int, int] = {}
    used_peduncles: set[int] = set()
    for _, cluster_index, peduncle_index in sorted(candidates, reverse=True):
        if cluster_index in assignments or peduncle_index in used_peduncles:
            continue
        assignments[cluster_index] = peduncle_index
        used_peduncles.add(peduncle_index)
    return assignments


def analyze_direct(
    image: np.ndarray, model: Any, device: str | int,
    cluster_confidence: float, peduncle_confidence: float, imgsz: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    result = model.predict(
        image, conf=min(cluster_confidence, peduncle_confidence), iou=0.55,
        imgsz=imgsz, device=device,
        retina_masks=True, verbose=False,
    )[0]
    height, width = image.shape[:2]
    clusters: list[dict] = []
    peduncles: list[dict] = []
    if result.boxes is not None and result.masks is not None:
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = str(result.names[class_id]).lower()
            instance = {
                "box": [float(value) for value in box.xyxy[0].tolist()],
                "confidence": float(box.conf[0]),
                "mask": result_mask(result, index, (height, width)),
            }
            if class_name == "grape_cluster" and instance["confidence"] >= cluster_confidence:
                clusters.append(instance)
            elif class_name == "peduncle" and instance["confidence"] >= peduncle_confidence:
                peduncles.append(instance)
    clusters.sort(key=lambda item: item["confidence"], reverse=True)
    assignments = associate_instances(clusters, peduncles)
    detections = []
    for index, cluster in enumerate(clusters, start=1):
        item = {
            "cluster_id": f"G{index:03d}",
            "cluster_confidence": round(cluster["confidence"], 6),
            "cluster_box": [round(value, 2) for value in cluster["box"]],
            "peduncle_status": "NOT_FOUND", "peduncle_confidence": None,
            "peduncle_box": None, "cut_point": None, "peduncle_polygon": [],
            "_mask": None,
        }
        peduncle_index = assignments.get(index - 1)
        if peduncle_index is not None:
            peduncle = peduncles[peduncle_index]
            mask = peduncle["mask"]
            cut_point = estimate_cut_point(mask)
            item.update({
                "peduncle_status": "FOUND",
                "peduncle_confidence": round(peduncle["confidence"], 6),
                "peduncle_box": [round(value, 2) for value in peduncle["box"]],
                "cut_point": list(cut_point) if cut_point else None,
                "peduncle_polygon": mask_polygon(mask),
                "_mask": mask,
            })
        detections.append(item)
    return detections, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-conf", type=float, default=DEFAULT_DIRECT_CLUSTER_CONF_THRESHOLD)
    parser.add_argument("--peduncle-conf", type=float, default=DEFAULT_DIRECT_PEDUNCLE_CONF_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_DIRECT_IMGSZ)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    YOLO = require_ultralytics()
    model, device = YOLO(str(args.model)), select_device(args.device)
    images = input_images(args.source)
    if args.max_images > 0:
        images = images[:args.max_images]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        detections, elapsed_ms = analyze_direct(
            image, model, device, args.cluster_conf, args.peduncle_conf, args.imgsz,
        )
        cv2.imwrite(str(args.output / f"{image_path.stem}_direct.jpg"), draw_pipeline(image, detections))
        payload = {
            "image": str(image_path), "elapsed_ms": round(elapsed_ms, 2),
            "clusters": len(detections),
            "peduncles_found": sum(item["peduncle_status"] == "FOUND" for item in detections),
            "detections": serializable(detections),
        }
        (args.output / f"{image_path.stem}_direct.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        summary.append({key: payload[key] for key in ("image", "elapsed_ms", "clusters", "peduncles_found")})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
