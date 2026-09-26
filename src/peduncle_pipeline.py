"""Two-stage grape-cluster and peduncle segmentation inference.

Stage 1 detects grape clusters in the complete image.  Stage 2 examines an
enlarged ROI above each cluster, segments the most plausible peduncle, and
estimates a visible cutting point.  Results are saved as annotated images and
JSON so they can later feed robot planning and human review.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.build_peduncle_dataset import roi_from_cluster_box
from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device
from src.config import (
    DEFAULT_PEDUNCLE_CONF_THRESHOLD,
    MODELS_DIR,
    PEDUNCLE_SEGMENTATION_MODEL,
    PROJECT_ROOT,
    RESULTS_DIR,
)


DEFAULT_CLUSTER_MODEL = MODELS_DIR / "grape_yolo11n_vinepics_wgisd_canopies_best.pt"
DEFAULT_PEDUNCLE_MODEL = PEDUNCLE_SEGMENTATION_MODEL
DEFAULT_OUTPUT = RESULTS_DIR / "peduncle_pipeline"


def mask_anchor(mask: np.ndarray) -> tuple[float, float] | None:
    """Return the lower attachment-side anchor of a binary peduncle mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    threshold = np.quantile(ys, 0.90)
    selected = xs[ys >= threshold]
    return float(np.median(selected)), float(np.max(ys))


def estimate_cut_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Choose a stable point in the upper-middle visible peduncle section."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    target_y = float(np.quantile(ys, 0.35))
    band = max(2, int(round((ys.max() - ys.min() + 1) * 0.06)))
    selected = np.abs(ys - target_y) <= band
    if not np.any(selected):
        index = int(np.argmin(np.abs(ys - target_y)))
        return int(xs[index]), int(ys[index])
    return int(round(float(np.median(xs[selected])))), int(round(float(np.median(ys[selected]))))


def mask_polygon(mask: np.ndarray, epsilon: float = 1.5) -> list[list[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[int(x), int(y)] for x, y in simplified]


def _candidate_mask(result: Any, index: int, shape: tuple[int, int]) -> np.ndarray:
    mask = result.masks.data[index].cpu().numpy()
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def choose_peduncle(result: Any, cluster_box_local: tuple[float, float, float, float], crop_shape):
    """Select the mask whose attachment anchor best matches the cluster top."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None
    cluster_x = (cluster_box_local[0] + cluster_box_local[2]) / 2.0
    cluster_y = cluster_box_local[1]
    cluster_scale = max(
        cluster_box_local[2] - cluster_box_local[0],
        cluster_box_local[3] - cluster_box_local[1], 1.0,
    )
    candidates = []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = result.names[class_id] if isinstance(result.names, dict) else result.names[class_id]
        if str(class_name).lower() != "peduncle":
            continue
        mask = _candidate_mask(result, index, crop_shape[:2])
        anchor = mask_anchor(mask)
        if anchor is None:
            continue
        distance = float(np.hypot(anchor[0] - cluster_x, anchor[1] - cluster_y)) / cluster_scale
        confidence = float(box.conf[0])
        candidates.append((confidence - 0.30 * distance, confidence, mask, box))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def analyze_image(
    image: np.ndarray, cluster_model: Any, peduncle_model: Any, device: str | int,
    cluster_conf: float, peduncle_conf: float, imgsz: int, peduncle_batch: int = 8,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    cluster_result = cluster_model.predict(
        image, conf=cluster_conf, iou=0.50, imgsz=imgsz, device=device, verbose=False,
    )[0]
    height, width = image.shape[:2]
    detections: list[dict] = []
    crop_jobs: list[tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int, int, int], dict]] = []
    boxes = cluster_result.boxes if cluster_result.boxes is not None else []
    for index, box in enumerate(boxes, start=1):
        cluster_box = tuple(float(value) for value in box.xyxy[0].tolist())
        roi = roi_from_cluster_box(cluster_box, width, height)
        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        local_box = (
            cluster_box[0] - x1, cluster_box[1] - y1,
            cluster_box[2] - x1, cluster_box[3] - y1,
        )
        item = {
            "cluster_id": f"G{index:03d}",
            "cluster_confidence": round(float(box.conf[0]), 6),
            "cluster_box": [round(value, 2) for value in cluster_box],
            "peduncle_roi": [x1, y1, x2, y2],
            "peduncle_status": "NOT_FOUND",
            "peduncle_confidence": None,
            "peduncle_box": None,
            "cut_point": None,
            "peduncle_polygon": [],
            "_mask": None,
        }
        detections.append(item)
        crop_jobs.append((crop, local_box, roi, item))

    # Batched second-stage inference avoids one GPU launch per detected bunch.
    # Chunks cap peak memory on dense vineyard images.
    peduncle_batch = max(1, peduncle_batch)
    for start in range(0, len(crop_jobs), peduncle_batch):
        batch = crop_jobs[start:start + peduncle_batch]
        results = peduncle_model.predict(
            [job[0] for job in batch], conf=peduncle_conf, iou=0.45,
            imgsz=imgsz, device=device, retina_masks=True, verbose=False,
        )
        for (crop, local_box, roi, item), peduncle_result in zip(batch, results):
            selected = choose_peduncle(peduncle_result, local_box, crop.shape)
            if selected is None:
                continue
            x1, y1, x2, y2 = roi
            _, confidence, local_mask, selected_box = selected
            global_mask = np.zeros((height, width), dtype=bool)
            global_mask[y1:y2, x1:x2] = local_mask
            local_cut = estimate_cut_point(local_mask)
            local_xyxy = [float(value) for value in selected_box.xyxy[0].tolist()]
            polygon = mask_polygon(local_mask)
            item.update({
                "peduncle_status": "FOUND",
                "peduncle_confidence": round(confidence, 6),
                "peduncle_box": [
                    round(local_xyxy[0] + x1, 2), round(local_xyxy[1] + y1, 2),
                    round(local_xyxy[2] + x1, 2), round(local_xyxy[3] + y1, 2),
                ],
                "cut_point": [local_cut[0] + x1, local_cut[1] + y1] if local_cut else None,
                "peduncle_polygon": [[px + x1, py + y1] for px, py in polygon],
                "_mask": global_mask,
            })
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return detections, elapsed_ms


def draw_pipeline(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image.copy()
    overlay = canvas.copy()
    for item in detections:
        mask = item.get("_mask")
        if mask is not None:
            overlay[mask] = (255, 210, 0)
    canvas = cv2.addWeighted(canvas, 0.68, overlay, 0.32, 0)
    for item in detections:
        x1, y1, x2, y2 = map(lambda value: int(round(value)), item["cluster_box"])
        found = item["peduncle_status"] == "FOUND"
        colour = (40, 190, 40) if found else (0, 165, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        ped_text = (
            f" P {item['peduncle_confidence']:.2f}" if found else " P missing"
        )
        label = f"{item['cluster_id']} G {item['cluster_confidence']:.2f}{ped_text}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, colour, 2, cv2.LINE_AA)
        if item["peduncle_box"]:
            px1, py1, px2, py2 = map(lambda value: int(round(value)), item["peduncle_box"])
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (255, 210, 0), 2)
        if item["cut_point"]:
            point = tuple(map(int, item["cut_point"]))
            cv2.circle(canvas, point, 6, (220, 0, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 10, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def serializable(detections: list[dict]) -> list[dict]:
    return [{key: value for key, value in item.items() if not key.startswith("_")} for item in detections]


def input_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"No supported image source found: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cluster-model", type=Path, default=DEFAULT_CLUSTER_MODEL)
    parser.add_argument("--peduncle-model", type=Path, default=DEFAULT_PEDUNCLE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-conf", type=float, default=0.25)
    parser.add_argument("--peduncle-conf", type=float, default=DEFAULT_PEDUNCLE_CONF_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--peduncle-batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=0, help="0 processes all images")
    args = parser.parse_args()
    for path in (args.cluster_model, args.peduncle_model):
        if not path.is_file():
            raise SystemExit(f"Model not found: {path}")
    YOLO = require_ultralytics()
    device = select_device(args.device)
    cluster_model, peduncle_model = YOLO(str(args.cluster_model)), YOLO(str(args.peduncle_model))
    images = input_images(args.source)
    if args.max_images > 0:
        images = images[:args.max_images]
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        detections, elapsed_ms = analyze_image(
            image, cluster_model, peduncle_model, device,
            args.cluster_conf, args.peduncle_conf, args.imgsz, args.peduncle_batch,
        )
        annotated = draw_pipeline(image, detections)
        output_image = args.output / f"{image_path.stem}_peduncles.jpg"
        cv2.imwrite(str(output_image), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        payload = {
            "image": str(image_path), "elapsed_ms": round(elapsed_ms, 2),
            "clusters": len(detections),
            "peduncles_found": sum(item["peduncle_status"] == "FOUND" for item in detections),
            "detections": serializable(detections),
        }
        output_json = args.output / f"{image_path.stem}_peduncles.json"
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summaries.append({key: payload[key] for key in ("image", "elapsed_ms", "clusters", "peduncles_found")})
        print(json.dumps(summaries[-1]))
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
