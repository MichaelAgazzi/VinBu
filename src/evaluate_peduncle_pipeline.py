"""Evaluate the complete cluster-to-peduncle pipeline on full held-out images.

Unlike ``evaluate_segmenter``, this script does not use ground-truth cluster
crops.  The detector must first find a cluster and the segmenter then receives
the detector-derived ROI, which reflects deployed behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device
from src.config import DEFAULT_PEDUNCLE_CONF_THRESHOLD, PROJECT_ROOT, RESULTS_DIR
from src.peduncle_pipeline import (
    DEFAULT_CLUSTER_MODEL,
    DEFAULT_PEDUNCLE_MODEL,
    analyze_image,
    draw_pipeline,
)
from src.sweep_segmentation_thresholds import mask_iou


DEFAULT_DATA = PROJECT_ROOT / "data" / "canopies_peduncle" / "full" / "dataset.yaml"


def split_paths(data_yaml: Path, split: str) -> tuple[list[Path], Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_dir = root / config[split]
    label_dir = root / config[split].replace("images", "labels")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return images, label_dir


def ground_truth(label_path: Path, shape: tuple[int, int]) -> tuple[list[list[float]], list[np.ndarray]]:
    """Read cluster boxes and peduncle masks from two-class YOLO polygons."""
    height, width = shape
    cluster_boxes: list[list[float]] = []
    peduncle_masks: list[np.ndarray] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            continue
        class_id = int(parts[0])
        normalized = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points = normalized * np.asarray([width, height], dtype=np.float32)
        if class_id == 0:
            cluster_boxes.append([
                float(points[:, 0].min()), float(points[:, 1].min()),
                float(points[:, 0].max()), float(points[:, 1].max()),
            ])
        elif class_id == 1:
            integer_points = np.rint(points).astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [integer_points], 1)
            peduncle_masks.append(mask.astype(bool))
    return cluster_boxes, peduncle_masks


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def greedy_matches(ground_truth_items, predictions, similarity, threshold: float) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (similarity(gt, prediction), gt_index, prediction_index)
            for gt_index, gt in enumerate(ground_truth_items)
            for prediction_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    matches: list[tuple[int, int, float]] = []
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    for score, gt_index, prediction_index in candidates:
        if score < threshold:
            break
        if gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index, score))
    return matches


def scores(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-model", type=Path, default=DEFAULT_CLUSTER_MODEL)
    parser.add_argument("--peduncle-model", type=Path, default=DEFAULT_PEDUNCLE_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--cluster-conf", type=float, default=0.25)
    parser.add_argument("--peduncle-conf", type=float, default=DEFAULT_PEDUNCLE_CONF_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--peduncle-batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "peduncle_pipeline" / "heldout_evaluation")
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    for path in (args.cluster_model, args.peduncle_model, args.data):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    YOLO = require_ultralytics()
    device = select_device(args.device)
    cluster_model = YOLO(str(args.cluster_model))
    peduncle_model = YOLO(str(args.peduncle_model))
    images, label_dir = split_paths(args.data, args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "cluster_tp": 0, "cluster_fp": 0, "cluster_fn": 0,
        "peduncle_tp": 0, "peduncle_fp": 0, "peduncle_fn": 0,
    }
    rows: list[dict] = []
    elapsed_times: list[float] = []
    sample_stride = max(1, len(images) // max(1, args.samples))
    for image_index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        gt_boxes, gt_masks = ground_truth(label_dir / f"{image_path.stem}.txt", image.shape[:2])
        detections, elapsed_ms = analyze_image(
            image, cluster_model, peduncle_model, device,
            args.cluster_conf, args.peduncle_conf, args.imgsz, args.peduncle_batch,
        )
        prediction_boxes = [item["cluster_box"] for item in detections]
        prediction_masks = [item["_mask"] for item in detections if item["_mask"] is not None]
        cluster_matches = greedy_matches(gt_boxes, prediction_boxes, box_iou, args.iou)
        peduncle_matches = greedy_matches(gt_masks, prediction_masks, mask_iou, args.iou)
        row = {
            "image": image_path.name,
            "elapsed_ms": round(elapsed_ms, 3),
            "gt_clusters": len(gt_boxes), "pred_clusters": len(prediction_boxes),
            "matched_clusters": len(cluster_matches),
            "gt_peduncles": len(gt_masks), "pred_peduncles": len(prediction_masks),
            "matched_peduncles": len(peduncle_matches),
        }
        rows.append(row)
        elapsed_times.append(elapsed_ms)
        totals["cluster_tp"] += len(cluster_matches)
        totals["cluster_fp"] += len(prediction_boxes) - len(cluster_matches)
        totals["cluster_fn"] += len(gt_boxes) - len(cluster_matches)
        totals["peduncle_tp"] += len(peduncle_matches)
        totals["peduncle_fp"] += len(prediction_masks) - len(peduncle_matches)
        totals["peduncle_fn"] += len(gt_masks) - len(peduncle_matches)
        if image_index % sample_stride == 0 and image_index // sample_stride < args.samples:
            cv2.imwrite(str(sample_dir / f"{image_path.stem}.jpg"), draw_pipeline(image, detections))
        if (image_index + 1) % 25 == 0 or image_index + 1 == len(images):
            print(f"Evaluated {image_index + 1}/{len(images)} images")

    cluster_metrics = scores(totals["cluster_tp"], totals["cluster_fp"], totals["cluster_fn"])
    peduncle_metrics = scores(totals["peduncle_tp"], totals["peduncle_fp"], totals["peduncle_fn"])
    warm_times = elapsed_times[1:] if len(elapsed_times) > 1 else elapsed_times
    summary = {
        "cluster_model": str(args.cluster_model.resolve()),
        "peduncle_model": str(args.peduncle_model.resolve()),
        "dataset": str(args.data.resolve()),
        "split": args.split,
        "images": len(rows),
        "thresholds": {
            "cluster_confidence": args.cluster_conf,
            "peduncle_confidence": args.peduncle_conf,
            "matching_iou": args.iou,
        },
        "cluster_box_metrics": cluster_metrics,
        "peduncle_mask_metrics": peduncle_metrics,
        "runtime_ms": {
            "first_image": elapsed_times[0] if elapsed_times else 0.0,
            "mean_excluding_warmup": statistics.fmean(warm_times) if warm_times else 0.0,
            "median_excluding_warmup": statistics.median(warm_times) if warm_times else 0.0,
        },
        "note": "Peduncle mask metrics include upstream cluster misses and use detector-derived ROIs.",
    }
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
