"""Evaluate direct two-class cluster/peduncle inference on held-out full images."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2

from src.common import require_ultralytics, select_device
from src.config import (
    DEFAULT_DIRECT_CLUSTER_CONF_THRESHOLD,
    DEFAULT_DIRECT_IMGSZ,
    DEFAULT_DIRECT_PEDUNCLE_CONF_THRESHOLD,
    DIRECT_PEDUNCLE_SEGMENTATION_MODEL,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from src.direct_peduncle_pipeline import analyze_direct
from src.evaluate_peduncle_pipeline import (
    box_iou,
    greedy_matches,
    ground_truth,
    scores,
    split_paths,
)
from src.peduncle_pipeline import draw_pipeline
from src.sweep_segmentation_thresholds import mask_iou


DEFAULT_MODEL = DIRECT_PEDUNCLE_SEGMENTATION_MODEL
DEFAULT_DATA = PROJECT_ROOT / "data" / "canopies_peduncle" / "full" / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--cluster-conf", type=float, default=DEFAULT_DIRECT_CLUSTER_CONF_THRESHOLD)
    parser.add_argument("--peduncle-conf", type=float, default=DEFAULT_DIRECT_PEDUNCLE_CONF_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_DIRECT_IMGSZ)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_DIR / "peduncle_pipeline" / "direct_heldout_evaluation",
    )
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    for path in (args.model, args.data):
        if not path.is_file():
            raise SystemExit(f"Required file not found: {path}")

    YOLO = require_ultralytics()
    device = select_device(args.device)
    model = YOLO(str(args.model))
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
        detections, elapsed_ms = analyze_direct(
            image, model, device, args.cluster_conf, args.peduncle_conf, args.imgsz,
        )
        prediction_boxes = [item["cluster_box"] for item in detections]
        prediction_masks = [item["_mask"] for item in detections if item["_mask"] is not None]
        cluster_matches = greedy_matches(gt_boxes, prediction_boxes, box_iou, args.iou)
        peduncle_matches = greedy_matches(gt_masks, prediction_masks, mask_iou, args.iou)
        row = {
            "image": image_path.name, "elapsed_ms": round(elapsed_ms, 3),
            "gt_clusters": len(gt_boxes), "pred_clusters": len(prediction_boxes),
            "matched_clusters": len(cluster_matches),
            "gt_peduncles": len(gt_masks), "pred_peduncles": len(prediction_masks),
            "matched_peduncles": len(peduncle_matches),
        }
        rows.append(row)
        elapsed_times.append(elapsed_ms)
        for prefix, matches, ground_truth_items, predictions in (
            ("cluster", cluster_matches, gt_boxes, prediction_boxes),
            ("peduncle", peduncle_matches, gt_masks, prediction_masks),
        ):
            totals[f"{prefix}_tp"] += len(matches)
            totals[f"{prefix}_fp"] += len(predictions) - len(matches)
            totals[f"{prefix}_fn"] += len(ground_truth_items) - len(matches)
        if image_index % sample_stride == 0 and image_index // sample_stride < args.samples:
            cv2.imwrite(str(sample_dir / f"{image_path.stem}.jpg"), draw_pipeline(image, detections))
        if (image_index + 1) % 25 == 0 or image_index + 1 == len(images):
            print(f"Evaluated {image_index + 1}/{len(images)} images")

    cluster_metrics = scores(totals["cluster_tp"], totals["cluster_fp"], totals["cluster_fn"])
    peduncle_metrics = scores(totals["peduncle_tp"], totals["peduncle_fp"], totals["peduncle_fn"])
    warm_times = elapsed_times[1:] if len(elapsed_times) > 1 else elapsed_times
    summary = {
        "model": str(args.model.resolve()), "dataset": str(args.data.resolve()),
        "split": args.split, "images": len(rows), "imgsz": args.imgsz,
        "thresholds": {
            "cluster_confidence": args.cluster_conf,
            "peduncle_confidence": args.peduncle_conf,
            "matching_iou": args.iou,
        },
        "cluster_box_metrics": cluster_metrics,
        "associated_peduncle_mask_metrics": peduncle_metrics,
        "runtime_ms": {
            "first_image": elapsed_times[0] if elapsed_times else 0.0,
            "mean_excluding_warmup": statistics.fmean(warm_times) if warm_times else 0.0,
            "median_excluding_warmup": statistics.median(warm_times) if warm_times else 0.0,
        },
        "note": "Peduncle metrics include cluster misses and association errors.",
    }
    if rows:
        with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
