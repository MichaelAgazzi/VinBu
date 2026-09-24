"""Calibrate and evaluate a two-model tiled ensemble on a labelled split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device
from src.config import PROJECT_ROOT
from src.evaluate import load_ground_truth, split_directories
from src.plot_test_predictions import (
    greedy_match,
    make_montage,
    make_plots,
    render_comparison,
)
from src.tiled_inference import classless_nms, predict_tiled


def score_samples(samples, conf_a: float, conf_b: float, nms_iou: float) -> dict:
    tp = fp = fn = 0
    for _, gt, predictions_a, predictions_b in samples:
        merged = classless_nms(
            [p for p in predictions_a if p["confidence"] >= conf_a]
            + [p for p in predictions_b if p["confidence"] >= conf_b],
            nms_iou,
        )
        merged.sort(key=lambda item: item["confidence"], reverse=True)
        matched_gt, matched_predictions = greedy_match(gt, merged, 0.50)
        tp += len(matched_gt)
        fp += len(merged) - len(matched_predictions)
        fn += len(gt) - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "conf_a": conf_a, "conf_b": conf_b, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--imgsz-a", type=int, default=640)
    parser.add_argument("--imgsz-b", type=int, default=960)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/yolo_dataset/dataset.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--prediction-conf", type=float, default=0.05)
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--max-conf", type=float, default=0.70)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--conf-a", type=float)
    parser.add_argument("--conf-b", type=float)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.split == "test" and (args.conf_a is None or args.conf_b is None):
        raise SystemExit("Test evaluation requires validation-selected --conf-a and --conf-b")

    image_dir, label_dir = split_directories(args.data, args.split)
    YOLO = require_ultralytics()
    model_a, model_b = YOLO(str(args.model_a)), YOLO(str(args.model_b))
    device = select_device(args.device)
    samples = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        gt = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        common = {
            "conf": args.prediction_conf, "device": device, "tile_size": args.tile_size,
            "overlap": args.tile_overlap, "nms_iou": args.nms_iou,
        }
        predictions_a = predict_tiled(model_a, image, imgsz=args.imgsz_a, **common)
        predictions_b = predict_tiled(model_b, image, imgsz=args.imgsz_b, **common)
        samples.append((image_path, gt, predictions_a, predictions_b))

    if args.conf_a is not None and args.conf_b is not None:
        best = score_samples(samples, args.conf_a, args.conf_b, args.nms_iou)
        rows = [best]
    else:
        rows = [
            score_samples(samples, float(conf_a), float(conf_b), args.nms_iou)
            for conf_a in np.linspace(args.min_conf, args.max_conf, args.steps)
            for conf_b in np.linspace(args.min_conf, args.max_conf, args.steps)
        ]
        best = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))

    output = PROJECT_ROOT / "results/ensemble" / args.name
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": args.split, "model_a": str(args.model_a.resolve()),
        "model_b": str(args.model_b.resolve()), "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap, "nms_iou": args.nms_iou,
        "prediction_conf": args.prediction_conf, "best": best,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rendered_dir = output / "images"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    gt_area_status: list[tuple[float, bool]] = []
    for image_path, gt, predictions_a, predictions_b in samples:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        merged = classless_nms(
            [p for p in predictions_a if p["confidence"] >= best["conf_a"]]
            + [p for p in predictions_b if p["confidence"] >= best["conf_b"]],
            args.nms_iou,
        )
        merged.sort(key=lambda item: item["confidence"], reverse=True)
        matched_gt, matched_predictions = greedy_match(gt, merged, 0.50)
        tp = len(matched_gt)
        fp = len(merged) - len(matched_predictions)
        fn = len(gt) - len(matched_gt)
        gt_areas = [
            ((box[2] - box[0]) * (box[3] - box[1])) / (width * height) for box in gt
        ]
        gt_area_status.extend((area, index in matched_gt) for index, area in enumerate(gt_areas))
        rendered_name = f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(
            str(rendered_dir / rendered_name),
            render_comparison(image, gt, merged, matched_gt, matched_predictions),
        )
        rows.append({
            "image": image_path.name, "rendered_name": rendered_name,
            "gt_boxes": len(gt), "predictions": len(merged), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "median_gt_area_percent": float(np.median(gt_areas) * 100) if gt_areas else 0.0,
        })
    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_plots(rows, gt_area_status, output, "YOLO11s+YOLO11m ensemble")
    make_montage(rows, rendered_dir, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
