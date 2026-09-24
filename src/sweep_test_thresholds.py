"""Sweep confidence thresholds on a labelled split, optionally using tiled inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device
from src.config import DEFAULT_IMGSZ, DEFAULT_TRAINED_MODEL, PROJECT_ROOT
from src.evaluate import load_ground_truth, split_directories
from src.plot_test_predictions import greedy_match
from src.tiled_inference import predict_tiled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/yolo_dataset/dataset.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--max-conf", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=19)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    image_dir, label_dir = split_directories(args.data, args.split)
    model = require_ultralytics()(str(args.model))
    device = select_device(args.device)
    samples = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        if args.tiled:
            predictions = predict_tiled(
                model, image, conf=args.min_conf, imgsz=args.imgsz, device=device,
                tile_size=args.tile_size, overlap=args.tile_overlap,
            )
        else:
            result = model.predict(
                image, conf=args.min_conf, imgsz=args.imgsz, device=device, verbose=False
            )[0]
            predictions = [] if result.boxes is None else [
                {"box": box.xyxy[0].tolist(), "confidence": float(box.conf[0])}
                for box in result.boxes
            ]
        samples.append((gt, predictions))

    rows = []
    for threshold in np.linspace(args.min_conf, args.max_conf, args.steps):
        tp = fp = fn = 0
        for gt, all_predictions in samples:
            predictions = [p for p in all_predictions if p["confidence"] >= threshold]
            predictions.sort(key=lambda item: item["confidence"], reverse=True)
            matched_gt, matched_pred = greedy_match(gt, predictions, 0.5)
            tp += len(matched_gt)
            fp += len(predictions) - len(matched_pred)
            fn += len(gt) - len(matched_gt)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "confidence": float(threshold), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })

    output = PROJECT_ROOT / "results/threshold_sweeps" / args.name
    output.mkdir(parents=True, exist_ok=True)
    with (output / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["f1"])
    summary = {"model": str(args.model.resolve()), "tiled": args.tiled, "best_f1": best}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, colour in (("precision", "#264653"), ("recall", "#E76F51"), ("f1", "#2A9D8F")):
        axes[0].plot([r["confidence"] for r in rows], [r[metric] for r in rows],
                     marker="o", markersize=3, label=metric.capitalize(), color=colour)
    axes[0].axvline(best["confidence"], color="black", linestyle="--", alpha=0.6)
    axes[0].set(xlabel="Confidence threshold", ylabel="Score", ylim=(0, 1),
                title=f"Threshold sweep ({'tiled' if args.tiled else 'full image'})")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot([r["recall"] for r in rows], [r["precision"] for r in rows],
                 marker="o", color="#7353BA")
    axes[1].scatter([best["recall"]], [best["precision"]], color="black", zorder=3,
                    label=f"best F1 @ {best['confidence']:.2f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
                title="Precision-recall operating points")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "threshold_sweep.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
