"""Render every test prediction against ground truth and plot per-image errors."""

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
from src.config import DEFAULT_CONF_THRESHOLD, DEFAULT_IMGSZ, DEFAULT_TRAINED_MODEL, PROJECT_ROOT
from src.evaluate import iou, load_ground_truth, split_directories
from src.tiled_inference import predict_tiled


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "test_predictions"


def greedy_match(gt_boxes, predictions, threshold: float = 0.5):
    """Confidence-ordered one-to-one matching, returning matched index sets."""
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pred_index, prediction in enumerate(predictions):
        candidates = [
            (iou(gt, prediction["box"]), gt_index)
            for gt_index, gt in enumerate(gt_boxes)
            if gt_index not in matched_gt
        ]
        if not candidates:
            continue
        score, gt_index = max(candidates)
        if score >= threshold:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return matched_gt, matched_pred


def label(canvas, text: str, x: int, y: int, colour) -> None:
    cv2.putText(
        canvas, text, (max(2, x), max(18, y)), cv2.FONT_HERSHEY_SIMPLEX,
        0.47, colour, 1, cv2.LINE_AA,
    )


def render_comparison(image, gt_boxes, predictions, matched_gt, matched_pred):
    canvas = image.copy()
    for index, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        is_match = index in matched_gt
        colour = (40, 190, 40) if is_match else (30, 30, 235)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label(canvas, "GT-TP" if is_match else "FN", x1, y1 - 4, colour)
    for index, prediction in enumerate(predictions):
        x1, y1, x2, y2 = (int(round(value)) for value in prediction["box"])
        is_match = index in matched_pred
        colour = (235, 180, 30) if is_match else (0, 145, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        label(
            canvas, f"{'P-TP' if is_match else 'FP'} {prediction['confidence']:.2f}",
            x1, y2 + 16, colour,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 29), (25, 25, 25), -1)
    label(canvas, "GT-TP green | FN red | matched prediction cyan | FP orange", 8, 20, (255, 255, 255))
    return canvas


def make_plots(
    rows: list[dict], gt_area_status: list[tuple[float, bool]], output: Path, model_name: str
) -> None:
    ordered = sorted(rows, key=lambda row: (row["recall"], -row["gt_boxes"]))
    names = [Path(row["image"]).stem[-18:] for row in ordered]
    x = np.arange(len(ordered))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(x, [row["recall"] for row in ordered], color="#7353BA")
    axes[0, 0].set(title="Per-image recall (worst to best)", ylabel="Recall", ylim=(0, 1.05))
    axes[0, 0].set_xticks(x, names, rotation=90, fontsize=7)

    axes[0, 1].bar(x, [row["tp"] for row in ordered], label="TP", color="#2A9D8F")
    axes[0, 1].bar(
        x, [row["fn"] for row in ordered], bottom=[row["tp"] for row in ordered],
        label="FN", color="#E76F51",
    )
    axes[0, 1].set(title="Ground-truth outcomes per image", ylabel="Clusters")
    axes[0, 1].set_xticks(x, names, rotation=90, fontsize=7)
    axes[0, 1].legend(frameon=False)

    areas = np.asarray([area * 100 for area, _ in gt_area_status])
    matched = np.asarray([ok for _, ok in gt_area_status])
    bins = np.linspace(0, max(1.0, float(np.percentile(areas, 98))), 25)
    axes[1, 0].hist(areas[matched], bins=bins, alpha=0.75, label="Matched GT", color="#2A9D8F")
    axes[1, 0].hist(areas[~matched], bins=bins, alpha=0.75, label="Missed GT", color="#E76F51")
    axes[1, 0].set(title="Object size and detection outcome", xlabel="Box area (% of image)", ylabel="Clusters")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        [row["median_gt_area_percent"] for row in rows], [row["recall"] for row in rows],
        s=[max(25, row["gt_boxes"] * 6) for row in rows], alpha=0.75, color="#264653",
    )
    axes[1, 1].set(
        title="Recall vs median object size", xlabel="Median GT box area (% of image)",
        ylabel="Recall", ylim=(-0.03, 1.03),
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(f"VINEPICs held-out test: per-image {model_name} error analysis", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "per_image_error_analysis.png", dpi=180)
    plt.close(fig)


def make_montage(rows: list[dict], image_dir: Path, output: Path, count: int = 6) -> None:
    worst = sorted(rows, key=lambda row: (row["recall"], -row["fn"]))[:count]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, row in zip(axes.flat, worst):
        image = cv2.imread(str(image_dir / row["rendered_name"]))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(
            f"{row['image']}\nGT {row['gt_boxes']} | TP {row['tp']} | "
            f"FP {row['fp']} | FN {row['fn']} | R {row['recall']:.2f}", fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Worst held-out VINEPICs images at IoU 0.50", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "worst_images_montage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "yolo_dataset" / "dataset.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iou-match", type=float, default=0.50)
    parser.add_argument("--tiled", action="store_true", help="Use overlapping native-resolution tiles")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="baseline_vinepics_test")
    args = parser.parse_args()

    image_source, label_source = split_directories(args.data, args.split)
    output = DEFAULT_OUTPUT / args.name
    rendered_dir = output / "images"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    model = require_ultralytics()(str(args.model))
    device = select_device(args.device)
    rows: list[dict] = []
    gt_area_status: list[tuple[float, bool]] = []
    for image_path in sorted(path for path in image_source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt_boxes = load_ground_truth(label_source / f"{image_path.stem}.txt", width, height)
        if args.tiled:
            predictions = predict_tiled(
                model, image, conf=args.conf, imgsz=args.imgsz, device=device,
                tile_size=args.tile_size, overlap=args.tile_overlap, nms_iou=args.nms_iou,
            )
        else:
            result = model.predict(
                image, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False
            )[0]
            predictions = []
            if result.boxes is not None:
                for box in result.boxes:
                    predictions.append(
                        {
                            "box": [float(value) for value in box.xyxy[0].tolist()],
                            "confidence": float(box.conf[0]),
                        }
                    )
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
        matched_gt, matched_pred = greedy_match(gt_boxes, predictions, args.iou_match)
        tp, fp, fn = len(matched_gt), len(predictions) - len(matched_pred), len(gt_boxes) - len(matched_gt)
        gt_areas = [((box[2] - box[0]) * (box[3] - box[1])) / (width * height) for box in gt_boxes]
        gt_area_status.extend((area, index in matched_gt) for index, area in enumerate(gt_areas))
        rendered_name = f"{image_path.stem}_comparison.jpg"
        cv2.imwrite(
            str(rendered_dir / rendered_name),
            render_comparison(image, gt_boxes, predictions, matched_gt, matched_pred),
        )
        rows.append(
            {
                "image": image_path.name,
                "rendered_name": rendered_name,
                "gt_boxes": len(gt_boxes),
                "predictions": len(predictions),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "median_gt_area_percent": float(np.median(gt_areas) * 100) if gt_areas else 0.0,
            }
        )

    with (output / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": str(args.model.resolve()), "dataset": str(args.data.resolve()),
        "confidence": args.conf, "imgsz": args.imgsz, "iou_match": args.iou_match,
        "tiled": args.tiled, "tile_size": args.tile_size if args.tiled else None,
        "tile_overlap": args.tile_overlap if args.tiled else None,
        "nms_iou": args.nms_iou if args.tiled else None,
        "images": len(rows), "gt": sum(row["gt_boxes"] for row in rows),
        "tp": sum(row["tp"] for row in rows), "fp": sum(row["fp"] for row in rows),
        "fn": sum(row["fn"] for row in rows),
    }
    summary["precision"] = summary["tp"] / max(1, summary["tp"] + summary["fp"])
    summary["recall"] = summary["tp"] / max(1, summary["tp"] + summary["fn"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(rows, gt_area_status, output, args.model.stem)
    make_montage(rows, rendered_dir, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
