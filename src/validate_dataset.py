"""Validate YOLO labels and create dataset-quality plots and overlays."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import IMAGE_SUFFIXES
from src.config import DATASET_CHECKS_DIR, DATASET_DIR, RANDOM_SEED


def read_yolo_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes, errors = [], []
    if not path.exists():
        return boxes, ["missing label"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 values")
            continue
        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue
        if class_id != 0:
            errors.append(f"line {line_number}: unexpected class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in (xc, yc, width, height)):
            errors.append(f"line {line_number}: normalized value outside [0,1]")
        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: non-positive box size")
        if xc - width / 2 < -1e-6 or xc + width / 2 > 1 + 1e-6:
            errors.append(f"line {line_number}: box exceeds horizontal bounds")
        if yc - height / 2 < -1e-6 or yc + height / 2 > 1 + 1e-6:
            errors.append(f"line {line_number}: box exceeds vertical bounds")
        boxes.append((class_id, xc, yc, width, height))
    return boxes, errors


def render_overlay(image_path: Path, boxes: list[tuple], output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    height, width = image.shape[:2]
    for index, (_, xc, yc, box_width, box_height) in enumerate(boxes, start=1):
        x1 = int((xc - box_width / 2) * width)
        y1 = int((yc - box_height / 2) * height)
        x2 = int((xc + box_width / 2) * width)
        y2 = int((yc + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 210, 30), 3)
        cv2.putText(image, f"G{index:03d}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 210, 30), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def validate(dataset_dir: Path, output_dir: Path, samples: int, seed: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"splits": {}, "errors": [], "class_distribution": Counter()}
    cluster_counts: list[int] = []
    area_ratios: list[float] = []
    sample_pool: list[tuple[Path, list[tuple]]] = []

    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        split_boxes = 0
        empty = 0
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes, errors = read_yolo_label(label_path)
            if not boxes:
                empty += 1
            for class_id, _, _, width, height in boxes:
                report["class_distribution"][str(class_id)] += 1
                area_ratios.append(width * height)
            split_boxes += len(boxes)
            cluster_counts.append(len(boxes))
            sample_pool.append((image_path, boxes))
            for error in errors:
                report["errors"].append({"file": str(label_path), "error": error})
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": split_boxes,
            "empty_labels": empty,
            "missing_labels": sorted(image_stems - label_stems),
            "orphan_labels": sorted(label_stems - image_stems),
        }

    rng = random.Random(seed)
    chosen = rng.sample(sample_pool, min(samples, len(sample_pool)))
    for index, (image_path, boxes) in enumerate(chosen, start=1):
        render_overlay(image_path, boxes, output_dir / "overlays" / f"sample_{index:02d}_{image_path.name}")

    if cluster_counts:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(cluster_counts, bins=range(0, max(cluster_counts) + 2), color="#6f2da8", edgecolor="white")
        axes[0].set(title="Grape clusters per image", xlabel="Annotated clusters", ylabel="Images")
        axes[1].hist(np.asarray(area_ratios) * 100, bins=30, color="#4a8f3c", edgecolor="white")
        axes[1].set(title="Bounding-box area distribution", xlabel="Percent of image area", ylabel="Boxes")
        fig.tight_layout()
        fig.savefig(output_dir / "dataset_distributions.png", dpi=180)
        plt.close(fig)

    report["class_distribution"] = dict(report["class_distribution"])
    report["summary"] = {
        "images": sum(item["images"] for item in report["splits"].values()),
        "boxes": sum(item["boxes"] for item in report["splits"].values()),
        "invalid_entries": len(report["errors"]),
        "mean_clusters_per_image": round(float(np.mean(cluster_counts)), 3) if cluster_counts else 0,
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_CHECKS_DIR)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if not (args.dataset / "dataset.yaml").is_file():
        raise SystemExit("YOLO dataset not found. Run: python -m src.dataset_utils")
    validate(args.dataset, args.output, args.samples, args.seed)


if __name__ == "__main__":
    main()
