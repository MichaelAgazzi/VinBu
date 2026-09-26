"""Select a peduncle-mask confidence threshold on validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device
from src.config import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR


DEFAULT_MODEL = MODELS_DIR / "grape_peduncle_roi_yolo11s_seg_best.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "canopies_peduncle" / "roi" / "dataset.yaml"


def split_paths(data_yaml: Path, split: str) -> tuple[list[Path], Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_dir = root / config[split]
    label_dir = root / config[split].replace("images", "labels")
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    return images, label_dir


def ground_truth_masks(
    label_path: Path, shape: tuple[int, int], class_id: int | None = None,
) -> list[np.ndarray]:
    height, width = shape
    masks = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        if class_id is not None and int(parts[0]) != class_id:
            continue
        points = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points = np.rint(points * [width, height]).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask.astype(bool))
    return masks


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def match_masks(ground_truth, predictions, iou_threshold: float) -> int:
    pairs = sorted(
        (
            (mask_iou(gt, prediction), gt_index, prediction_index)
            for gt_index, gt in enumerate(ground_truth)
            for prediction_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_gt, matched_predictions = set(), set()
    for score, gt_index, prediction_index in pairs:
        if score < iou_threshold:
            break
        if gt_index not in matched_gt and prediction_index not in matched_predictions:
            matched_gt.add(gt_index)
            matched_predictions.add(prediction_index)
    return len(matched_gt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--class-name", default="peduncle",
        help="Dataset class to calibrate (resolved from names in dataset.yaml)",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=[
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70,
    ])
    parser.add_argument("--name", default="grape_peduncle_roi_val")
    args = parser.parse_args()
    if not args.model.is_file() or not args.data.is_file():
        raise SystemExit("Model or dataset is missing")
    unique_thresholds = sorted(set(args.thresholds))
    if args.split == "test" and len(unique_thresholds) != 1:
        raise SystemExit(
            "Refusing to tune multiple thresholds on test. Select one threshold on validation first."
        )
    YOLO = require_ultralytics()
    model = YOLO(str(args.model))
    device = select_device(args.device)
    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    raw_names = data_config.get("names", {})
    names = (
        {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, dict)
        else dict(enumerate(raw_names))
    )
    matching_ids = [class_id for class_id, name in names.items() if name == args.class_name]
    if len(matching_ids) != 1:
        raise SystemExit(
            f"Class {args.class_name!r} is not uniquely defined in dataset names: {names}"
        )
    class_id = matching_ids[0]
    images, label_dir = split_paths(args.data, args.split)
    prediction_floor = min(args.thresholds)
    cached = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        result = model.predict(
            image, conf=prediction_floor, imgsz=args.imgsz, device=device,
            retina_masks=True, verbose=False,
        )[0]
        predictions = []
        if result.masks is not None and result.boxes is not None:
            for index, box in enumerate(result.boxes):
                if int(box.cls[0]) != class_id:
                    continue
                mask = result.masks.data[index].cpu().numpy()
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                predictions.append((float(box.conf[0]), mask > 0.5))
        cached.append((
            ground_truth_masks(
                label_dir / f"{image_path.stem}.txt", (height, width), class_id=class_id,
            ),
            predictions,
        ))
    rows = []
    for threshold in unique_thresholds:
        tp = fp = fn = 0
        for ground_truth, predictions in cached:
            selected = [mask for confidence, mask in predictions if confidence >= threshold]
            matched = match_masks(ground_truth, selected, args.iou)
            tp += matched
            fp += len(selected) - matched
            fn += len(ground_truth) - matched
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    best = max(rows, key=lambda row: (row["f1"], row["recall"]))
    output = RESULTS_DIR / "threshold_sweeps" / args.name
    output.mkdir(parents=True, exist_ok=True)
    with (output / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    point_key = "operating_point" if len(rows) == 1 else "best"
    summary = {
        "model": str(args.model.resolve()), "dataset": str(args.data.resolve()),
        "evaluated_split": args.split, "mask_iou": args.iou,
        "class_name": args.class_name, "class_id": class_id,
        point_key: best, "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
