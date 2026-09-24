"""Evaluate trained grape-cluster weights and generate error-analysis samples."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml
from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device, write_resolved_dataset_yaml
from src.config import (
    DATASET_DIR,
    DATASET_YAML,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_TRAINED_MODEL,
    EVALUATION_DIR,
)


def load_ground_truth(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = map(float, parts)
        boxes.append([
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])
    return boxes


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(gt_boxes: list[list[float]], predictions: list[list[float]], threshold: float = 0.5):
    pairs = sorted(
        ((iou(gt, pred), gi, pi) for gi, gt in enumerate(gt_boxes)
         for pi, pred in enumerate(predictions)),
        reverse=True,
    )
    matched_gt, matched_pred = set(), set()
    for score, gt_index, pred_index in pairs:
        if score < threshold:
            break
        if gt_index not in matched_gt and pred_index not in matched_pred:
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
    return matched_gt, matched_pred


def draw_error_sample(image, gt_boxes, predictions, matched_gt, matched_pred, path: Path) -> None:
    canvas = image.copy()
    for index, box in enumerate(gt_boxes):
        colour = (40, 190, 40) if index in matched_gt else (30, 30, 235)
        label = "TP ground truth" if index in matched_gt else "FN missed cluster"
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(canvas, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, colour, 2, cv2.LINE_AA)
    for index, box in enumerate(predictions):
        if index in matched_pred:
            continue
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 150, 255), 3)
        cv2.putText(canvas, "FP prediction", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 150, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def split_directories(data_yaml: Path, split: str) -> tuple[Path, Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    configured = config.get(split)
    if not isinstance(configured, str):
        raise ValueError(f"Dataset YAML has no string path for split '{split}': {data_yaml}")
    image_dir = root / configured
    parts = list(Path(configured).parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"Cannot derive label directory from '{configured}'") from exc
    return image_dir, root.joinpath(*parts)


def custom_error_analysis(
    model, data_yaml: Path, split: str, conf: float, imgsz: int, device,
    output: Path, max_samples: int,
) -> dict:
    image_dir, label_dir = split_directories(data_yaml, split)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    counts = {"tp": 0, "fp": 0, "fn": 0}
    saved = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for image_path in images:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt_boxes = load_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        result = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        predictions = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
        matched_gt, matched_pred = match_boxes(gt_boxes, predictions)
        tp, fp, fn = len(matched_gt), len(predictions) - len(matched_pred), len(gt_boxes) - len(matched_gt)
        counts["tp"] += tp
        counts["fp"] += fp
        counts["fn"] += fn
        category = None
        if fn and saved["false_negative"] < max_samples:
            category = "false_negative"
        elif fp and saved["false_positive"] < max_samples:
            category = "false_positive"
        elif tp and saved["true_positive"] < max_samples:
            category = "true_positive"
        if category:
            saved[category] += 1
            draw_error_sample(
                image, gt_boxes, predictions, matched_gt, matched_pred,
                output / "samples" / category / image_path.name,
            )
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    return {**counts, "precision_at_conf_iou50": precision, "grape_cluster_recall_at_conf_iou50": recall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--data", type=Path, default=DATASET_YAML)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-error-samples", type=int, default=5)
    parser.add_argument("--name", help="Output subdirectory name (defaults to the split name)")
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Trained model not found: {args.model}")
    if not args.data.is_file():
        raise SystemExit("Dataset not prepared. Run: python -m src.dataset_utils")

    YOLO = require_ultralytics()
    device = select_device(args.device)
    model = YOLO(str(args.model))
    run_name = args.name or args.split
    run_dir = EVALUATION_DIR / run_name
    runtime_data = write_resolved_dataset_yaml(args.data, run_dir / "resolved_dataset.yaml")
    metrics = model.val(
        data=str(runtime_data), split=args.split, conf=args.conf, imgsz=args.imgsz,
        device=device, project=str(EVALUATION_DIR.resolve()), name=run_name,
        exist_ok=True, plots=True, verbose=True,
    )
    summary = {
        "model": str(args.model.resolve()),
        "split": args.split,
        "confidence_threshold": args.conf,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    summary["error_analysis"] = custom_error_analysis(
        model, args.data, args.split, args.conf, args.imgsz, device, run_dir,
        args.max_error_samples,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Keep common Ultralytics plot names easy to find at results/evaluation root too.
    for name in (
        "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png",
        "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png",
    ):
        source = run_dir / name
        if source.is_file():
            shutil.copy2(source, EVALUATION_DIR / name)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
