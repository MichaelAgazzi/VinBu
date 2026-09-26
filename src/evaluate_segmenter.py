"""Evaluate a YOLO segmentation checkpoint and save mask prediction samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common import IMAGE_SUFFIXES, require_ultralytics, select_device, write_resolved_dataset_yaml
from src.config import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR


DEFAULT_MODEL = MODELS_DIR / "grape_peduncle_roi_yolo11s_seg_best.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "canopies_peduncle" / "roi" / "dataset.yaml"


def split_images(data_yaml: Path, split: str) -> list[Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_dir = root / config[split]
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--conf", type=float, default=0.001,
        help="Low confidence floor used for AP computation (Ultralytics default: 0.001)",
    )
    parser.add_argument("--sample-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--name", default="grape_peduncle_roi_test")
    args = parser.parse_args()
    if not args.model.is_file() or not args.data.is_file():
        raise SystemExit("Model or dataset not found; build and train the peduncle pipeline first")
    YOLO = require_ultralytics()
    device = select_device(args.device)
    model = YOLO(str(args.model))
    output = RESULTS_DIR / "evaluation" / args.name
    runtime_data = write_resolved_dataset_yaml(args.data, output / "resolved_dataset.yaml")
    metrics = model.val(
        data=str(runtime_data), split=args.split, conf=args.conf, imgsz=args.imgsz,
        device=device, project=str((RESULTS_DIR / "evaluation").resolve()), name=args.name,
        exist_ok=True, plots=True, verbose=True,
    )
    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    class_metrics = {}
    for metric_name, metric in (("box", metrics.box), ("mask", metrics.seg)):
        for metric_index, class_id in enumerate(metric.ap_class_index):
            class_id = int(class_id)
            best_index = int(np.argmax(metric.f1_curve[metric_index]))
            class_metrics.setdefault(str(names[class_id]), {})[metric_name] = {
                "precision": float(metric.p[metric_index]),
                "recall": float(metric.r[metric_index]),
                "f1": float(metric.f1[metric_index]),
                "best_f1_confidence": float(metric.px[best_index]),
                "best_class_f1": float(metric.f1_curve[metric_index][best_index]),
                "mAP50": float(metric.ap50[metric_index]),
                "mAP50-95": float(metric.ap[metric_index]),
            }
    summary = {
        "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "split": args.split, "confidence_threshold": args.conf,
        "box_precision": float(metrics.box.mp), "box_recall": float(metrics.box.mr),
        "box_mAP50": float(metrics.box.map50), "box_mAP50-95": float(metrics.box.map),
        "mask_precision": float(metrics.seg.mp), "mask_recall": float(metrics.seg.mr),
        "mask_mAP50": float(metrics.seg.map50), "mask_mAP50-95": float(metrics.seg.map),
        "per_class": class_metrics,
    }
    sample_dir = output / "prediction_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    images = split_images(args.data, args.split)
    if args.samples > 0 and images:
        step = max(1, len(images) // args.samples)
        selected = images[::step][:args.samples]
        predictions = model.predict(
            selected, conf=args.sample_conf, imgsz=args.imgsz, device=device,
            retina_masks=True, verbose=False,
        )
        for image_path, result in zip(selected, predictions):
            cv2.imwrite(str(sample_dir / f"{image_path.stem}_prediction.jpg"), result.plot())
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
