"""Run grape-cluster inference on an image, directory, or video."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.common import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    draw_detections,
    model_is_grape_detector,
    predictions_to_detections,
    require_ultralytics,
    result_to_detections,
    select_device,
    timed_predict,
    write_detection_exports,
)
from src.tiled_inference import predict_tiled
from src.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TRAINED_MODEL,
    DETECTIONS_DIR,
)


def detect_image(model, image_path: Path, output_dir: Path, args) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if args.tiled:
        started = time.perf_counter()
        predictions = predict_tiled(
            model, image, conf=args.conf, imgsz=args.imgsz, device=args.device_resolved,
            tile_size=args.tile_size, overlap=args.tile_overlap, nms_iou=args.iou,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = predictions_to_detections(predictions, image_path.name)
    else:
        result, elapsed_ms = timed_predict(
            model, image, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=args.device_resolved,
        )
        detections = result_to_detections(result, image_path.name)
    annotated = draw_detections(image, detections)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_detected{image_path.suffix}"), annotated)
    write_detection_exports(detections, output_dir / f"{image_path.stem}_detections")
    print(f"{image_path.name}: {len(detections)} detections in {elapsed_ms:.1f} ms")
    return detections


def detect_video(model, video_path: Path, output_dir: Path, args) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    all_detections: list[dict] = []
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.tiled:
            predictions = predict_tiled(
                model, frame, conf=args.conf, imgsz=args.imgsz, device=args.device_resolved,
                tile_size=args.tile_size, overlap=args.tile_overlap, nms_iou=args.iou,
            )
            detections = predictions_to_detections(
                predictions, video_path.name, frame_id
            )
        else:
            result, _ = timed_predict(
                model, frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                device=args.device_resolved,
            )
            detections = result_to_detections(result, video_path.name, frame_id)
        writer.write(draw_detections(frame, detections))
        all_detections.extend(detections)
        frame_id += 1
    capture.release()
    writer.release()
    write_detection_exports(all_detections, output_dir / f"{video_path.stem}_detections")
    print(f"{video_path.name}: {len(all_detections)} detections across {frame_id} frames")
    return all_detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--output", type=Path, default=DETECTIONS_DIR)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiled", action="store_true", help="Use overlapping tiles for small objects")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--allow-non-grape-model", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source does not exist: {args.source}")
    if not args.model.is_file():
        raise SystemExit(
            f"Trained model not found: {args.model}\n"
            "Train first with: python -m src.train_detector"
        )
    args.device_resolved = select_device(args.device)
    YOLO = require_ultralytics()
    model = YOLO(str(args.model))
    if not model_is_grape_detector(model) and not args.allow_non_grape_model:
        raise SystemExit(
            "The supplied weights do not contain a 'grape_cluster' class. "
            "Use trained project weights, or explicitly pass --allow-non-grape-model for diagnostics."
        )

    if args.source.is_dir():
        sources = sorted(path for path in args.source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not sources:
            raise SystemExit(f"No supported images found in: {args.source}")
        for source in sources:
            detect_image(model, source, args.output, args)
    elif args.source.suffix.lower() in IMAGE_SUFFIXES:
        detect_image(model, args.source, args.output, args)
    elif args.source.suffix.lower() in VIDEO_SUFFIXES:
        detect_video(model, args.source, args.output, args)
    else:
        raise SystemExit(f"Unsupported source type: {args.source.suffix}")


if __name__ == "__main__":
    main()
