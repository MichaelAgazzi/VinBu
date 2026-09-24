"""Train a one-class Ultralytics YOLO11 grape-cluster detector."""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common import require_ultralytics, select_device, write_resolved_dataset_yaml
from src.config import (
    DATASET_YAML,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL,
    DEFAULT_PATIENCE,
    MODELS_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET_YAML)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1, ...")
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--name", default="grape_yolo11n")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--small-object-augmentations", action="store_true",
        help="Use crop-friendly geometry plus stronger vineyard photometric augmentation",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit("dataset.yaml not found. Run: python -m src.dataset_utils")
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")

    YOLO = require_ultralytics()
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)

    run_dir = RESULTS_DIR / "training" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_data = write_resolved_dataset_yaml(args.data, run_dir / "resolved_dataset.yaml")
    config = {
        **vars(args),
        "data": str(args.data.resolve()),
        "device_resolved": device,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    augmentation_kwargs = {}
    if args.small_object_augmentations:
        augmentation_kwargs = {
            # Preserve the enlarged scale created by native-resolution crops.
            "scale": 0.25,
            "translate": 0.08,
            "degrees": 3.0,
            "mosaic": 0.50,
            "close_mosaic": 10,
            # Vineyard illumination and colour shifts. Offline crop generation
            # additionally adds shadows, backlight, contrast, and blur.
            "hsv_h": 0.012,
            "hsv_s": 0.60,
            "hsv_v": 0.50,
            "bgr": 0.05,
            "fliplr": 0.50,
        }

    model.train(
        data=str(runtime_data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        cache=args.cache,
        project=str((RESULTS_DIR / "training").resolve()),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
        **augmentation_kwargs,
    )

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training ended but best weights were not found at {best}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / f"{args.name}_best.pt"
    shutil.copy2(best, destination)
    print(f"Best model copied to: {destination}")
    print("Evaluate it with: python -m src.evaluate --model " + str(destination))


if __name__ == "__main__":
    main()
