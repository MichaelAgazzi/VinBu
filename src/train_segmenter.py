"""Train an Ultralytics YOLO11 segmentation model for grape peduncles."""

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
from src.config import MODELS_DIR, PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR


DEFAULT_DATA = PROJECT_ROOT / "data" / "canopies_peduncle" / "roi" / "dataset.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--name", default="grape_peduncle_roi_yolo11s_seg")
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--mask-ratio", type=int, default=2)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--mosaic", type=float, default=0.50)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument(
        "--save-period", type=int, default=-1,
        help="Save an intermediate checkpoint every N epochs (-1 disables it)",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        raise SystemExit("Peduncle dataset missing. Run: python -m src.build_peduncle_dataset --force")
    if args.epochs < 1 or args.imgsz < 64 or args.batch == 0:
        raise SystemExit("Invalid training dimensions: epochs/imgsz must be positive and batch non-zero")

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
        **vars(args), "data": str(args.data.resolve()), "device_resolved": device,
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8",
    )
    cache: bool | str = False if args.cache == "false" else args.cache
    print(f"Training segmentation on device: {device}")
    model = YOLO(args.model)
    model.train(
        data=str(runtime_data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=device, workers=args.workers, seed=args.seed,
        deterministic=True, cache=cache, project=str((RESULTS_DIR / "training").resolve()),
        name=args.name, exist_ok=True, plots=True, verbose=True,
        save_period=args.save_period,
        # Preserve thin peduncle masks at twice the default label resolution.
        mask_ratio=args.mask_ratio, overlap_mask=True,
        # Moderate geometry and strong illumination changes match vineyard data.
        degrees=5.0, translate=0.08, scale=0.30, shear=2.0, perspective=0.0003,
        fliplr=0.50, flipud=0.05, mosaic=args.mosaic, close_mosaic=args.close_mosaic,
        hsv_h=0.012, hsv_s=0.55, hsv_v=0.45,
        multi_scale=args.multi_scale, copy_paste=args.copy_paste,
        optimizer=args.optimizer, lr0=args.lr0, cos_lr=args.cos_lr,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
    )
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Training ended without best weights: {best}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / f"{args.name}_best.pt"
    shutil.copy2(best, destination)
    from ultralytics.utils.torch_utils import strip_optimizer

    strip_optimizer(destination)
    print(f"Best segmenter copied to: {destination}")


if __name__ == "__main__":
    main()
