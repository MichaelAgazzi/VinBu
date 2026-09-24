"""Build a VINEPICs full-image/crop mixture for small-object training.

Only the training split is expanded. Validation and test remain byte-identical
to the original session-separated dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.config import DATASET_DIR, PROJECT_ROOT, RANDOM_SEED
from src.tiled_inference import tile_origins


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "small_object_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def load_pixel_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, xc, yc, bw, bh = line.split()
        if class_id != "0":
            raise ValueError(f"Expected one class in {label_path}: {line}")
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))
        boxes.append([
            (xc - bw / 2) * width, (yc - bh / 2) * height,
            (xc + bw / 2) * width, (yc + bh / 2) * height,
        ])
    return boxes


def boxes_for_crop(
    boxes: list[list[float]], x0: int, y0: int, crop_width: int, crop_height: int,
    canvas_size: int, min_visible: float = 0.5,
) -> list[str]:
    """Clip boxes to a crop, retaining centre-owned and sufficiently visible objects."""
    lines: list[str] = []
    x_end, y_end = x0 + crop_width, y0 + crop_height
    for x1, y1, x2, y2 in boxes:
        centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
        if not (x0 <= centre_x < x_end and y0 <= centre_y < y_end):
            continue
        clipped_x1, clipped_y1 = max(x1, x0), max(y1, y0)
        clipped_x2, clipped_y2 = min(x2, x_end), min(y2, y_end)
        original_area = max(0, x2 - x1) * max(0, y2 - y1)
        visible_area = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
        if original_area <= 0 or visible_area / original_area < min_visible:
            continue
        left, top = clipped_x1 - x0, clipped_y1 - y0
        right, bottom = clipped_x2 - x0, clipped_y2 - y0
        xc, yc = (left + right) / 2 / canvas_size, (top + bottom) / 2 / canvas_size
        bw, bh = (right - left) / canvas_size, (bottom - top) / canvas_size
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def crop_score(lines: list[str], boxes: list[list[float]], image_area: float) -> float:
    """Prefer crops rich in objects, especially boxes below 0.5% of the image."""
    if not lines:
        return 0.0
    small = sum(
        1 for x1, y1, x2, y2 in boxes
        if ((x2 - x1) * (y2 - y1)) / image_area < 0.005
    )
    return len(lines) + min(len(lines), small) * 1.5


def augment_photometric(image: np.ndarray, key: str, seed: int) -> tuple[np.ndarray, str]:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    mode = digest[0] % 4
    if mode == 0:
        return cv2.convertScaleAbs(image, alpha=1.25, beta=-18), "contrast"
    if mode == 1:
        gamma = 0.68
        table = np.asarray([((value / 255) ** gamma) * 255 for value in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table), "backlight"
    if mode == 2:
        rng = random.Random(int.from_bytes(digest[:8], "little"))
        overlay = image.copy()
        height, width = image.shape[:2]
        start = rng.randint(width // 5, width // 2)
        polygon = np.array([[0, 0], [start, 0], [width, height], [width // 2, height]], np.int32)
        cv2.fillPoly(overlay, [polygon], (0, 0, 0))
        return cv2.addWeighted(image, 0.60, overlay, 0.40, 0), "shadow"
    return cv2.GaussianBlur(image, (5, 5), 0), "blur"


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_dataset(
    source: Path, output: Path, tile_size: int, overlap: float, crops_per_image: int,
    full_copies: int, seed: int,
) -> dict:
    if output.resolve() == source.resolve() or output.resolve().parent != (PROJECT_ROOT / "data").resolve():
        raise ValueError(f"Output must be a generated child of project data/: {output}")
    if output.exists():
        shutil.rmtree(output)
    manifest: list[dict] = []
    counts: Counter = Counter()

    for split in ("val", "test"):
        for image_path in sorted((source / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            materialize(image_path, output / "images" / split / image_path.name)
            materialize(label_path, output / "labels" / split / label_path.name)
            counts[f"{split}_images"] += 1

    for image_path in sorted((source / "images" / "train").iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = source / "labels" / "train" / f"{image_path.stem}.txt"
        label_lines = label_path.read_text(encoding="utf-8").splitlines()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        boxes = load_pixel_boxes(label_path, width, height)

        for copy_index in range(full_copies):
            stem = f"full{copy_index}_{image_path.stem}"
            destination = output / "images" / "train" / f"{stem}{image_path.suffix.lower()}"
            mode = materialize(image_path, destination)
            write_label(output / "labels" / "train" / f"{stem}.txt", label_lines)
            manifest.append({"prepared": destination.name, "source": image_path.name,
                             "kind": "full", "augmentation": "online", "boxes": len(boxes)})
            counts[f"full_{mode}"] += 1

        candidates = []
        for y0 in tile_origins(height, tile_size, overlap):
            for x0 in tile_origins(width, tile_size, overlap):
                crop_width, crop_height = min(tile_size, width - x0), min(tile_size, height - y0)
                lines = boxes_for_crop(boxes, x0, y0, crop_width, crop_height, tile_size)
                score = crop_score(lines, boxes, width * height)
                candidates.append((score, len(lines), -y0, -x0, x0, y0, crop_width, crop_height, lines))
        candidates.sort(reverse=True)
        for crop_index, candidate in enumerate(candidates[:crops_per_image]):
            _, _, _, _, x0, y0, crop_width, crop_height, lines = candidate
            crop = image[y0:y0 + crop_height, x0:x0 + crop_width]
            canvas = np.full((tile_size, tile_size, 3), 114, dtype=np.uint8)
            canvas[:crop_height, :crop_width] = crop
            augmentation = "none"
            if crop_index % 2 == 1:
                canvas, augmentation = augment_photometric(canvas, f"{image_path.stem}:{crop_index}", seed)
            stem = f"crop{crop_index}_{image_path.stem}_x{x0}_y{y0}"
            destination = output / "images" / "train" / f"{stem}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write {destination}")
            write_label(output / "labels" / "train" / f"{stem}.txt", lines)
            manifest.append({"prepared": destination.name, "source": image_path.name,
                             "kind": "crop", "augmentation": augmentation, "boxes": len(lines),
                             "x": x0, "y": y0})
            counts["crop_images"] += 1
            counts["crop_boxes"] += len(lines)

    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.yaml").write_text(
        "# VINEPICs full-image/crop small-object training dataset\n"
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: grape_cluster\n", encoding="utf-8",
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["prepared", "source", "kind", "augmentation", "boxes", "x", "y"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = {
        "source": str(source.resolve()), "output": str(output.resolve()), "tile_size": tile_size,
        "overlap": overlap, "crops_per_image": crops_per_image, "full_copies": full_copies,
        "seed": seed, **counts,
    }
    summary["train_images"] = len(manifest)
    summary["full_fraction"] = sum(1 for row in manifest if row["kind"] == "full") / len(manifest)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--full-copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    print(json.dumps(build_dataset(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
