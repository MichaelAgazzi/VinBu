"""Validate YOLO polygon labels and render deterministic overlay samples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common import IMAGE_SUFFIXES
from src.config import PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR


DEFAULT_DATA = PROJECT_ROOT / "data" / "canopies_peduncle" / "roi" / "dataset.yaml"


def dataset_paths(data_yaml: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, config


def label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    try:
        reverse_index = parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"Cannot map image path to labels: {image_dir}") from exc
    parts[len(parts) - reverse_index - 1] = "labels"
    return Path(*parts)


def read_polygons(path: Path, class_count: int) -> tuple[list[tuple[int, np.ndarray]], list[str]]:
    polygons, errors = [], []
    if not path.is_file():
        return polygons, ["missing label"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            errors.append(f"line {line_number}: expected class plus >=3 xy pairs")
            continue
        try:
            class_id = int(parts[0])
            coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue
        if not 0 <= class_id < class_count:
            errors.append(f"line {line_number}: class {class_id} outside [0,{class_count - 1}]")
        if not np.all((coordinates >= 0.0) & (coordinates <= 1.0)):
            errors.append(f"line {line_number}: coordinates outside [0,1]")
        if abs(cv2.contourArea(coordinates)) < 1e-8:
            errors.append(f"line {line_number}: zero-area polygon")
        polygons.append((class_id, coordinates))
    return polygons, errors


def draw_overlay(image: np.ndarray, polygons, names) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()
    height, width = image.shape[:2]
    colours = [(50, 190, 50), (255, 190, 0), (220, 70, 220)]
    for class_id, normalized in polygons:
        points = np.rint(normalized * [width, height]).astype(np.int32)
        colour = colours[class_id % len(colours)]
        cv2.fillPoly(overlay, [points], colour)
        cv2.polylines(canvas, [points], True, colour, 2, cv2.LINE_AA)
        anchor = tuple(points[0])
        cv2.putText(canvas, str(names.get(class_id, class_id)), anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return cv2.addWeighted(canvas, 0.72, overlay, 0.28, 0)


def validate(data_yaml: Path, output: Path, samples: int, seed: int) -> dict:
    root, config = dataset_paths(data_yaml)
    raw_names = config.get("names", {})
    names = {int(key): value for key, value in raw_names.items()} if isinstance(raw_names, dict) else dict(enumerate(raw_names))
    counts: Counter = Counter()
    errors: list[dict] = []
    candidates = []
    for split in ("train", "val", "test"):
        image_dir = root / config[split]
        label_dir = label_dir_from_image_dir(image_dir)
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        counts[f"{split}:images"] = len(images)
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        for stem in sorted(image_stems - label_stems):
            errors.append({"split": split, "file": stem, "error": "missing label"})
        for stem in sorted(label_stems - image_stems):
            errors.append({"split": split, "file": stem, "error": "orphan label"})
        for image_path in images:
            polygons, line_errors = read_polygons(label_dir / f"{image_path.stem}.txt", len(names))
            counts[f"{split}:instances"] += len(polygons)
            if not polygons:
                counts[f"{split}:empty"] += 1
            for class_id, _ in polygons:
                counts[f"{split}:class:{names.get(class_id, class_id)}"] += 1
            for error in line_errors:
                errors.append({"split": split, "file": image_path.name, "error": error})
            if polygons:
                candidates.append((split, image_path, polygons))
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for index, (split, image_path, polygons) in enumerate(candidates[:samples], start=1):
        image = cv2.imread(str(image_path))
        overlay = draw_overlay(image, polygons, names)
        cv2.imwrite(str(output / f"sample_{index:02d}_{split}_{image_path.stem}.jpg"), overlay)
    report = {
        "dataset": str(data_yaml.resolve()), "names": names,
        "counts": dict(sorted(counts.items())), "error_count": len(errors), "errors": errors[:100],
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "peduncle_dataset_checks")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    report = validate(args.data, args.output, args.samples, args.seed)
    print(json.dumps(report, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
