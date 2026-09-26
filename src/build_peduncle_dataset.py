"""Build leakage-safe CANOPIES segmentation datasets.

Three complementary datasets are generated from the original VIA polygons:

* ``full`` keeps complete images and two classes (cluster and peduncle);
* ``roi`` crops the area around the top of each ground-truth cluster and keeps
  peduncle masks.  This makes the thin target substantially larger without
  leaking capture sequences across train, validation, and test.
* ``multiscale`` combines complete training images with two-class instance
  crops, while validation and test contain only untouched complete images.
  It lets a direct model learn small peduncles without changing its benchmark.

The downloaded CANOPIES files are treated as read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.config import PROJECT_ROOT, RANDOM_SEED


CANOPIES_DIR = PROJECT_ROOT / "data" / "external" / "canopies" / "data_0001"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "canopies_peduncle"
CLASS_IDS = {"cluster": 0, "peduncle": 1}


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def sequence_from_filename(filename: str) -> str:
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CANOPIES filename: {filename}")
    return parts[1]


def assign_sequences(sequences: list[str], seed: int) -> dict[str, str]:
    """Assign complete capture sequences to a deterministic 60/20/20 split."""
    shuffled = sorted(set(sequences))
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        raise ValueError("At least three capture sequences are required")
    n_test = max(1, round(len(shuffled) * 0.20))
    n_val = max(1, round(len(shuffled) * 0.20))
    return {
        sequence: ("test" if index < n_test else "val" if index < n_test + n_val else "train")
        for index, sequence in enumerate(shuffled)
    }


def region_polygon(region: dict, width: int, height: int) -> list[tuple[float, float]] | None:
    shape = region.get("shape_attributes", {})
    xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
    if shape.get("name") != "polygon" or len(xs) < 3 or len(xs) != len(ys):
        return None
    points = [
        (min(max(float(x), 0.0), width - 1.0), min(max(float(y), 0.0), height - 1.0))
        for x, y in zip(xs, ys)
    ]
    deduplicated: list[tuple[float, float]] = []
    for point in points:
        if not deduplicated or point != deduplicated[-1]:
            deduplicated.append(point)
    if len(set(deduplicated)) < 3 or polygon_area(deduplicated) < 1.0:
        return None
    return deduplicated


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )) / 2.0


def yolo_segment_line(class_id: int, points: list[tuple[float, float]], width: int, height: int) -> str:
    coordinates = []
    for x, y in points:
        coordinates.extend((min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0)))
    return f"{class_id} " + " ".join(f"{value:.6f}" for value in coordinates)


def polygon_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def roi_from_cluster_box(
    box: tuple[float, float, float, float], image_width: int, image_height: int,
    min_side: int = 160, scale: float = 1.55, above_fraction: float = 0.48,
) -> tuple[int, int, int, int]:
    """Return a square ROI covering the cluster top and its likely peduncle."""
    x1, y1, x2, y2 = box
    box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    side = int(math.ceil(max(min_side, scale * box_width, 1.05 * box_height)))
    side = min(side, image_width, image_height)
    centre_x = (x1 + x2) / 2.0
    left = int(round(centre_x - side / 2.0))
    top = int(round(y1 - above_fraction * side))
    left = min(max(left, 0), image_width - side)
    top = min(max(top, 0), image_height - side)
    return left, top, left + side, top + side


def _clip_edge(points, inside, intersect):
    if not points:
        return []
    output = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous, previous_inside = current, current_inside
    return output


def clip_polygon_to_rect(
    points: list[tuple[float, float]], x1: float, y1: float, x2: float, y2: float,
) -> list[tuple[float, float]]:
    """Clip a polygon to an axis-aligned rectangle (Sutherland-Hodgman)."""
    def vertical(boundary):
        def intersection(first, second):
            delta = second[0] - first[0]
            ratio = 0.0 if abs(delta) < 1e-9 else (boundary - first[0]) / delta
            return boundary, first[1] + ratio * (second[1] - first[1])
        return intersection

    def horizontal(boundary):
        def intersection(first, second):
            delta = second[1] - first[1]
            ratio = 0.0 if abs(delta) < 1e-9 else (boundary - first[1]) / delta
            return first[0] + ratio * (second[0] - first[0]), boundary
        return intersection

    clipped = list(points)
    clipped = _clip_edge(clipped, lambda p: p[0] >= x1, vertical(x1))
    clipped = _clip_edge(clipped, lambda p: p[0] <= x2, vertical(x2))
    clipped = _clip_edge(clipped, lambda p: p[1] >= y1, horizontal(y1))
    clipped = _clip_edge(clipped, lambda p: p[1] <= y2, horizontal(y2))
    deduplicated = []
    for point in clipped:
        if not deduplicated or abs(point[0] - deduplicated[-1][0]) > 1e-6 or abs(point[1] - deduplicated[-1][1]) > 1e-6:
            deduplicated.append(point)
    return deduplicated


def entry_polygons(entry: dict, width: int, height: int) -> dict[str, list[list[tuple[float, float]]]]:
    output = {name: [] for name in CLASS_IDS}
    for region in entry.get("regions", []):
        region_type = region.get("region_attributes", {}).get("type")
        if region_type not in output:
            continue
        polygon = region_polygon(region, width, height)
        if polygon is not None:
            output[region_type].append(polygon)
    return output


def write_yaml(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name_rows = "\n".join(f"  {index}: {name}" for index, name in enumerate(names))
    path.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n"
        + name_rows + "\n",
        encoding="utf-8",
    )


def build(output: Path, seed: int, force: bool, min_visible: float = 0.20) -> dict:
    labels_path = CANOPIES_DIR / "labels.json"
    frames_dir = CANOPIES_DIR / "frames"
    if not labels_path.is_file() or not frames_dir.is_dir():
        raise FileNotFoundError(f"CANOPIES source is incomplete: {CANOPIES_DIR}")
    resolved_output = output.resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    if resolved_output == data_root or not resolved_output.is_relative_to(data_root):
        raise ValueError(f"Output must be a generated child of data/: {output}")
    if output.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output}. Use --force to rebuild it.")
        shutil.rmtree(output)

    annotations = json.loads(labels_path.read_text(encoding="utf-8"))
    entries = sorted(annotations.values(), key=lambda item: item["filename"])
    images = {path.name: path for path in frames_dir.rglob("*.jpg")}
    assignments = assign_sequences([sequence_from_filename(item["filename"]) for item in entries], seed)
    counters: Counter = Counter()
    manifest: list[dict] = []

    for entry in entries:
        filename = entry["filename"]
        image_path = images.get(filename)
        if image_path is None:
            raise FileNotFoundError(f"Image referenced by labels is missing: {filename}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        sequence = sequence_from_filename(filename)
        split = assignments[sequence]
        polygons = entry_polygons(entry, width, height)
        stem = image_path.stem

        full_image = output / "full" / "images" / split / image_path.name
        mode = materialize(image_path, full_image)
        full_lines = [
            yolo_segment_line(CLASS_IDS[class_name], polygon, width, height)
            for class_name in ("cluster", "peduncle") for polygon in polygons[class_name]
        ]
        full_label = output / "full" / "labels" / split / f"{stem}.txt"
        full_label.parent.mkdir(parents=True, exist_ok=True)
        full_label.write_text("\n".join(full_lines), encoding="utf-8")
        counters[f"full:{split}:images"] += 1
        counters[f"materialization:{mode}"] += 1
        for class_name in CLASS_IDS:
            counters[f"full:{split}:{class_name}"] += len(polygons[class_name])

        # The multiscale benchmark keeps validation/test exactly equal to the
        # full-image dataset. Training additionally receives instance crops.
        multiscale_image = output / "multiscale" / "images" / split / f"full_{image_path.name}"
        multiscale_mode = materialize(image_path, multiscale_image)
        multiscale_label = output / "multiscale" / "labels" / split / f"full_{stem}.txt"
        multiscale_label.parent.mkdir(parents=True, exist_ok=True)
        multiscale_label.write_text("\n".join(full_lines), encoding="utf-8")
        counters[f"multiscale:{split}:images"] += 1
        counters[f"materialization:{multiscale_mode}"] += 1
        for class_name in CLASS_IDS:
            counters[f"multiscale:{split}:{class_name}"] += len(polygons[class_name])

        for cluster_index, cluster in enumerate(polygons["cluster"], start=1):
            roi = roi_from_cluster_box(polygon_bounds(cluster), width, height)
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            crop = image[roi_y1:roi_y2, roi_x1:roi_x2]
            crop_height, crop_width = crop.shape[:2]
            roi_lines = []
            visible_peduncles = 0
            for peduncle in polygons["peduncle"]:
                original_area = polygon_area(peduncle)
                clipped = clip_polygon_to_rect(peduncle, roi_x1, roi_y1, roi_x2 - 1, roi_y2 - 1)
                if len(clipped) < 3 or polygon_area(clipped) < 1.0:
                    continue
                if polygon_area(clipped) / max(original_area, 1e-9) < min_visible:
                    continue
                local = [(x - roi_x1, y - roi_y1) for x, y in clipped]
                roi_lines.append(yolo_segment_line(0, local, crop_width, crop_height))
                visible_peduncles += 1
            roi_stem = f"{stem}_c{cluster_index:02d}"
            roi_image = output / "roi" / "images" / split / f"{roi_stem}.jpg"
            roi_image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(roi_image), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Could not write ROI: {roi_image}")
            roi_label = output / "roi" / "labels" / split / f"{roi_stem}.txt"
            roi_label.parent.mkdir(parents=True, exist_ok=True)
            roi_label.write_text("\n".join(roi_lines), encoding="utf-8")
            counters[f"roi:{split}:images"] += 1
            counters[f"roi:{split}:peduncle"] += visible_peduncles
            if not roi_lines:
                counters[f"roi:{split}:empty"] += 1

            if split == "train":
                multiscale_lines = []
                multiscale_instances = Counter()
                for class_name in ("cluster", "peduncle"):
                    for polygon in polygons[class_name]:
                        original_area = polygon_area(polygon)
                        clipped = clip_polygon_to_rect(
                            polygon, roi_x1, roi_y1, roi_x2 - 1, roi_y2 - 1,
                        )
                        if len(clipped) < 3 or polygon_area(clipped) < 1.0:
                            continue
                        if polygon_area(clipped) / max(original_area, 1e-9) < min_visible:
                            continue
                        local = [(x - roi_x1, y - roi_y1) for x, y in clipped]
                        multiscale_lines.append(
                            yolo_segment_line(CLASS_IDS[class_name], local, crop_width, crop_height)
                        )
                        multiscale_instances[class_name] += 1
                multiscale_crop = (
                    output / "multiscale" / "images" / "train" / f"crop_{roi_stem}.jpg"
                )
                crop_mode = materialize(roi_image, multiscale_crop)
                multiscale_crop_label = (
                    output / "multiscale" / "labels" / "train" / f"crop_{roi_stem}.txt"
                )
                multiscale_crop_label.parent.mkdir(parents=True, exist_ok=True)
                multiscale_crop_label.write_text("\n".join(multiscale_lines), encoding="utf-8")
                counters["multiscale:train:images"] += 1
                counters[f"materialization:{crop_mode}"] += 1
                for class_name in CLASS_IDS:
                    counters[f"multiscale:train:{class_name}"] += multiscale_instances[class_name]
            manifest.append({
                "prepared": roi_image.name, "source": filename, "sequence": sequence,
                "split": split, "cluster_index": cluster_index,
                "x1": roi_x1, "y1": roi_y1, "x2": roi_x2, "y2": roi_y2,
                "peduncles": visible_peduncles,
            })

    write_yaml(output / "full" / "dataset.yaml", ["grape_cluster", "peduncle"])
    write_yaml(output / "roi" / "dataset.yaml", ["peduncle"])
    write_yaml(output / "multiscale" / "dataset.yaml", ["grape_cluster", "peduncle"])
    with (output / "roi_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "prepared", "source", "sequence", "split", "cluster_index",
            "x1", "y1", "x2", "y2", "peduncles",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "source": str(CANOPIES_DIR.resolve()), "seed": seed,
        "sequence_assignments": dict(sorted(assignments.items())),
        "unknown_or_invalid_regions": sum(
            1 for item in entries for region in item.get("regions", [])
            if region.get("region_attributes", {}).get("type") not in CLASS_IDS
        ),
        "counts": dict(sorted(counters.items())),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--min-visible", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.seed, args.force, args.min_visible), indent=2))


if __name__ == "__main__":
    main()
