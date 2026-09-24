"""Build a leakage-safe VINEPICs + WGISD + CANOPIES YOLO dataset.

The original downloads are never modified. Images are hard-linked when possible
and copied only as a fallback. VINEPICs keeps its existing session split, WGISD
keeps its official test set, and CANOPIES is split by complete capture sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src.config import PROJECT_ROOT, RANDOM_SEED


VINEPICS_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
WGISD_DIR = PROJECT_ROOT / "data" / "external" / "wgisd"
CANOPIES_DIR = PROJECT_ROOT / "data" / "external" / "canopies" / "data_0001"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "combined_dataset"


def materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_yolo_lines(lines: list[str], source: Path) -> int:
    count = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid one-class YOLO row at {source}:{line_number}: {line}")
        values = [float(value) for value in parts[1:]]
        if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Invalid normalized box at {source}:{line_number}: {line}")
        count += 1
    return count


def clip_yolo_lines(lines: list[str]) -> tuple[list[str], int]:
    """Clip normalized source boxes to the image boundary."""
    output: list[str] = []
    clipped = 0
    for line in lines:
        if not line.strip():
            continue
        class_id, xc_text, yc_text, width_text, height_text = line.split()
        xc, yc, width, height = map(float, (xc_text, yc_text, width_text, height_text))
        x1, y1 = max(0.0, xc - width / 2), max(0.0, yc - height / 2)
        x2, y2 = min(1.0, xc + width / 2), min(1.0, yc + height / 2)
        if (x1, y1, x2, y2) != (xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2):
            clipped += 1
        if x2 <= x1 or y2 <= y1:
            continue
        output.append(
            f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
            f"{x2 - x1:.6f} {y2 - y1:.6f}"
        )
    return output, clipped


def write_item(
    output: Path,
    split: str,
    name: str,
    image_source: Path,
    label_lines: list[str],
    source_dataset: str,
    original_name: str,
    sequence: str,
    manifest: list[dict],
    counters: Counter,
) -> None:
    image_name = f"{name}{image_source.suffix.lower()}"
    target_image = output / "images" / split / image_name
    target_label = output / "labels" / split / f"{name}.txt"
    link_mode = materialize(image_source, target_image)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    target_label.write_text("\n".join(line.strip() for line in label_lines if line.strip()), encoding="utf-8")
    box_count = validate_yolo_lines(label_lines, target_label)
    counters[f"{source_dataset}:{split}:images"] += 1
    counters[f"{source_dataset}:{split}:boxes"] += box_count
    counters[f"materialization:{link_mode}"] += 1
    manifest.append(
        {
            "source": source_dataset,
            "original_name": original_name,
            "sequence": sequence,
            "split": split,
            "prepared_image": image_name,
            "boxes": box_count,
        }
    )


def add_vinepics(output: Path, manifest: list[dict], counters: Counter) -> None:
    for split in ("train", "val", "test"):
        for image_path in sorted((VINEPICS_DIR / "images" / split).iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = VINEPICS_DIR / "labels" / split / f"{image_path.stem}.txt"
            lines = label_path.read_text(encoding="utf-8").splitlines()
            sequence = image_path.stem.split("_rgb", 1)[0]
            write_item(
                output, split, f"vinepics_{image_path.stem}", image_path, lines,
                "vinepics", image_path.name, sequence, manifest, counters,
            )


def wgisd_validation_ids(train_ids: list[str], seed: int) -> set[str]:
    by_variety: dict[str, list[str]] = defaultdict(list)
    for image_id in train_ids:
        by_variety[image_id.split("_", 1)[0]].append(image_id)
    validation: set[str] = set()
    for offset, variety in enumerate(sorted(by_variety)):
        members = sorted(by_variety[variety])
        random.Random(seed + offset).shuffle(members)
        validation.update(members[: max(1, round(len(members) * 0.15))])
    return validation


def add_wgisd(output: Path, manifest: list[dict], counters: Counter, seed: int) -> None:
    train_ids = (WGISD_DIR / "train.txt").read_text(encoding="utf-8").split()
    test_ids = (WGISD_DIR / "test.txt").read_text(encoding="utf-8").split()
    val_ids = wgisd_validation_ids(train_ids, seed)
    assignments = [(item, "val" if item in val_ids else "train") for item in train_ids]
    assignments.extend((item, "wgisd_test") for item in test_ids)
    for image_id, split in assignments:
        image_path = WGISD_DIR / "data" / f"{image_id}.jpg"
        label_path = WGISD_DIR / "data" / f"{image_id}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Incomplete WGISD pair for {image_id}")
        lines, clipped = clip_yolo_lines(label_path.read_text(encoding="utf-8").splitlines())
        counters["wgisd:clipped_boxes"] += clipped
        write_item(
            output, split, f"wgisd_{image_id}", image_path, lines,
            "wgisd", image_path.name, image_id.split("_", 1)[0], manifest, counters,
        )


def canopies_sequence_assignments(sequences: list[str], seed: int) -> dict[str, str]:
    shuffled = sorted(sequences)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * 0.20))
    n_val = max(1, round(len(shuffled) * 0.20))
    result = {}
    for index, sequence in enumerate(shuffled):
        if index < n_test:
            result[sequence] = "canopies_test"
        elif index < n_test + n_val:
            result[sequence] = "val"
        else:
            result[sequence] = "train"
    return result


def polygon_boxes(entry: dict, width: int, height: int) -> list[str]:
    lines = []
    for region in entry.get("regions", []):
        if region.get("region_attributes", {}).get("type") != "cluster":
            continue
        shape = region.get("shape_attributes", {})
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if shape.get("name") != "polygon" or len(xs) < 3 or len(xs) != len(ys):
            continue
        x1 = max(0.0, min(map(float, xs)))
        y1 = max(0.0, min(map(float, ys)))
        x2 = min(float(width), max(map(float, xs)) + 1.0)
        y2 = min(float(height), max(map(float, ys)) + 1.0)
        if x2 <= x1 or y2 <= y1:
            continue
        xc = (x1 + x2) / 2.0 / width
        yc = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def add_canopies(output: Path, manifest: list[dict], counters: Counter, seed: int) -> dict[str, str]:
    annotations = json.loads((CANOPIES_DIR / "labels.json").read_text(encoding="utf-8"))
    images = {path.name: path for path in (CANOPIES_DIR / "frames").rglob("*.jpg")}
    sequences = sorted({entry["filename"].split("_")[1] for entry in annotations.values()})
    assignments = canopies_sequence_assignments(sequences, seed)
    for entry in sorted(annotations.values(), key=lambda item: item["filename"]):
        image_path = images.get(entry["filename"])
        if image_path is None:
            raise FileNotFoundError(f"CANOPIES image missing: {entry['filename']}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable CANOPIES image: {image_path}")
        height, width = image.shape[:2]
        sequence = entry["filename"].split("_")[1]
        split = assignments[sequence]
        lines = polygon_boxes(entry, width, height)
        write_item(
            output, split, f"canopies_{image_path.stem}", image_path, lines,
            "canopies", image_path.name, sequence, manifest, counters,
        )
    return assignments


def write_yaml(path: Path, test_folder: str = "images/test") -> None:
    path.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"test: {test_folder}\n\n"
        "names:\n  0: grape_cluster\n",
        encoding="utf-8",
    )


def build(output: Path, seed: int, force: bool) -> dict:
    required = [VINEPICS_DIR / "dataset.yaml", WGISD_DIR / "train.txt", CANOPIES_DIR / "labels.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source datasets: " + ", ".join(missing))
    if output.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output}. Use --force to rebuild it.")
        if output.resolve() == PROJECT_ROOT.resolve() or not output.resolve().is_relative_to(PROJECT_ROOT):
            raise ValueError(f"Refusing to remove unsafe output path: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest: list[dict] = []
    counters: Counter = Counter()
    add_vinepics(output, manifest, counters)
    add_wgisd(output, manifest, counters, seed)
    canopies_assignments = add_canopies(output, manifest, counters, seed)
    write_yaml(output / "dataset.yaml")
    write_yaml(output / "wgisd_test.yaml", "images/wgisd_test")
    write_yaml(output / "canopies_test.yaml", "images/canopies_test")

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "seed": seed,
        "training_sources": ["VINEPICs", "Embrapa WGISD", "CANOPIES GBPD"],
        "primary_test": "Original VINEPICs test split only (directly comparable to baseline)",
        "wgisd_split": "Official test retained; official train stratified 85/15 by variety",
        "canopies_split": "Complete capture sequences: 60% train, 20% val, 20% held-out test",
        "canopies_sequence_assignments": dict(sorted(canopies_assignments.items())),
        "counts": dict(sorted(counters.items())),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(args.output, args.seed, args.force)


if __name__ == "__main__":
    main()
