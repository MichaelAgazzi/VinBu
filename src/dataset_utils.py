"""Convert VINEPICs COCO boxes to a leakage-safe Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.config import (
    CLASS_NAMES,
    COCO_CATEGORY_TO_YOLO,
    DATASET_DIR,
    DEFAULT_SESSION_SPLIT,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_ANNOTATIONS_FILE,
    RAW_IMAGES_DIR,
    SESSION_METADATA,
)


def load_coco(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, ...]:
    x, y, box_width, box_height = map(float, bbox)
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + box_width))
    y2 = max(0.0, min(float(height), y + box_height))
    clipped_width, clipped_height = x2 - x1, y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        raise ValueError(f"Invalid box after clipping: {bbox}")
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        clipped_width / width,
        clipped_height / height,
    )


def _session(file_name: str) -> str:
    return Path(file_name.replace("\\", "/")).parts[0]


def build_session_split(sessions: set[str], seed: int) -> dict[str, str]:
    """Use the curated default split; fall back to deterministic grouped allocation."""
    if seed == RANDOM_SEED and sessions.issubset(DEFAULT_SESSION_SPLIT):
        return {session: DEFAULT_SESSION_SPLIT[session] for session in sessions}

    grouped: dict[str, list[str]] = defaultdict(list)
    for session in sessions:
        grouped[SESSION_METADATA.get(session, {}).get("variety", "unknown")].append(session)
    output: dict[str, str] = {}
    for offset, (_, members) in enumerate(sorted(grouped.items())):
        members = sorted(members)
        random.Random(seed + offset).shuffle(members)
        n = len(members)
        n_test = max(1, round(n * 0.10)) if n >= 3 else 0
        n_val = max(1, round(n * 0.20)) if n >= 2 else 0
        for index, session in enumerate(members):
            if index < n_test:
                split = "test"
            elif index < n_test + n_val:
                split = "val"
            else:
                split = "train"
            output[session] = split
    return output


def _materialize_image(source: Path, destination: Path, link_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    modes = [link_mode] if link_mode != "auto" else ["hardlink", "symlink", "copy"]
    for mode in modes:
        try:
            if mode == "hardlink":
                os.link(source, destination)
            elif mode == "symlink":
                destination.symlink_to(source.resolve())
            elif mode == "copy":
                shutil.copy2(source, destination)
            else:
                raise ValueError(f"Unknown link mode: {mode}")
            return mode
        except (OSError, NotImplementedError):
            if link_mode != "auto":
                raise
    raise RuntimeError(f"Could not materialize {source}")


def convert_and_split(
    annotations_file: Path = RAW_ANNOTATIONS_FILE,
    images_dir: Path = RAW_IMAGES_DIR,
    output_dir: Path = DATASET_DIR,
    seed: int = RANDOM_SEED,
    link_mode: str = "auto",
    force_labels: bool = False,
) -> dict:
    coco = load_coco(annotations_file)
    categories = {int(item["id"]): item["name"] for item in coco.get("categories", [])}
    if categories.get(1) != "bunch":
        raise ValueError(f"Expected COCO category 1='bunch', found {categories!r}")

    images = {int(item["id"]): item for item in coco["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    sessions = {_session(item["file_name"]) for item in images.values()}
    session_split = build_session_split(sessions, seed)

    stats = Counter()
    split_images = Counter()
    split_annotations = Counter()
    materialization = Counter()
    invalid: list[dict] = []

    for image_id, info in images.items():
        session = _session(info["file_name"])
        split = session_split[session]
        source = images_dir / Path(info["file_name"])
        if not source.is_file():
            stats["missing_images"] += 1
            continue
        flat_name = f"{session}_{Path(info['file_name']).name}"
        target_image = output_dir / "images" / split / flat_name
        materialization[_materialize_image(source, target_image, link_mode)] += 1

        target_label = output_dir / "labels" / split / f"{Path(flat_name).stem}.txt"
        target_label.parent.mkdir(parents=True, exist_ok=True)
        if target_label.exists() and not force_labels:
            stats["existing_labels"] += 1
        else:
            lines: list[str] = []
            for annotation in annotations.get(image_id, []):
                class_id = COCO_CATEGORY_TO_YOLO.get(int(annotation["category_id"]))
                if class_id is None:
                    continue
                try:
                    box = coco_bbox_to_yolo(annotation["bbox"], info["width"], info["height"])
                except ValueError:
                    invalid.append({"annotation_id": annotation.get("id"), "bbox": annotation["bbox"]})
                    continue
                lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in box))
            target_label.write_text("\n".join(lines), encoding="utf-8")
            stats["written_labels"] += 1

        split_images[split] += 1
        split_annotations[split] += len(annotations.get(image_id, []))
        stats["images"] += 1
        stats["annotations"] += len(annotations.get(image_id, []))
        if not annotations.get(image_id):
            stats["empty_images"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "# Generated from VINEPICs COCO annotations\n"
        "path: .\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "names:\n  0: grape_cluster\n",
        encoding="utf-8",
    )
    summary = {
        "source_annotations": (
            annotations_file.resolve().relative_to(PROJECT_ROOT).as_posix()
            if annotations_file.resolve().is_relative_to(PROJECT_ROOT)
            else str(annotations_file.resolve())
        ),
        "seed": seed,
        "split_strategy": "capture-session (prevents adjacent-frame leakage)",
        "session_assignments": dict(sorted(session_split.items())),
        "split_image_counts": dict(split_images),
        "split_annotation_counts": dict(split_annotations),
        "stats": dict(stats),
        "materialization": dict(materialization),
        "invalid_annotations": invalid,
        "classes": CLASS_NAMES,
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=RAW_ANNOTATIONS_FILE)
    parser.add_argument("--images", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "symlink", "copy"), default="auto")
    parser.add_argument("--force-labels", action="store_true")
    args = parser.parse_args()
    convert_and_split(args.annotations, args.images, args.output, args.seed, args.link_mode, args.force_labels)


if __name__ == "__main__":
    main()
