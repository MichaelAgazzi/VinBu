"""Create a compact comparison table and plot from completed evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import EVALUATION_DIR


EXPERIMENTS = {
    "Baseline VINEPICs": {
        "VINEPICs": "test",
        "WGISD": "baseline_on_wgisd_test",
        "CANOPIES": "baseline_on_canopies_test",
    },
    "Combined": {
        "VINEPICs": "combined_on_vinepics_test",
        "WGISD": "combined_on_wgisd_test",
        "CANOPIES": "combined_on_canopies_test",
    },
    "Combined + VINEPICs FT": {
        "VINEPICs": "finetuned_on_vinepics_test",
        "WGISD": "finetuned_on_wgisd_test",
        "CANOPIES": "finetuned_on_canopies_test",
    },
}
METRICS = ("precision", "recall", "mAP50", "mAP50-95")


def main() -> None:
    rows = []
    for model_name, datasets in EXPERIMENTS.items():
        for dataset_name, run_name in datasets.items():
            path = EVALUATION_DIR / run_name / "metrics.json"
            if not path.is_file():
                raise SystemExit(f"Missing completed evaluation: {path}")
            metrics = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {"model": model_name, "dataset": dataset_name, **{key: metrics[key] for key in METRICS}}
            )

    csv_path = EVALUATION_DIR / "model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "dataset", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    (EVALUATION_DIR / "model_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    datasets = ("VINEPICs", "WGISD", "CANOPIES")
    models = tuple(EXPERIMENTS)
    x = np.arange(len(datasets))
    width = 0.24
    colours = ("#7353BA", "#2A9D8F", "#E76F51")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, metric in zip(axes.flat, METRICS):
        for index, (model_name, colour) in enumerate(zip(models, colours)):
            values = [
                next(row[metric] for row in rows if row["model"] == model_name and row["dataset"] == dataset)
                for dataset in datasets
            ]
            bars = ax.bar(x + (index - 1) * width, values, width, label=model_name, color=colour)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, rotation=90, padding=2)
        ax.set_title(metric)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x, datasets)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=3, frameon=False,
    )
    fig.suptitle("YOLO11n grape-cluster evaluation by held-out dataset", y=0.985, fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(EVALUATION_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    target_runs = {
        "YOLO11n baseline": "test",
        "YOLO11n + external data": "combined_on_vinepics_test",
        "YOLO11n external + FT": "finetuned_on_vinepics_test",
        "YOLO11s VINEPICs": "grape_yolo11s_vinepics_test",
        "YOLO11s crop-aware (full)": "grape_yolo11s_small_objects_vinepics_test",
    }
    target_rows = []
    for model_name, run_name in target_runs.items():
        metrics = json.loads(
            (EVALUATION_DIR / run_name / "metrics.json").read_text(encoding="utf-8")
        )
        target_rows.append({"model": model_name, **{key: metrics[key] for key in METRICS}})
    target_csv = EVALUATION_DIR / "vinepics_model_comparison.csv"
    with target_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", *METRICS))
        writer.writeheader()
        writer.writerows(target_rows)
    x = np.arange(len(METRICS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, row in enumerate(target_rows):
        bars = ax.bar(
            x + (index - (len(target_rows) - 1) / 2) * width,
            [row[metric] for metric in METRICS], width,
            label=row["model"],
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, rotation=90, padding=2)
    ax.set_xticks(x, METRICS)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score")
    ax.set_title("Held-out VINEPICs test: model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2)
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "vinepics_model_comparison.png", dpi=180)
    plt.close(fig)

    strategy_runs = {
        "YOLO11n full\nconf .25": "baseline_vinepics_test",
        "YOLO11s full\nconf .25": "yolo11s_vinepics_test",
        "Crop-aware full\nconf .25": "yolo11s_small_objects_vinepics_test",
        "Crop+tiled balanced\nconf .50": "yolo11s_small_objects_tiled640_calibrated",
        "Crop+tiled high recall\nconf .25": "yolo11s_small_objects_tiled640",
    }
    strategy_rows = []
    predictions_root = EVALUATION_DIR.parent / "test_predictions"
    for strategy, run_name in strategy_runs.items():
        summary = json.loads(
            (predictions_root / run_name / "summary.json").read_text(encoding="utf-8")
        )
        precision, recall = summary["precision"], summary["recall"]
        strategy_rows.append({
            "strategy": strategy.replace("\n", " "), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall),
        })
    strategy_csv = EVALUATION_DIR / "small_object_strategy_comparison.csv"
    with strategy_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("strategy", "precision", "recall", "f1"))
        writer.writeheader()
        writer.writerows(strategy_rows)
    x = np.arange(len(strategy_rows))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 6))
    for offset, (metric, colour) in enumerate(
        (("precision", "#264653"), ("recall", "#E76F51"), ("f1", "#2A9D8F"))
    ):
        bars = ax.bar(
            x + (offset - 1) * width, [row[metric] for row in strategy_rows], width,
            label=metric.capitalize(), color=colour,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=8, rotation=90, padding=2)
    ax.set_xticks(x, strategy_runs)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Score at IoU 0.50")
    ax.set_title("Held-out VINEPICs: small-object inference strategies")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / "small_object_strategy_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {csv_path}, {target_csv}, and comparison plots")


if __name__ == "__main__":
    main()
