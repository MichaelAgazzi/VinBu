"""Gradio demo for two-stage grape-cluster and peduncle segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import require_ultralytics, select_device  # noqa: E402
from src.config import DEFAULT_PEDUNCLE_CONF_THRESHOLD, PEDUNCLE_SEGMENTATION_MODEL  # noqa: E402
from src.peduncle_pipeline import analyze_image, draw_pipeline, serializable  # noqa: E402


DEFAULT_CLUSTER_MODEL = PROJECT_ROOT / "models" / "grape_yolo11n_vinepics_wgisd_canopies_best.pt"


@lru_cache(maxsize=1)
def load_models(cluster_path: str, peduncle_path: str):
    paths = [Path(cluster_path), Path(peduncle_path)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise gr.Error("Missing trained model: " + ", ".join(missing))
    YOLO = require_ultralytics()
    return YOLO(str(paths[0])), YOLO(str(paths[1]))


def run_pipeline(image_path: str, cluster_conf: float, peduncle_conf: float,
                 cluster_path: str, peduncle_path: str):
    if not image_path:
        raise gr.Error("Upload or select an image first.")
    image = cv2.imread(image_path)
    if image is None:
        raise gr.Error(f"Could not read image: {image_path}")
    cluster_model, peduncle_model = load_models(cluster_path, peduncle_path)
    detections, elapsed_ms = analyze_image(
        image, cluster_model, peduncle_model, select_device("auto"),
        cluster_conf, peduncle_conf, 640,
    )
    annotated = cv2.cvtColor(draw_pipeline(image, detections), cv2.COLOR_BGR2RGB)
    found = sum(item["peduncle_status"] == "FOUND" for item in detections)
    summary = (
        f"**Clusters:** {len(detections)}  |  **Peduncles found:** {found}  |  "
        f"**Inference:** {elapsed_ms:.1f} ms"
    )
    table = [
        [
            item["cluster_id"], item["cluster_confidence"], item["peduncle_status"],
            item["peduncle_confidence"], str(item["cut_point"]),
        ]
        for item in detections
    ]
    payload = json.dumps({"detections": serializable(detections)}, indent=2)
    return annotated, summary, table, payload


def build_demo(cluster_path: str, peduncle_path: str) -> gr.Blocks:
    example_root = PROJECT_ROOT / "data" / "external" / "canopies" / "data_0001" / "frames"
    examples = [[str(path)] for path in sorted(example_root.rglob("*.jpg"))[:6]]
    with gr.Blocks(title="Grape Cluster and Peduncle Pipeline") as demo:
        gr.Markdown(
            "# Grape cluster and peduncle segmentation\n"
            "The first network detects each bunch; a specialised segmentation network then "
            "looks above it for the visible peduncle and proposes a cutting point."
        )
        with gr.Row():
            with gr.Column():
                source = gr.Image(type="filepath", label="Vineyard image")
                cluster_conf = gr.Slider(0.05, 0.90, value=0.25, step=0.025,
                                         label="Cluster confidence")
                peduncle_conf = gr.Slider(0.05, 0.90, value=DEFAULT_PEDUNCLE_CONF_THRESHOLD, step=0.025,
                                          label="Peduncle confidence")
                run_button = gr.Button("Find clusters and peduncles", variant="primary")
            output = gr.Image(type="numpy", label="Masks and cutting points")
        if examples:
            gr.Examples(examples=examples, inputs=source, label="CANOPIES examples")
        summary = gr.Markdown("Run the pipeline to see results.")
        table = gr.Dataframe(
            headers=["cluster", "cluster confidence", "peduncle", "peduncle confidence", "cut point"],
            datatype=["str", "number", "str", "number", "str"], interactive=False,
        )
        payload = gr.Code(label="Machine-readable result", language="json")
        run_button.click(
            fn=lambda image, gc, pc: run_pipeline(image, gc, pc, cluster_path, peduncle_path),
            inputs=[source, cluster_conf, peduncle_conf], outputs=[output, summary, table, payload],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-model", type=Path, default=DEFAULT_CLUSTER_MODEL)
    parser.add_argument("--peduncle-model", type=Path, default=PEDUNCLE_SEGMENTATION_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo(str(args.cluster_model.resolve()), str(args.peduncle_model.resolve())).launch(
        server_name=args.host, server_port=args.port, share=args.share,
    )


if __name__ == "__main__":
    main()
