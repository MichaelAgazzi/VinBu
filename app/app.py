"""Small local Gradio demo for grape-cluster detection and human review."""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (  # noqa: E402
    draw_detections,
    model_is_grape_detector,
    predictions_to_detections,
    require_ultralytics,
    result_to_detections,
    select_device,
    timed_predict,
)
from src.config import (  # noqa: E402
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_TRAINED_MODEL,
    MULTISCALE_TRAINED_MODEL,
    REVIEWED_DETECTIONS_FILE,
    ReviewStatus,
    SMALL_OBJECT_TRAINED_MODEL,
)
from src.tiled_inference import classless_nms, predict_tiled  # noqa: E402


@lru_cache(maxsize=2)
def load_model(path: str):
    model_path = Path(path)
    if not model_path.is_file():
        raise gr.Error(
            f"Trained model not found: {model_path}. "
            "Run 'python -m src.train_detector' first."
        )
    model = require_ultralytics()(str(model_path))
    if not model_is_grape_detector(model):
        raise gr.Error("Selected weights are not a trained grape_cluster model.")
    return model


def run_detection(image_path: str, conf: float, tiled: bool, ensemble: bool, model_path: str):
    if not image_path:
        raise gr.Error("Upload or select an image first.")
    model = load_model(model_path)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise gr.Error(f"Could not read image: {image_path}")
    identifier = Path(image_path).name
    device = select_device("auto")
    if ensemble:
        started = time.perf_counter()
        large_model = load_model(str(MULTISCALE_TRAINED_MODEL.resolve()))
        predictions = classless_nms(
            predict_tiled(
                model, bgr, conf=conf, imgsz=640, device=device,
                tile_size=640, overlap=0.25, nms_iou=0.50,
            )
            + predict_tiled(
                large_model, bgr, conf=conf, imgsz=960, device=device,
                tile_size=640, overlap=0.25, nms_iou=0.50,
            ),
            0.50,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = predictions_to_detections(predictions, identifier)
        mode = "YOLO11s+YOLO11m tiled ensemble"
    elif tiled:
        started = time.perf_counter()
        predictions = predict_tiled(
            model, bgr, conf=conf, imgsz=DEFAULT_IMGSZ, device=device,
            tile_size=640, overlap=0.25, nms_iou=0.50,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = predictions_to_detections(predictions, identifier)
        mode = "tiled 640"
    else:
        result, elapsed_ms = timed_predict(
            model, bgr, conf=conf, iou=DEFAULT_IOU_THRESHOLD,
            imgsz=DEFAULT_IMGSZ, device=device,
        )
        detections = result_to_detections(result, identifier)
        mode = "full image"
    annotated = cv2.cvtColor(draw_detections(bgr, detections), cv2.COLOR_BGR2RGB)
    average = sum(item["confidence"] for item in detections) / len(detections) if detections else 0.0
    summary = (
        f"**Grape clusters:** {len(detections)}  |  "
        f"**Average confidence:** {average:.3f}  |  "
        f"**Mode:** {mode}  |  **Inference time:** {elapsed_ms:.1f} ms"
    )
    table = [
        [item["detection_id"], item["confidence"], item["confidence_tier"],
         item["x1"], item["y1"], item["x2"], item["y2"]]
        for item in detections
    ]
    choices = [item["detection_id"] for item in detections]
    state = {"image": identifier, "detections": detections}
    return annotated, summary, table, state, gr.Dropdown(
        choices=choices, value=choices[0] if choices else None
    )


def save_review(state: dict, detection_id: str, review_status: str):
    if not state or not detection_id:
        raise gr.Error("Run detection and select a detection first.")
    match = next((item for item in state["detections"] if item["detection_id"] == detection_id), None)
    if match is None:
        raise gr.Error(f"Detection {detection_id} is not in the current result.")
    match["review_status"] = review_status

    REVIEWED_DETECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        database = json.loads(REVIEWED_DETECTIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        database = {"images": {}}
    database.setdefault("images", {})[state["image"]] = state
    REVIEWED_DETECTIONS_FILE.write_text(json.dumps(database, indent=2), encoding="utf-8")
    return f"Saved {detection_id} as {review_status}.", state


def build_demo(model_path: str) -> gr.Blocks:
    examples = [
        PROJECT_ROOT / "VINEPICs" / "data" / "images" / "2021-08-23" / "rgb27.png",
        PROJECT_ROOT / "VINEPICs" / "data" / "images" / "2022-09-15-12-13-19" /
        "rgb-2022-09-15-12-13-43-393479.png",
        PROJECT_ROOT / "VINEPICs" / "data" / "images" / "2022-09-15-14-08-23" /
        "rgb-2022-09-15-14-08-44-200591.png",
    ]
    examples = [[str(path)] for path in examples if path.is_file()]

    with gr.Blocks(title="Vineyard Grape-Cluster Detector") as demo:
        gr.Markdown(
            "# Vineyard grape-cluster detector\n"
            "Upload a vineyard image or select an example, then run the trained YOLO11 model. "
            "Confidence tiers are configurable demo defaults, not validated decision thresholds."
        )
        current = gr.State({})
        with gr.Row():
            with gr.Column():
                source = gr.Image(type="filepath", label="Input vineyard image")
                confidence = gr.Slider(0.05, 0.95, value=0.50, step=0.025,
                                       label="Detection confidence threshold")
                tiled = gr.Checkbox(
                    value=True,
                    label="Small-object tiled mode (better recall, slower)",
                )
                ensemble = gr.Checkbox(
                    value=False,
                    label="High-accuracy ensemble (best F1, much slower; use confidence 0.60)",
                )
                detect_button = gr.Button("Detect grape clusters", variant="primary")
            output = gr.Image(type="numpy", label="Detected grape clusters")
        if examples:
            gr.Examples(examples=examples, inputs=source, label="Dataset examples")
        summary = gr.Markdown("Run detection to see results.")
        table = gr.Dataframe(
            headers=["ID", "confidence", "tier", "x1", "y1", "x2", "y2"],
            datatype=["str", "number", "str", "number", "number", "number", "number"],
            interactive=False,
            label="Detection table",
        )
        gr.Markdown("### Human review preparation")
        with gr.Row():
            detection_id = gr.Dropdown([], label="Detection ID")
            status = gr.Radio(
                [ReviewStatus.HUMAN_APPROVED, ReviewStatus.REJECTED, ReviewStatus.REVIEW_REQUIRED],
                value=ReviewStatus.HUMAN_APPROVED,
                label="Review decision",
            )
            save_button = gr.Button("Save review")
        review_message = gr.Markdown()

        detect_button.click(
            fn=lambda image, conf, use_tiles, use_ensemble: run_detection(
                image, conf, use_tiles, use_ensemble, model_path
            ),
            inputs=[source, confidence, tiled, ensemble],
            outputs=[output, summary, table, current, detection_id],
        )
        save_button.click(
            save_review,
            inputs=[current, detection_id, status],
            outputs=[review_message, current],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SMALL_OBJECT_TRAINED_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_demo(str(args.model.resolve())).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
