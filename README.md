# Vineyard grape-cluster detection demo

This repository contains the first perception prototype for the Farm Robotics
Challenge: detect every visible grape bunch in an RGB vineyard image, draw a
bounding box and confidence score, export machine-readable detections, and
prepare each result for human approval or rejection.

Only grape-cluster object detection is implemented. Robot control, navigation,
cutting, depth localization, and manipulation are intentionally out of scope.

## Current dataset status

The workspace contains the complete VINEPICs dataset:

- 238 PNG images in 18 capture sessions;
- 2,403 manually annotated grape bunches in COCO JSON;
- one source class, `bunch`, converted to YOLO class `0: grape_cluster`;
- polygon segmentations and bounding boxes for all 2,403 objects;
- 235 positive images and 3 valid empty/background images;
- 26 images at 480 x 848 and 212 images at 720 x 1280;
- no invalid or out-of-frame source boxes found.

The prepared YOLO split contains 169 train, 42 validation, and 27 test images.
It is grouped by capture session, not random frames, so adjacent robot frames do
not leak across splits. Image entries are hard links where Windows permits it;
the original `VINEPICs/` data is never edited.

Generated ground-truth checks are in `results/dataset_checks/`, including:

- `annotated_grape_examples.png` (three varieties with grape boxes);
- `dataset_distributions.png` (clusters/image and relative box size);
- `clusters_by_session.png` (coverage across the 18 sessions).

These are ground-truth annotations, not model predictions.

## Current result

YOLO11n and YOLO11s were trained for 50 epochs and evaluated on the untouched
27-image test split (238 labelled clusters). YOLO11s is now the default:

| Metric | YOLO11n | YOLO11s |
|---|---:|---:|
| Precision | 0.684 | 0.679 |
| Recall | 0.429 | **0.513** |
| mAP50 | 0.425 | **0.472** |
| mAP50-95 | 0.212 | **0.260** |

At the fixed demo threshold (`confidence=0.25`, `IoU=0.50`), the separate greedy
error analysis counted 115 TP, 110 FP, and 123 FN (precision 0.511, recall
0.483). This diagnostic uses a different matching/aggregation procedure from
the Ultralytics metrics above, so the two sets of precision/recall values should
not be compared directly.

At confidence 0.25, the YOLO11s fixed-threshold diagnostic counted 131 TP, 109
FP and 107 FN (precision 0.546, recall 0.550). Its validation-selected best-F1
threshold is 0.375; confidence 0.25 remains the demo default when missed bunches
are more costly than extra proposals. The default checkpoint is
`models/grape_yolo11s_best.pt`. These are research results, not safety
validation for autonomous harvesting.

### Small-object crop-aware result

The follow-up YOLO11s experiment trains on a 50/50 mixture of full VINEPICs
training images and native-resolution 640x640 crops. Validation and test remain
the original session-separated images. Crop annotations are clipped and
renormalized; partial objects are retained only when their centre belongs to
the crop and at least 50% of the original box remains visible.

| Test strategy (IoU 0.50) | Confidence | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Previous YOLO11s, full image | 0.375 | 0.684 | 0.492 | 0.572 |
| Crop-aware YOLO11s, tiled 640 | 0.500 | **0.674** | **0.538** | **0.598** |
| Crop-aware YOLO11s, tiled 640, high recall | 0.250 | 0.502 | **0.601** | 0.547 |

The balanced 0.50 threshold was selected on validation, not the test. The
Gradio demo now defaults to `models/grape_yolo11s_small_objects_best.pt` with
tiled mode enabled. Disable tiled mode for faster inference, or lower confidence
to 0.25 when missed clusters are more costly than false proposals.

Rebuild the generated crop dataset and reproduce training with:

```powershell
.\.venv\Scripts\python.exe -m src.build_small_object_dataset
.\.venv\Scripts\python.exe -m src.validate_dataset --dataset data\small_object_dataset --output results\small_object_dataset_checks
.\.venv\Scripts\python.exe -m src.train_detector --data data\small_object_dataset\dataset.yaml --model yolo11s.pt --epochs 50 --imgsz 640 --batch -1 --small-object-augmentations --name grape_yolo11s_small_objects
```

## WGISD + CANOPIES experiment

Two public datasets were downloaded and integrated without modifying their
sources:

- Embrapa WGISD: 300 images and 4,430 YOLO boxes, CC BY-NC 4.0;
- CANOPIES GBPD: 810 images and 1,312 cluster polygons, CC BY-NC-SA 3.0,
  academic/non-commercial use.

The combined preparation uses hard links and preserves the original VINEPICs
test split. WGISD's official test set is retained, while CANOPIES is split by
complete capture sequence. The primary combined model was trained for 50 epochs
at 640 px as `models/grape_yolo11n_vinepics_wgisd_canopies_best.pt`.

| Training | Held-out dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| VINEPICs baseline | VINEPICs | 0.684 | 0.429 | 0.425 | 0.212 |
| Combined | VINEPICs | 0.667 | 0.424 | 0.411 | 0.212 |
| VINEPICs baseline | WGISD | 0.562 | 0.165 | 0.129 | 0.052 |
| Combined | WGISD | 0.821 | 0.819 | 0.825 | 0.515 |
| VINEPICs baseline | CANOPIES | 0.014 | 0.003 | 0.000 | 0.000 |
| Combined | CANOPIES | 0.656 | 0.738 | 0.644 | 0.417 |

The external data dramatically improves cross-dataset generalisation but does
not improve the untouched VINEPICs test. CANOPIES intentionally labels a bunch
only when its peduncle is visible without zooming, so many visible bunches are
unlabelled under its task definition. This conflicts with the project's goal of
detecting every visible bunch. A 20-epoch VINEPICs fine-tune was also tested but
caused catastrophic forgetting on CANOPIES and was rejected as the default.

Rebuild and validate the combined data with:

```powershell
.\.venv\Scripts\python.exe -m src.build_combined_dataset --force
.\.venv\Scripts\python.exe -m src.validate_dataset --dataset data\combined_dataset --output results\combined_dataset_checks
```

Repeat the combined training with:

```powershell
.\.venv\Scripts\python.exe -m src.train_detector --data data\combined_dataset\dataset.yaml --model yolo11n.pt --epochs 50 --imgsz 640 --batch -1 --name grape_yolo11n_vinepics_wgisd_canopies
```

Launch the demo with the generalist model explicitly:

```powershell
.\.venv\Scripts\python.exe app\app.py --model models\grape_yolo11n_vinepics_wgisd_canopies_best.pt
```

## Installation (Windows PowerShell)

The tested environment uses 64-bit Python 3.11.9, Ultralytics 8.4.160, PyTorch
2.14.0 with CUDA 13.0, and an NVIDIA GeForce RTX 5060 Laptop GPU (8 GB):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the PyTorch installation selector for a different GPU/driver platform. The
scripts select CUDA automatically when available and fall back to CPU.

## Prepare and inspect the dataset

The dataset is already prepared in this workspace. To regenerate it with Python:

```powershell
.\.venv\Scripts\python.exe -m src.dataset_utils --force-labels
.\.venv\Scripts\python.exe -m src.validate_dataset --samples 12
```

Windows users can also recreate the same labels/split without Python with:

```powershell
.\tools\bootstrap_dataset.ps1 -ForceLabels
```

## Train YOLO11

Default training uses `yolo11n.pt`, 50 epochs, 640-pixel images, automatic batch
size, patience 15, and deterministic seed 42:

```powershell
.\.venv\Scripts\python.exe -m src.train_detector
```

Equivalent configurable example:

```powershell
.\.venv\Scripts\python.exe -m src.train_detector --model yolo11n.pt --epochs 50 --imgsz 640 --batch -1 --device auto
```

On first use, Ultralytics downloads the small pretrained `yolo11n.pt`
checkpoint. The tested run selected batch size 14 automatically and took about
five minutes on the RTX 5060 Laptop GPU. The best fine-tuned weights are copied
to `models/grape_yolo11n_best.pt`. Use `yolo11s.pt` only after confirming GPU
memory and comparing validation recall.

## Run inference

One image:

```powershell
.\.venv\Scripts\python.exe -m src.detect --source VINEPICs\data\images\2021-08-23\rgb27.png
```

A folder or video:

```powershell
.\.venv\Scripts\python.exe -m src.detect --source VINEPICs\data\images\2021-08-23
.\.venv\Scripts\python.exe -m src.detect --source path\to\vineyard_video.mp4
```

Annotated media and a CSV plus JSON export are written to `results/detections/`.
Each row contains image/frame ID, `G001`-style detection ID, class, confidence,
confidence tier, box corners, center coordinates, and review status.

The detector rejects generic COCO weights by default: using untrained
`yolo11n.pt` as if it were a grape detector would produce misleading results.

## Launch the local demo

The trained checkpoint is already present. Launch the demo with:

```powershell
.\.venv\Scripts\python.exe app\app.py
```

Open `http://127.0.0.1:7860`. Upload an image or select a bundled example. The
demo shows the annotated image, cluster count, average confidence, inference
time, and detection table. Review decisions are stored in
`results/reviewed_detections.json`.

## Evaluate

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --split test
```

This writes precision, recall, mAP50, mAP50-95, confusion/PR plots, and sample
true-positive, false-positive, and false-negative images under
`results/evaluation/`. Grape-cluster recall at confidence threshold 0.25 and IoU
0.50 is also reported.

Useful outputs from the completed run are:

- `results/training/grape_yolo11n/results.png` for training curves;
- `results/evaluation/test/BoxPR_curve.png` for the test PR curve;
- `results/evaluation/test/confusion_matrix.png` for error counts;
- `results/evaluation/test/samples/` for TP/FP/FN visual examples;
- `results/detections/` for annotated predictions and CSV/JSON exports.
- `results/evaluation/model_comparison.png` for the three-model,
  three-dataset comparison.

## Confidence tiers

The UI uses demonstration defaults only:

- `HIGH_CONFIDENCE`: confidence >= 0.90
- `REVIEW`: 0.65 <= confidence < 0.90
- `LOW_CONFIDENCE`: confidence < 0.65

They are not scientifically validated thresholds and should be tuned on the
held-out validation data for the eventual harvesting risk profile.

## Project layout

```text
app/app.py                     Gradio demo and review capture
data/yolo_dataset/             Generated YOLO images, labels, YAML, split summary
data/combined_dataset/         Generated leakage-safe combined experiment
data/external/                 Preserved WGISD and CANOPIES sources (ignored by Git)
models/                        Best trained checkpoint (after training)
results/dataset_checks/        Annotation overlays and dataset plots
results/detections/            Inference images/videos and CSV/JSON
results/evaluation/            Metrics and error-analysis images
src/config.py                  Paths, split, classes, thresholds
src/dataset_utils.py           COCO-to-YOLO conversion and session split
src/build_combined_dataset.py  External-data conversion and grouped splits
src/validate_dataset.py        Pair/box validation and visual checks
src/train_detector.py          Reproducible YOLO11 training
src/detect.py                  Image/folder/video inference
src/evaluate.py                Metrics and TP/FP/FN analysis
src/summarize_experiments.py   Cross-dataset result table and plot
tools/                         Windows bootstrap and annotation report tools
```

## Limitations and next step

The dataset is small and contains strong shifts in grape variety, foliage,
lighting, scale, and camera angle. Qualitative test review shows useful results
for many isolated Red Globe and Ortrugo clusters, but frequent misses on small,
dense, or heavily occluded Cabernet clusters. Bounding boxes also cannot express
the precise cut point or depth required by a harvesting robot.

No manual annotation is needed to reproduce this baseline: all VINEPICs images
already have labels. Manual labelling is needed only when adding deployment
images or correcting discovered label errors.

The single most important next step is an error-driven data iteration: capture
and label more deployment-like images dominated by dense, small, and occluded
clusters, retrain with the same session-separated evaluation, and measure
whether test recall improves before increasing model size.

# VinBu
