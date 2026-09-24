# Progress

Updated: 2026-09-24

## Workspace findings

- Source dataset: VINEPICs, preserved under `VINEPICs/data/`.
- Images: 238 PNG files in 18 time-based capture sessions.
- Resolutions: 26 at 480 x 848; 212 at 720 x 1280.
- Annotations: COCO JSON with 2,403 manually drawn single-class `bunch`
  instances. Every instance has both a bounding box and polygon segmentation.
- Positive/background balance: 235 annotated images and 3 empty images.
- Source annotation validation: 0 non-positive boxes and 0 out-of-frame boxes.
- Conditions: Red Globe, Ortrugo, and Cabernet Sauvignon; 45/90-degree camera
  views; no/partial/full defoliation; occlusion, backlight, and large scale shifts.
- Hardware seen: NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB.
- Environment: Python 3.11.9 virtual environment, Ultralytics 8.4.160, PyTorch
  2.14.0+cu130, and working CUDA acceleration on the RTX 5060 Laptop GPU.

## Implemented

- Deterministic capture-session split to prevent adjacent-frame leakage.
- COCO-to-YOLO converter with clipping checks and hard-link/copy fallback.
- Prepared YOLO dataset: 169 train / 42 val / 27 test images and labels.
- Generated 2,403 YOLO boxes; validation found 0 malformed rows.
- Dataset checker for missing/orphan labels, invalid boxes, distributions, and
  sampled overlays.
- YOLO11n training entry point with automatic CUDA/CPU selection, reproducible
  configuration, early stopping, Windows-safe workers, and best-weight export.
- Image, folder, and video inference with annotated media plus CSV/JSON exports.
- Configurable confidence threshold and three explicitly non-validated tiers.
- Evaluation script reporting precision, recall, mAP50, mAP50-95, recall at
  IoU=0.5, confusion/PR plots, and TP/FP/FN sample images.
- Gradio upload/example demo with counts, confidence, timing, detection table,
  and persisted approve/reject/review decisions.
- Ground-truth plots under `results/dataset_checks/`.
- Downloaded Embrapa WGISD (300 images, 4,430 boxes) and CANOPIES GBPD
  (810 images, 1,312 cluster polygons), preserving both sources under
  `data/external/`.
- Built `data/combined_dataset/` using hard links, WGISD's official held-out
  test, and sequence-level CANOPIES splits. Combined validation reports 1,116
  primary images, 6,921 boxes, and 0 invalid entries.
- Added cross-dataset evaluation routing and an automated comparison CSV/JSON
  and plot under `results/evaluation/model_comparison.*`.
- Per-image test overlays, worst-image montage, object-size error analysis, and
  confidence-threshold sweeps are available under `results/test_predictions/`
  and `results/threshold_sweeps/`.
- Automated checks: 6 unit tests pass, all Python modules compile, the
  prepared dataset validator reports 2,403 valid boxes and 0 invalid rows, and
  the live Gradio service returned HTTP 200 in a launch smoke test.

## Training and metrics

- Improved architecture: YOLO11s trained on VINEPICs for 50 epochs at 640 px,
  batch 7, seed 42; runtime about 7 minutes.
- Default checkpoint: `models/grape_yolo11s_best.pt`.
- Held-out test: precision 0.679, recall 0.513, mAP50 0.472, mAP50-95 0.260.
- At confidence 0.25 / IoU 0.50: 131 TP, 109 FP, 107 FN; fixed-threshold
  precision 0.546 and recall 0.550.
- Validation-selected best-F1 confidence is 0.375. Full-image inference at 640
  was retained: 960/1280 px and 640/960-pixel tiling did not improve F1.

### Small-object crop experiment

- Generated a leakage-safe training set with 338 full-image entries and 338
  native-resolution 640x640 crops. Original validation (42) and test (27)
  images are unchanged. Validation found 6,776 boxes and 0 invalid entries.
- Half of the crops receive a deterministic shadow, backlight, contrast, or
  blur transform; online training uses reduced scale/mosaic augmentation so
  the enlarged small-object scale is preserved.
- YOLO11s trained for 50 epochs; checkpoint:
  `models/grape_yolo11s_small_objects_best.pt`.
- Full-image test did not merit replacing the previous model by itself:
  P 0.615, R 0.523, mAP50 0.470, mAP50-95 0.244.
- Crop-aware tiled-640 inference, calibrated on validation at confidence 0.50,
  achieved 128 TP, 62 FP and 110 FN on the original test: P 0.674, R 0.538,
  F1 0.598. This beats the previous calibrated full-image F1 of 0.572.
- High-recall tiled operation at confidence 0.25 achieved 143 TP and 95 FN:
  P 0.502, R 0.601. The demo now exposes tiled mode and uses it by default.

### YOLO11m multiscale experiment

- Replaced identical full-image copies with real deterministic photometric
  variants and generated balanced 512/640/768-pixel crops: 1,014 training
  images, 4,274 crop boxes, original validation/test unchanged, 0 invalid rows.
- YOLO11m trained at 960 px with batch 4. Early stopping completed at epoch 45;
  the best epoch 30 validation result was P 0.749, R 0.589, mAP50 0.679, and
  mAP50-95 0.363.
- Original full-image test: P 0.671, R 0.508, mAP50 0.485, mAP50-95 0.277.
- YOLO11m tiled alone did not beat the existing operational F1 and was not
  promoted as the default.
- A two-model tiled ensemble, calibrated on validation at confidence 0.60 for
  both networks, achieved 135 TP, 70 FP, and 103 FN on the original test:
  P 0.659, R 0.567, F1 0.609. This is the best balanced F1 so far, at the cost
  of roughly doubling inference work.
- Tiled AP evaluation was added. On test, YOLO11m tiled reached AP50 0.541 and
  AP50-95 0.297; YOLO11s tiled remained slightly higher at 0.551 and 0.310.

- Starting architecture: Ultralytics YOLO11n (`yolo11n.pt`).
- Training status: **complete**, 50 epochs at 640 px, automatically selected
  batch size 14, seed 42, CUDA device 0. Runtime was about five minutes.
- Model checkpoint: `models/grape_yolo11n_best.pt`.
- Validation: precision 0.662, recall 0.504, mAP50 0.566, mAP50-95 0.296.
- Held-out test (27 images, 238 clusters): precision 0.684, recall 0.429,
  mAP50 0.425, mAP50-95 0.212.
- Fixed-threshold diagnostic at confidence 0.25 and IoU 0.50: 115 TP, 110 FP,
  123 FN; precision 0.511 and recall 0.483. This uses a separate greedy matcher
  and is not directly comparable with Ultralytics' aggregate precision/recall.
- Generated training, PR, confusion-matrix, and TP/FP/FN plots are retained
  under `results/training/` and `results/evaluation/`.

### External-data experiment

- Combined training: VINEPICs + WGISD + CANOPIES, YOLO11n, 50 epochs, 640 px,
  batch 13, seed 42; runtime about 23 minutes.
- Combined checkpoint: `models/grape_yolo11n_vinepics_wgisd_canopies_best.pt`.
- Combined validation: precision 0.733, recall 0.635, mAP50 0.702,
  mAP50-95 0.415.
- On the unchanged VINEPICs test, combined training did not improve the
  baseline: P 0.667, R 0.424, mAP50 0.411, mAP50-95 0.212.
- On held-out WGISD, the baseline-to-combined change was R 0.165 -> 0.819 and
  mAP50 0.129 -> 0.825.
- On held-out CANOPIES, the baseline-to-combined change was R 0.003 -> 0.738
  and mAP50 approximately 0.000 -> 0.644.
- A 20-epoch target fine-tune was evaluated and rejected for deployment: it did
  not improve VINEPICs and catastrophically forgot CANOPIES.

## Known limitations

- Bounding-box detection is implemented; segmentation masks are retained only in
  the original COCO annotations for future work.
- Human review is stored as JSON but does not yet convert edits into new labels.
- The explicit split prevents session leakage but is necessarily coarse because
  there are only 18 sessions.
- Qualitative review finds many misses on dense, small, and occluded Cabernet
  clusters; confidence tiers are demo defaults rather than calibrated harvesting
  decisions.
- CANOPIES labels a bunch only when its peduncle is visible without zooming;
  visible unlabelled bunches therefore act as background and conflict with the
  project's all-visible-bunch target.

## Next recommended task

Collect and fully label deployment-like images rich in dense, small, and
occluded Cabernet clusters. WGISD and CANOPIES improve broad generalisation but
do not replace target-domain labels with the same annotation policy.
