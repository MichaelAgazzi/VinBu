# Models

`grape_yolo11n_best.pt` is the best checkpoint from the completed 50-epoch
YOLO11n baseline. `grape_yolo11s_best.pt` is the improved 50-epoch VINEPICs
checkpoint and is the default for the demo: on the held-out VINEPICs test it
improved recall from 0.429 to 0.513 and mAP50 from 0.425 to 0.472. The larger
checkpoint is about 19 MB versus 5.4 MB for YOLO11n.

`smoke_yolo11n_best.pt` is only the earlier one-epoch pipeline smoke test and
should not be used for the demo.

`grape_yolo11n_vinepics_wgisd_canopies_best.pt` is the 50-epoch generalist
trained on all three datasets. It generalises much better to WGISD and CANOPIES,
but does not improve the VINEPICs held-out test.

`grape_yolo11n_combined_finetuned_vinepics_best.pt` is an ablation checkpoint.
It showed catastrophic forgetting on CANOPIES and is retained only to make the
experiment reproducible; it is not recommended for the demo.

`grape_yolo11s_small_objects_best.pt` was trained on a 50/50 mix of full
VINEPICs training images and 640-pixel crops. With tiled 640 inference at
confidence 0.50 it reached fixed-threshold precision 0.674, recall 0.538 and
F1 0.598 on the original held-out test. The Gradio demo uses this checkpoint
and inference mode by default. Confidence 0.25 is the high-recall setting
(recall 0.601, precision 0.502).

Training automatically copies the best checkpoint from its Ultralytics run
directory into this folder.
