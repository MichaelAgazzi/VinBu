# Generated data

The original, read-only dataset lives in `VINEPICs/data/`.

Run `python -m src.dataset_utils` to generate `yolo_dataset/`. By default, the
converter tries hard links first, then symbolic links, and only copies as a last
resort. Labels are converted from the original COCO bounding boxes. Capture
sessions, rather than individual frames, are assigned to train/validation/test.

