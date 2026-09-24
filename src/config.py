"""Central configuration for the grape-cluster detection prototype."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original VINEPICs data. Project code treats this directory as read-only.
RAW_DATA_DIR = PROJECT_ROOT / "VINEPICs" / "data"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_ANNOTATIONS_FILE = RAW_DATA_DIR / "annotations" / "VINEPICs_annotations.json"

DATASET_DIR = PROJECT_ROOT / "data" / "yolo_dataset"
DATASET_YAML = DATASET_DIR / "dataset.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASET_CHECKS_DIR = RESULTS_DIR / "dataset_checks"
EVALUATION_DIR = RESULTS_DIR / "evaluation"
DETECTIONS_DIR = RESULTS_DIR / "detections"
REVIEWED_DETECTIONS_FILE = RESULTS_DIR / "reviewed_detections.json"

DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_TRAINED_MODEL = MODELS_DIR / "grape_yolo11s_best.pt"
SMALL_OBJECT_TRAINED_MODEL = MODELS_DIR / "grape_yolo11s_small_objects_best.pt"
MULTISCALE_TRAINED_MODEL = MODELS_DIR / "grape_yolo11m_multiscale_960_b4_best.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = -1
DEFAULT_PATIENCE = 15

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
HIGH_CONFIDENCE_THRESHOLD = 0.90
REVIEW_CONFIDENCE_THRESHOLD = 0.65

CLASS_NAMES = {0: "grape_cluster"}
COCO_CATEGORY_TO_YOLO = {1: 0}
RANDOM_SEED = 42


def get_confidence_tier(confidence: float) -> str:
    """Return a configurable demo confidence tier (not scientifically validated)."""
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH_CONFIDENCE"
    if confidence >= REVIEW_CONFIDENCE_THRESHOLD:
        return "REVIEW"
    return "LOW_CONFIDENCE"


# Capture-session split. Consecutive frames never cross splits, and every variety
# is represented in train/val/test. Counts are 169/42/27 images (~71/18/11).
DEFAULT_SESSION_SPLIT = {
    "2021-07-27": "train",
    "2021-08-23": "val",
    "2021-09-06": "train",
    "2022-08-23-12-23-31": "train",
    "2022-08-23-12-32-48": "train",
    "2022-08-23-14-17-26": "train",
    "2022-08-23-14-32-03": "test",
    "2022-08-23-15-22-59": "train",
    "2022-08-23-15-32-40": "val",
    "2022-08-23-17-02-58": "train",
    "2022-08-23-17-46-30": "train",
    "2022-08-23-17-53-22": "test",
    "2022-09-15-12-13-19": "train",
    "2022-09-15-12-20-49": "train",
    "2022-09-15-13-08-12": "test",
    "2022-09-15-13-14-36": "train",
    "2022-09-15-14-02-31": "train",
    "2022-09-15-14-08-23": "val",
}

SESSION_METADATA = {
    "2021-07-27": {"variety": "Red Globe", "defoliation": "yes", "camera": "45deg"},
    "2021-08-23": {"variety": "Red Globe", "defoliation": "yes", "camera": "45deg"},
    "2021-09-06": {"variety": "Red Globe", "defoliation": "yes", "camera": "45deg"},
    "2022-08-23-12-23-31": {"variety": "Red Globe", "defoliation": "no", "camera": "45deg"},
    "2022-08-23-12-32-48": {"variety": "Red Globe", "defoliation": "no", "camera": "90deg"},
    "2022-08-23-14-17-26": {"variety": "Red Globe", "defoliation": "yes", "camera": "90deg"},
    "2022-08-23-14-32-03": {"variety": "Red Globe", "defoliation": "yes", "camera": "45deg"},
    "2022-08-23-15-22-59": {"variety": "Ortrugo", "defoliation": "no", "camera": "45deg"},
    "2022-08-23-15-32-40": {"variety": "Ortrugo", "defoliation": "no", "camera": "90deg"},
    "2022-08-23-17-02-58": {"variety": "Ortrugo", "defoliation": "50%", "camera": "45deg"},
    "2022-08-23-17-46-30": {"variety": "Ortrugo", "defoliation": "100%", "camera": "45deg"},
    "2022-08-23-17-53-22": {"variety": "Ortrugo", "defoliation": "100%", "camera": "90deg"},
    "2022-09-15-12-13-19": {"variety": "Cabernet Sauvignon", "defoliation": "no", "camera": "90deg"},
    "2022-09-15-12-20-49": {"variety": "Cabernet Sauvignon", "defoliation": "no", "camera": "45deg"},
    "2022-09-15-13-08-12": {"variety": "Cabernet Sauvignon", "defoliation": "50%", "camera": "45deg"},
    "2022-09-15-13-14-36": {"variety": "Cabernet Sauvignon", "defoliation": "50%", "camera": "90deg"},
    "2022-09-15-14-02-31": {"variety": "Cabernet Sauvignon", "defoliation": "100%", "camera": "90deg"},
    "2022-09-15-14-08-23": {"variety": "Cabernet Sauvignon", "defoliation": "100%", "camera": "45deg"},
}


class ReviewStatus:
    AUTO_APPROVED = "AUTO_APPROVED"
    HUMAN_APPROVED = "HUMAN_APPROVED"
    REJECTED = "REJECTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
