"""Dependency-free unit tests for dataset conversion logic."""

import unittest

from src.config import DEFAULT_SESSION_SPLIT, SESSION_METADATA
from src.dataset_utils import build_session_split, coco_bbox_to_yolo


class DatasetUtilsTests(unittest.TestCase):
    def test_coco_to_yolo(self):
        result = coco_bbox_to_yolo([10, 20, 20, 40], 100, 200)
        self.assertEqual(result, (0.2, 0.2, 0.2, 0.2))

    def test_out_of_frame_box_is_clipped(self):
        result = coco_bbox_to_yolo([-10, -20, 30, 50], 100, 200)
        self.assertEqual(result, (0.1, 0.075, 0.2, 0.15))

    def test_default_split_is_deterministic(self):
        sessions = set(DEFAULT_SESSION_SPLIT)
        self.assertEqual(build_session_split(sessions, 42), DEFAULT_SESSION_SPLIT)

    def test_each_variety_has_all_splits(self):
        varieties = {item["variety"] for item in SESSION_METADATA.values()}
        for variety in varieties:
            splits = {
                DEFAULT_SESSION_SPLIT[session]
                for session, metadata in SESSION_METADATA.items()
                if metadata["variety"] == variety
            }
            self.assertEqual(splits, {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
