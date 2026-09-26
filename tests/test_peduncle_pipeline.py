import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.evaluate_peduncle_pipeline import box_iou, greedy_matches
from src.direct_peduncle_pipeline import associate_instances, association_score
from src.peduncle_pipeline import estimate_cut_point, mask_anchor
from src.sweep_segmentation_thresholds import ground_truth_masks


class PedunclePipelineTests(unittest.TestCase):
    def test_anchor_is_at_lower_end(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:81, 45:56] = True
        anchor = mask_anchor(mask)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0], 50.0)
        self.assertEqual(anchor[1], 80.0)

    def test_cut_point_is_inside_mask(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:81, 45:56] = True
        point = estimate_cut_point(mask)
        self.assertIsNotNone(point)
        self.assertTrue(mask[point[1], point[0]])
        self.assertLess(point[1], 55)

    def test_box_iou(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)

    def test_greedy_matching_is_one_to_one(self):
        ground_truth = [[0, 0, 10, 10], [20, 20, 30, 30]]
        predictions = [[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]]
        matches = greedy_matches(ground_truth, predictions, box_iou, 0.5)
        self.assertEqual(len(matches), 2)
        self.assertEqual({match[1] for match in matches}, {0, 2})

    def test_direct_association_prefers_nearby_peduncle(self):
        near = np.zeros((120, 120), dtype=bool)
        near[20:51, 48:53] = True
        far = np.zeros((120, 120), dtype=bool)
        far[20:51, 5:10] = True
        cluster = {"box": [35, 50, 65, 100], "confidence": 0.9}
        peduncles = [
            {"mask": far, "confidence": 0.9},
            {"mask": near, "confidence": 0.8},
        ]
        self.assertIsNone(association_score(cluster["box"], far, 0.9))
        self.assertEqual(associate_instances([cluster], peduncles), {0: 1})

    def test_threshold_ground_truth_filters_requested_class(self):
        with TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "sample.txt"
            label_path.write_text(
                "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
                "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9\n",
                encoding="utf-8",
            )
            class_zero = ground_truth_masks(label_path, (100, 100), class_id=0)
            class_one = ground_truth_masks(label_path, (100, 100), class_id=1)
            self.assertEqual(len(class_zero), 1)
            self.assertEqual(len(class_one), 1)
            self.assertTrue(class_zero[0][20, 20])
            self.assertFalse(class_zero[0][70, 70])
            self.assertTrue(class_one[0][70, 70])


if __name__ == "__main__":
    unittest.main()
