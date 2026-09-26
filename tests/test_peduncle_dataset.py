import unittest

from src.build_peduncle_dataset import (
    assign_sequences,
    clip_polygon_to_rect,
    polygon_area,
    roi_from_cluster_box,
)


class PeduncleDatasetTests(unittest.TestCase):
    def test_sequence_split_is_deterministic_and_disjoint(self):
        sequences = [f"sequence-{index}" for index in range(10)]
        first = assign_sequences(sequences, 42)
        second = assign_sequences(sequences, 42)
        self.assertEqual(first, second)
        self.assertEqual({"train", "val", "test"}, set(first.values()))
        self.assertEqual(10, len(first))

    def test_roi_keeps_space_above_cluster(self):
        roi = roi_from_cluster_box((300, 250, 500, 600), 1280, 720)
        self.assertLess(roi[1], 250)
        self.assertEqual(roi[2] - roi[0], roi[3] - roi[1])
        self.assertGreaterEqual(roi[0], 0)
        self.assertLessEqual(roi[2], 1280)

    def test_polygon_is_clipped_to_crop(self):
        polygon = [(-10, 10), (50, 10), (50, 50), (-10, 50)]
        clipped = clip_polygon_to_rect(polygon, 0, 0, 40, 40)
        self.assertAlmostEqual(polygon_area(clipped), 1200.0)
        self.assertTrue(all(0 <= x <= 40 and 0 <= y <= 40 for x, y in clipped))


if __name__ == "__main__":
    unittest.main()
