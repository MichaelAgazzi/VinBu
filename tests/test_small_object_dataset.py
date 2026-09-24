import unittest

from src.build_small_object_dataset import boxes_for_crop


class SmallObjectDatasetTests(unittest.TestCase):
    def test_box_is_rebased_and_normalized_to_padded_canvas(self):
        lines = boxes_for_crop([[100, 100, 200, 200]], 50, 50, 640, 480, 640)
        values = [float(value) for value in lines[0].split()[1:]]
        self.assertEqual(values, [0.15625, 0.15625, 0.15625, 0.15625])

    def test_box_owned_by_other_crop_is_not_duplicated(self):
        lines = boxes_for_crop([[620, 100, 700, 200]], 0, 0, 640, 640, 640)
        self.assertEqual(lines, [])

    def test_mostly_truncated_box_is_rejected(self):
        lines = boxes_for_crop([[600, 100, 700, 200]], 0, 0, 640, 640, 640, min_visible=0.5)
        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
