import unittest

from src.tiled_inference import classless_nms, tile_origins


class TiledInferenceTests(unittest.TestCase):
    def test_origins_cover_edge_without_duplicates(self):
        self.assertEqual(tile_origins(1280, 640, 0.25), [0, 480, 640])
        self.assertEqual(tile_origins(500, 640, 0.25), [0])

    def test_nms_keeps_best_overlap_and_separate_box(self):
        detections = [
            {"box": [0, 0, 100, 100], "confidence": 0.7},
            {"box": [5, 5, 105, 105], "confidence": 0.9},
            {"box": [200, 200, 250, 250], "confidence": 0.6},
        ]
        kept = classless_nms(detections, 0.5)
        self.assertEqual([item["confidence"] for item in kept], [0.9, 0.6])


if __name__ == "__main__":
    unittest.main()
