import unittest

from src.sweep_test_thresholds import average_precision


class AveragePrecisionTests(unittest.TestCase):
    def test_perfect_prediction_has_unit_ap(self):
        samples = [([[0, 0, 10, 10]], [{"box": [0, 0, 10, 10], "confidence": 0.9}])]
        self.assertAlmostEqual(average_precision(samples, 0.5), 1.0)

    def test_higher_confidence_false_positive_reduces_ap(self):
        samples = [
            (
                [[0, 0, 10, 10]],
                [
                    {"box": [20, 20, 30, 30], "confidence": 0.9},
                    {"box": [0, 0, 10, 10], "confidence": 0.8},
                ],
            )
        ]
        self.assertAlmostEqual(average_precision(samples, 0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
