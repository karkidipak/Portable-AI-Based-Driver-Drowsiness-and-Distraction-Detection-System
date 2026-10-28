import unittest
from src.main import aspect_ratio, distance

class DetectionMathTests(unittest.TestCase):
    def test_distance_same_point(self):
        self.assertEqual(distance((2, 3), (2, 3)), 0.0)

    def test_eye_ratio_open_shape_positive(self):
        points = [(0, 2), (2, 0), (4, 0), (6, 2), (4, 4), (2, 4)]
        ratio = aspect_ratio(points)
        self.assertGreater(ratio, 0.0)

    def test_eye_ratio_zero_width(self):
        points = [(1, 1)] * 6
        self.assertEqual(aspect_ratio(points), 0.0)

if __name__ == "__main__":
    unittest.main()
