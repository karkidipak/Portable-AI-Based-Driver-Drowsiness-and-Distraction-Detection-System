"""
Main entry point for the AUTONOMOUS driver-monitoring prototype.

The full implementation uses MediaPipe Tasks API, which is compatible
with the MediaPipe package used in the Python 3.13 environment.
Run from the project root with: python src/main.py
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def distance(point1, point2):
    """Return Euclidean distance between two 2D points."""
    return float(math.hypot(
        point2[0] - point1[0],
        point2[1] - point1[1]
    ))


def aspect_ratio(points):
    """Calculate the eye aspect ratio from six ordered eye landmarks."""
    if len(points) != 6:
        raise ValueError("aspect_ratio requires exactly 6 points")

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


from Function_Tests.driver_monitor import main


if __name__ == "__main__":
    main()
