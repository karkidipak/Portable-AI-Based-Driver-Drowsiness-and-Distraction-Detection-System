
import cv2
import mediapipe as mp
import math
import time
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError("face_landmarker.task is missing.")

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# Face Landmarker mouth landmarks
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    raise RuntimeError("Could not open DroidCam.")

open_since = None
timestamp_ms = 0

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    print("Yawning test started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read camera frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
        result = landmarker.detect_for_video(image, timestamp_ms)

        status = "NO FACE"
        color = (0, 0, 255)

        if result.face_landmarks:
            face = result.face_landmarks[0]

            mouth_height = distance(face[TOP_LIP], face[BOTTOM_LIP])
            mouth_width = distance(face[LEFT_MOUTH], face[RIGHT_MOUTH])
            ratio = mouth_height / mouth_width if mouth_width else 0

            # Starting threshold for testing; calibration is required.
            mouth_open = ratio > 0.35

            if mouth_open:
                if open_since is None:
                    open_since = time.monotonic()

                duration = time.monotonic() - open_since

                if duration >= 1.0:
                    status = f"POSSIBLE YAWN ({duration:.1f}s)"
                    color = (0, 0, 255)
                else:
                    status = f"MOUTH OPEN ({duration:.1f}s)"
                    color = (0, 165, 255)
            else:
                open_since = None
                status = f"MOUTH CLOSED | Ratio: {ratio:.3f}"
                color = (0, 255, 0)

        cv2.putText(
            frame, status, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )
        cv2.imshow("AUTONOMOUS - Yawning Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
