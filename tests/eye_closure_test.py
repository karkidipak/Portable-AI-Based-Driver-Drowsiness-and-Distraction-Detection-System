
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

# MediaPipe Face Mesh landmark indices for each eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def eye_ratio(landmarks, indices):
    p = [landmarks[i] for i in indices]
    vertical = distance(p[1], p[5]) + distance(p[2], p[4])
    horizontal = 2 * distance(p[0], p[3])
    return vertical / horizontal if horizontal else 0

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    raise RuntimeError("Could not open DroidCam.")

closed_since = None
timestamp_ms = 0

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    print("Eye test started. Press Q to quit.")
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
            left = eye_ratio(face, LEFT_EYE)
            right = eye_ratio(face, RIGHT_EYE)
            avg_ratio = (left + right) / 2

            # Initial test threshold; tune using your own camera.
            eyes_closed = avg_ratio < 0.18

            if eyes_closed:
                if closed_since is None:
                    closed_since = time.monotonic()
                duration = time.monotonic() - closed_since
                status = f"EYES CLOSED ({duration:.1f}s)"
                color = (0, 165, 255)

                if duration >= 2.0:
                    status = "PROLONGED EYE CLOSURE"
                    color = (0, 0, 255)
            else:
                closed_since = None
                status = f"EYES OPEN | Ratio: {avg_ratio:.3f}"
                color = (0, 255, 0)

        cv2.putText(
            frame, status, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )
        cv2.imshow("AUTONOMOUS - Eye Closure Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
