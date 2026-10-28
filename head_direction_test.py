
import cv2
import mediapipe as mp
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

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    raise RuntimeError("Could not open DroidCam.")

timestamp_ms = 0

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    print("Head-direction test started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read camera frame.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = max(timestamp_ms + 1, int(time.monotonic() * 1000))
        result = landmarker.detect_for_video(image, timestamp_ms)

        status = "NO FACE"
        color = (0, 0, 255)

        if result.face_landmarks:
            face = result.face_landmarks[0]

            # Approximate head pose using nose position relative to eye corners.
            nose = face[1]
            left_eye = face[33]
            right_eye = face[263]
            forehead = face[10]
            chin = face[152]

            eye_center_x = (left_eye.x + right_eye.x) / 2
            eye_width = abs(right_eye.x - left_eye.x)
            eye_center_y = (left_eye.y + right_eye.y) / 2
            face_height = abs(chin.y - forehead.y)

            horizontal = (nose.x - eye_center_x) / max(eye_width, 0.001)
            vertical = (nose.y - eye_center_y) / max(face_height, 0.001)

            # These are rough starting thresholds, not calibrated angles.
            if horizontal < -0.12:
                status = "LOOKING LEFT (estimate)"
            elif horizontal > 0.12:
                status = "LOOKING RIGHT (estimate)"
            elif vertical < -0.10:
                status = "LOOKING UP (estimate)"
            elif vertical > 0.10:
                status = "LOOKING DOWN (estimate)"
            else:
                status = "FACING FORWARD (estimate)"

            color = (0, 255, 0)

            cv2.putText(
                frame,
                f"H: {horizontal:.2f}  V: {vertical:.2f}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        cv2.putText(
            frame, status, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        cv2.imshow("AUTONOMOUS - Head Direction Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
