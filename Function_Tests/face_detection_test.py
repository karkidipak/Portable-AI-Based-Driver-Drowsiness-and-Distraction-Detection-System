
import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Download face_landmarker.task into this project folder."
    )

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("ERROR: Camera could not open.")
    raise SystemExit(1)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    print("Face landmark test started. Press Q to quit.")
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to receive camera frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms += 33
        result = landmarker.detect_for_video(image, timestamp_ms)

        if result.face_landmarks:
            h, w = frame.shape[:2]

            for face in result.face_landmarks:
                for point in face:
                    x = int(point.x * w)
                    y = int(point.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            cv2.putText(
                frame, "FACE LANDMARKS DETECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                frame, "NO FACE DETECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        cv2.imshow("AUTONOMOUS - Face Landmark Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
