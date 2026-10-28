
import cv2
import time
from ultralytics import YOLO

# Downloads the pretrained model on first use if not cached.
# Internet is needed for the initial download.
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    raise RuntimeError("Could not open DroidCam.")

print("Phone detection started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    results = model.predict(
        source=frame,
        classes=[67],  # COCO class 67 = cell phone
        conf=0.35,
        verbose=False
    )

    phone_found = False

    for result in results:
        for box in result.boxes:
            phone_found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            cv2.putText(
                frame, f"Phone {confidence:.0%}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    status = "PHONE DETECTED" if phone_found else "NO PHONE DETECTED"
    color = (0, 0, 255) if phone_found else (0, 255, 0)

    cv2.putText(
        frame, status, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )

    cv2.imshow("AUTONOMOUS - Phone Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
