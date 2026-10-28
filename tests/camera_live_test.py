
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Camera failed to open.")
    raise SystemExit(1)

print("Camera connected. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame.")
        break

    cv2.imshow("DroidCam Live Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
