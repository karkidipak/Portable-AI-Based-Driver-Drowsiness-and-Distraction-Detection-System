
import os
import time
import math
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# =========================================================
# AUTONOMOUS - PORTABLE AI DRIVER MONITORING SYSTEM
# =========================================================

CAMERA_INDEX = 0
FACE_MODEL_PATH = "face_landmarker.task"
YOLO_MODEL_PATH = "yolo11n.pt"

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

EYE_THRESHOLD = 0.18
EYE_CLOSED_SECONDS = 2.0
MOUTH_THRESHOLD = 0.35
YAWN_SECONDS = 1.0

YAW_THRESHOLD = 20.0
PITCH_THRESHOLD = 15.0
DISTRACTION_SECONDS = 2.5

PHONE_CONFIDENCE = 0.35
PHONE_CLASS_ID = 67
PHONE_EVERY_N_FRAMES = 4

CYAN = (0, 255, 220)
MESH_COLOR = (180, 150, 20)
WHITE = (245, 245, 245)
GREEN = (70, 220, 100)
YELLOW = (0, 220, 255)
RED = (40, 40, 255)
DARK = (18, 24, 28)

WINDOW_NAME = "AUTONOMOUS - Driver Monitoring"


def log(message):
    print(message, flush=True)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def opening_ratio(points, horizontal, vertical_pairs):
    width = distance(points[horizontal[0]], points[horizontal[1]])
    if width < 1:
        return 0.0

    height = sum(
        distance(points[a], points[b])
        for a, b in vertical_pairs
    )
    return height / (2.0 * width)


def put_text(frame, text, position, color=WHITE,
             scale=0.5, thickness=1):
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA
    )


# ---------------------------------------------------------
# DENSE TRIANGULAR FACE MESH
# ---------------------------------------------------------

def draw_face_mesh(frame, points):
    """
    Draw a dense cyan triangular mesh from facial landmarks.
    Delaunay triangulation is an approximation of the reference image.
    """
    height, width = frame.shape[:2]

    if len(points) < 20:
        return

    subdiv = cv2.Subdiv2D((0, 0, width, height))
    valid_points = []
    seen = set()

    for x, y in points:
        x = int(np.clip(x, 1, width - 2))
        y = int(np.clip(y, 1, height - 2))

        if (x, y) in seen:
            continue

        seen.add((x, y))
        valid_points.append((x, y))

    for point in valid_points:
        try:
            subdiv.insert(point)
        except cv2.error:
            pass

    # Draw triangulation lines.
    for triangle in subdiv.getTriangleList():
        vertices = [
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[2]), int(triangle[3])),
            (int(triangle[4]), int(triangle[5]))
        ]

        if all(
            0 <= x < width and 0 <= y < height
            for x, y in vertices
        ):
            cv2.line(
                frame, vertices[0], vertices[1],
                MESH_COLOR, 1, cv2.LINE_AA
            )
            cv2.line(
                frame, vertices[1], vertices[2],
                MESH_COLOR, 1, cv2.LINE_AA
            )
            cv2.line(
                frame, vertices[2], vertices[0],
                MESH_COLOR, 1, cv2.LINE_AA
            )

    # Draw landmark dots.
    for x, y in valid_points:
        cv2.circle(frame, (x, y), 1, CYAN, -1, cv2.LINE_AA)

    # Highlight eye, nose, and mouth landmarks.
    for index in (33, 133, 263, 362, 1, 13, 14, 61, 291):
        if index < len(points):
            cv2.circle(frame, points[index], 2, CYAN, -1)


# ---------------------------------------------------------
# HEAD POSE ESTIMATION
# ---------------------------------------------------------

def estimate_head_pose(points, width, height):
    """
    Estimate head rotation using solvePnP.
    Returns approximate yaw and pitch angles in degrees.
    """
    # Generic 3D face model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    # Matching MediaPipe landmark indices.
    indices = [1, 152, 33, 263, 61, 291]

    image_points = np.array(
        [points[i] for i in indices],
        dtype=np.float64
    )

    focal_length = float(width)

    camera_matrix = np.array([
        [focal_length, 0, width / 2.0],
        [0, focal_length, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Euler angles from the rotation matrix.
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = float(angles[0])
        yaw = float(angles[1])

        return yaw, pitch

    except (cv2.error, ValueError, IndexError):
        return None, None


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

def draw_dashboard(frame, statuses, alerts):
    height, width = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (430, 190), DARK, -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    put_text(
        frame, "AUTONOMOUS // MACHINE VISION",
        (22, 35), CYAN, 0.55, 2
    )
    put_text(
        frame, "DRIVER MONITORING - PROTOTYPE",
        (22, 57), WHITE, 0.43
    )

    y = 82
    for label, value, color in statuses:
        put_text(
            frame, f"{label}: {value}",
            (22, y), color, 0.43
        )
        y += 19

    if alerts:
        cv2.rectangle(
            frame, (10, height - 52),
            (width - 10, height - 10), RED, 2
        )
        put_text(
            frame, " | ".join(alerts)[:110],
            (20, height - 25), RED, 0.48, 2
        )


# ---------------------------------------------------------
# CAMERA SETUP
# ---------------------------------------------------------

def open_camera(index):
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", None)
    ]

    for name, backend in backends:
        log(f"[INFO] Trying camera {index} with {name}...")

        camera = None

        try:
            if backend is None:
                camera = cv2.VideoCapture(index)
            else:
                camera = cv2.VideoCapture(index, backend)

            if camera.isOpened():
                log(f"[INFO] Camera device opened using {name}.")
                # Do not force resolution or buffer-size changes.
                return camera

        except cv2.error as exc:
            log(f"[WARNING] Camera backend error: {exc}")

        if camera is not None:
            camera.release()

    return None


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

def main():
    if not os.path.isfile(FACE_MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find {FACE_MODEL_PATH} in the project folder."
        )

    log("[INFO] Loading MediaPipe Face Landmarker...")

    vision = mp.tasks.vision
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=FACE_MODEL_PATH
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    face_model = vision.FaceLandmarker.create_from_options(
        face_options
    )

    camera = None

    try:
        log("[INFO] Loading YOLO phone detector...")
        phone_model = YOLO(YOLO_MODEL_PATH)
        log("[INFO] Models loaded.")

        log("[INFO] Opening camera now...")
        camera = open_camera(CAMERA_INDEX)

        if camera is None:
            raise RuntimeError(
                "Cannot open the camera. Check that DroidCam is running."
            )

        log("[INFO] Camera ready.")
        log("[INFO] Face the camera for calibration.")
        log("[INFO] Press Q or Esc in the video window to exit.")

        program_start = time.perf_counter()
        last_timestamp = -1
        frame_count = 0

        # Calibration starts when a face is first detected.
        calibration_start = None
        calibration_samples = []
        neutral_yaw = None
        neutral_pitch = None
        calibrated = False

        eyes_closed_start = None
        mouth_open_start = None
        look_away_start = None

        phone_boxes = []
        phone_detected = False

        while True:
            try:
                ok, raw_frame = camera.read()
            except cv2.error as exc:
                log(f"[ERROR] Camera frame read failed: {exc}")
                log(
                    "[INFO] Close other camera apps, restart DroidCam, "
                    "and run the program again."
                )
                break

            if not ok or raw_frame is None:
                log("[WARNING] Empty camera frame. Retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1

            try:
                frame = cv2.resize(
                    raw_frame, (FRAME_WIDTH, FRAME_HEIGHT)
                )
            except cv2.error as exc:
                log(f"[ERROR] Could not resize camera frame: {exc}")
                break

            # Mirror the view like a webcam preview.
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            now = time.perf_counter()
            elapsed = now - program_start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(elapsed * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = face_model.detect_for_video(
                mp_image, timestamp
            )

            found_face = bool(result.face_landmarks)
            alerts = []

            eye_status = "NOT TRACKED"
            mouth_status = "NOT TRACKED"
            head_status = "NOT TRACKED"
            yaw = None
            pitch = None

            if found_face:
                face = result.face_landmarks[0]

                points = [
                    (
                        int(p.x * width),
                        int(p.y * height)
                    )
                    for p in face
                ]

                # Dense triangular mesh similar to the reference.
                draw_face_mesh(frame, points)

                put_text(
                    frame, "FACE TRACKED",
                    (18, height - 68), CYAN, 0.48
                )

                # Eye opening estimates.
                left_eye = opening_ratio(
                    points, (33, 133),
                    ((160, 144), (158, 153))
                )

                right_eye = opening_ratio(
                    points, (362, 263),
                    ((385, 380), (387, 373))
                )

                eye_ratio = (left_eye + right_eye) / 2.0

                if eye_ratio < EYE_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = now

                    closed_time = now - eyes_closed_start
                    eye_status = f"CLOSED {closed_time:.1f}s"

                    if closed_time >= EYE_CLOSED_SECONDS:
                        alerts.append("POSSIBLE DROWSINESS")
                else:
                    eyes_closed_start = None
                    eye_status = f"OPEN {eye_ratio:.2f}"

                # Mouth opening estimate; not a confirmed yawn classifier.
                mouth_ratio = opening_ratio(
                    points, (61, 291),
                    ((13, 14), (82, 87))
                )

                if mouth_ratio > MOUTH_THRESHOLD:
                    if mouth_open_start is None:
                        mouth_open_start = now

                    mouth_time = now - mouth_open_start
                    mouth_status = f"OPEN {mouth_time:.1f}s"

                    if mouth_time >= YAWN_SECONDS:
                        alerts.append("POSSIBLE YAWN")
                else:
                    mouth_open_start = None
                    mouth_status = f"NORMAL {mouth_ratio:.2f}"

                # Estimate head rotation.
                yaw, pitch = estimate_head_pose(
                    points, width, height
                )

                if yaw is None:
                    head_status = "POSE UNAVAILABLE"
                else:
                    if calibration_start is None:
                        calibration_start = now

                    if not calibrated:
                        if now - calibration_start < 2.0:
                            calibration_samples.append((yaw, pitch))
                            head_status = "CALIBRATING - LOOK FORWARD"
                        elif calibration_samples:
                            neutral_yaw = float(np.mean(
                                [sample[0] for sample in calibration_samples]
                            ))
                            neutral_pitch = float(np.mean(
                                [sample[1] for sample in calibration_samples]
                            ))
                            calibrated = True
                            head_status = "CALIBRATION COMPLETE"
                        else:
                            head_status = "WAITING FOR CALIBRATION"

                    if calibrated:
                        yaw_change = yaw - neutral_yaw
                        pitch_change = pitch - neutral_pitch

                        looking_away = (
                            abs(yaw_change) > YAW_THRESHOLD
                            or abs(pitch_change) > PITCH_THRESHOLD
                        )

                        if looking_away:
                            if look_away_start is None:
                                look_away_start = now

                            away_time = now - look_away_start
                            head_status = (
                                f"AWAY {away_time:.1f}s "
                                f"Y:{yaw_change:.0f} "
                                f"P:{pitch_change:.0f}"
                            )

                            if away_time >= DISTRACTION_SECONDS:
                                alerts.append("HEAD TURN DISTRACTION")
                        else:
                            look_away_start = None
                            head_status = (
                                f"FORWARD Y:{yaw_change:.0f} "
                                f"P:{pitch_change:.0f}"
                            )

            else:
                put_text(
                    frame, "NO FACE DETECTED",
                    (20, 220), YELLOW, 0.7, 2
                )
                eyes_closed_start = None
                mouth_open_start = None
                look_away_start = None
                calibration_start = None
                calibration_samples = []
                calibrated = False
                neutral_yaw = None
                neutral_pitch = None

            # Run YOLO phone detection every few frames.
            if frame_count % PHONE_EVERY_N_FRAMES == 0 or frame_count == 1:
                phone_boxes = []
                phone_detected = False

                try:
                    predictions = phone_model.predict(
                        frame,
                        verbose=False,
                        conf=PHONE_CONFIDENCE,
                        imgsz=640
                    )

                    if predictions and predictions[0].boxes is not None:
                        for box in predictions[0].boxes:
                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())

                            if (
                                class_id == PHONE_CLASS_ID
                                and confidence >= PHONE_CONFIDENCE
                            ):
                                x1, y1, x2, y2 = [
                                    int(value)
                                    for value in box.xyxy[0].tolist()
                                ]

                                phone_boxes.append(
                                    (x1, y1, x2, y2, confidence)
                                )
                                phone_detected = True

                except Exception as exc:
                    log(f"[WARNING] Phone detector error: {exc}")

            for x1, y1, x2, y2, confidence in phone_boxes:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), RED, 2
                )
                put_text(
                    frame, f"PHONE {confidence:.2f}",
                    (x1, max(20, y1 - 7)), RED, 0.5, 2
                )

            if phone_detected:
                alerts.append("PHONE DETECTED")

            statuses = [
                (
                    "FACE",
                    "TRACKED" if found_face else "NOT FOUND",
                    GREEN if found_face else YELLOW
                ),
                (
                    "EYES",
                    eye_status,
                    RED if "CLOSED" in eye_status else GREEN
                ),
                (
                    "MOUTH",
                    mouth_status,
                    YELLOW if "OPEN" in mouth_status else GREEN
                ),
                (
                    "HEAD",
                    head_status,
                    YELLOW if "AWAY" in head_status else GREEN
                ),
                (
                    "PHONE",
                    "DETECTED" if phone_detected else "NOT DETECTED",
                    RED if phone_detected else GREEN
                )
            ]

            draw_dashboard(frame, statuses, alerts)

            try:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as exc:
                log(f"[ERROR] Display error: {exc}")
                break

            if key in (ord("q"), 27):
                break

            if frame_count % 120 == 0:
                log(
                    f"[INFO] Frames={frame_count}, "
                    f"face_found={found_face}"
                )

    finally:
        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()
        face_model.close()
        log("[INFO] Camera and models released.")


if __name__ == "__main__":
    main()
