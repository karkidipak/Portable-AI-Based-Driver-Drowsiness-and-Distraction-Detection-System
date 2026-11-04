import os
import time
import math
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# =========================================================
# AUTONOMOUS - PORTABLE AI DRIVER MONITORING SYSTEM
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL_PATH = PROJECT_ROOT / "Function_Tests" / "face_landmarker.task"
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
#DEFAULT_CAMERA = "http://192.168.137.2:4747/video"
DEFAULT_CAMERA = 0

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

#CYAN = (0, 255, 220)
# Change to a bright vibrant green
GREEN = (70, 220, 100)
MESH_COLOR = (50, 225, 50)
WHITE = (245, 245, 245)
#YELLOW = (0, 220, 255)
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

# ---------------------------------------------------------
# GEOMETRIC KEYPOINT FACE MESH (HUD STYLE)
# ---------------------------------------------------------

def draw_face_mesh(frame, points):
    """
    Draw a clean geometric wireframe connecting key facial landmarks
    matching the reference HUD style.
    """
    if len(points) < 468:
        return

    # Draw specific geometric connective lines resembling the HUD wireframe
    connections = [
        # Brow / Forehead outline
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (10, 109), (109, 67), (67, 103), (103, 54), (54, 21), (21, 162),
        # Eye connections to nose bridge
        (33, 133), (33, 168), (133, 168), (168, 6), (6, 1),
        (362, 263), (263, 168), (362, 168),
        # Nose to mouth
        (1, 2), (2, 98), (98, 327), (1, 4), (4, 5),
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
        (291, 375), (375, 321), (321, 405), (405, 314),
        # Jawline connections from cheeks
        (234, 127), (127, 162), (234, 93), (93, 132), (132, 58), (58, 172), (172, 152),
        (454, 356), (356, 389), (454, 323), (323, 361), (361, 288), (288, 397), (397, 152),
        # Cross facial mesh lines
        (33, 61), (263, 291), (133, 61), (362, 291), (1, 61), (1, 291)
    ]

    for start_idx, end_idx in connections:
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], MESH_COLOR, 1, cv2.LINE_AA)

    # Draw prominent nodes (dots) at key intersections
    highlight_indices = [
        10, 168, 6, 1, 4, 
        33, 133, 362, 263, 
        61, 291, 13, 14, 
        234, 454, 152
    ]
    
    for idx in highlight_indices:
        if idx < len(points):
            cv2.circle(frame, points[idx], 3, GREEN, -1, cv2.LINE_AA)

# ---------------------------------------------------------
# HEAD POSE ESTIMATION
# ---------------------------------------------------------

def estimate_head_pose(points, width, height):
    """
    Estimate head rotation using solvePnP.
    Returns approximate yaw and pitch angles in degrees.
    """
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye corner
        (225.0, 170.0, -135.0),     # Right eye corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)

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
        (22, 35), GREEN, 0.55, 2
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

class StreamWrapper:
    """Reader for the iOS DroidCam MJPEG HTTP stream using OpenCV VideoCapture."""

    def __init__(self, url):
        self.url = url
        self.cap = None
        self.is_open = False

        try:
            log(f"[INFO] Connecting to iOS DroidCam stream: {self.url}")
            
            # Force FFmpeg backend to use TCP transport for stability over local HTTP streams
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.url)

            if self.cap.isOpened():
                self.is_open = True
                log("[INFO] DroidCam stream connected.")
            else:
                log("[ERROR] OpenCV could not open the DroidCam URL stream.")
        except Exception as exc:
            log(f"[ERROR] Could not connect to DroidCam: {exc}")
            self.is_open = False

    def isOpened(self):
        return self.is_open and self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        self.is_open = False
        if self.cap is not None:
            self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Portable AI driver drowsiness and distraction monitor"
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="Camera index (0, 1, ...) or DroidCam HTTP stream URL"
    )
    parser.add_argument(
        "--no-phone",
        action="store_true",
        help="Disable YOLO phone detection"
    )
    return parser.parse_args()


def open_camera(source):
    if isinstance(source, str) and source.lower().startswith(("http://", "https://")):
        stream = StreamWrapper(source)
        return stream if stream.isOpened() else None

    try:
        camera_index = int(source)
    except (TypeError, ValueError):
        camera_index = source

    backends = [
        ("FFmpeg", cv2.CAP_FFMPEG),
        ("Media Foundation", cv2.CAP_MSMF),
        ("DirectShow", cv2.CAP_DSHOW),
        ("Default", None)
    ]

    for name, backend in backends:
        log(f"[INFO] Trying camera {camera_index} with {name}...")
        camera = None
        try:
            camera = (
                cv2.VideoCapture(camera_index)
                if backend is None
                else cv2.VideoCapture(camera_index, backend)
            )
            if camera.isOpened():
                log(f"[INFO] Camera opened using {name}.")
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
    args = parse_args()

    if not FACE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find the MediaPipe model: {FACE_MODEL_PATH}"
        )

    if not args.no_phone and not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find the YOLO model: {YOLO_MODEL_PATH}. "
            "Use --no-phone to run without phone detection."
        )

    log(f"[INFO] Python MediaPipe version: {mp.__version__}")
    log("[INFO] Loading MediaPipe Face Landmarker...")

    vision = mp.tasks.vision
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(FACE_MODEL_PATH)
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
        phone_model = None
        if not args.no_phone:
            log("[INFO] Loading YOLO phone detector...")
            phone_model = YOLO(str(YOLO_MODEL_PATH))
            log("[INFO] YOLO phone detector loaded.")
        else:
            log("[INFO] Phone detection disabled by --no-phone.")

        log(f"[INFO] Opening camera: {args.camera}")
        camera = open_camera(args.camera)

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

                draw_face_mesh(frame, points)

                put_text(
                    frame, "FACE TRACKED",
                    (18, height - 68), GREEN, 0.48
                )

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
                    (20, 220), GREEN, 0.7, 2
                )
                eyes_closed_start = None
                mouth_open_start = None
                look_away_start = None
                calibration_start = None
                calibration_samples = []
                calibrated = False
                neutral_yaw = None
                neutral_pitch = None

            if phone_model is not None and (
                frame_count % PHONE_EVERY_N_FRAMES == 0 or frame_count == 1
            ):
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
                    GREEN if found_face else RED
                ),
                (
                    "EYES",
                    eye_status,
                    RED if "CLOSED" in eye_status else GREEN
                ),
                (
                    "MOUTH",
                    mouth_status,
                    RED if "OPEN" in mouth_status else GREEN
                ),
                (
                    "HEAD",
                    head_status,
                    RED if "AWAY" in head_status else GREEN
                ),
                (
                    "PHONE",
                    ("DISABLED" if phone_model is None else
                     ("DETECTED" if phone_detected else "NOT DETECTED")),
                    RED if phone_model is None else (RED if phone_detected else GREEN)
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