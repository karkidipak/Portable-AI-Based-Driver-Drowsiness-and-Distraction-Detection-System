import os
import time
import math
import asyncio
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent
FACE_MODEL_PATH = PROJECT_ROOT / "Function_Tests" / "face_landmarker.task"
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"

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

GREEN = (70, 220, 100)
MESH_COLOR = (50, 225, 50)
WHITE = (245, 245, 245)
RED = (40, 40, 255)
DARK = (18, 24, 28)

app = FastAPI(
    title="AUTONOMOUS Driver Monitoring API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared server state
state_lock = threading.Lock()
camera_connected = False
ai_ready = False
latest_jpeg = None
active_connections = set()
main_loop = None

class AIEngineWorker(threading.Thread):
    def __init__(self, camera_source=DEFAULT_CAMERA, enable_phone=True):
        super().__init__()
        self.daemon = True
        self.camera_source = camera_source
        self.enable_phone = enable_phone
        self.running = True

    def run(self):
        global camera_connected, ai_ready, latest_jpeg

        print(f"[INFO] Loading MediaPipe Face Landmarker from {FACE_MODEL_PATH}...", flush=True)
        if not FACE_MODEL_PATH.is_file():
            print(f"[ERROR] Face model not found at {FACE_MODEL_PATH}", flush=True)
            return

        vision = mp.tasks.vision
        face_options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_model = vision.FaceLandmarker.create_from_options(face_options)

        phone_model = None
        if self.enable_phone and YOLO_MODEL_PATH.is_file():
            print(f"[INFO] Loading YOLO model from {YOLO_MODEL_PATH}...", flush=True)
            try:
                phone_model = YOLO(str(YOLO_MODEL_PATH))
                print("[INFO] YOLO model loaded.", flush=True)
            except Exception as e:
                print(f"[WARNING] Could not load YOLO model: {e}", flush=True)

        print(f"[INFO] Opening camera source: {self.camera_source}", flush=True)
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera {self.camera_source}", flush=True)
            with state_lock:
                camera_connected = False
                ai_ready = True
            return

        with state_lock:
            camera_connected = True
            ai_ready = True

        print("[INFO] AI Engine & Camera worker running.", flush=True)

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

        last_drowsy_emit = 0.0
        last_distraction_emit = 0.0

        def distance(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def opening_ratio(points, horizontal, vertical_pairs):
            width = distance(points[horizontal[0]], points[horizontal[1]])
            if width < 1:
                return 0.0
            height = sum(distance(points[a], points[b]) for a, b in vertical_pairs)
            return height / (2.0 * width)

        def put_text(frame, text, position, color=WHITE, scale=0.5, thickness=1):
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

        def estimate_head_pose(points, width, height):
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ], dtype=np.float64)
            indices = [1, 152, 33, 263, 61, 291]
            image_points = np.array([points[i] for i in indices], dtype=np.float64)
            focal_length = float(width)
            camera_matrix = np.array([
                [focal_length, 0, width / 2.0],
                [0, focal_length, height / 2.0],
                [0, 0, 1]
            ], dtype=np.float64)
            distortion = np.zeros((4, 1), dtype=np.float64)
            try:
                success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    return None, None
                rmat, _ = cv2.Rodrigues(rvec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                return float(angles[1]), float(angles[0])
            except Exception:
                return None, None

        while self.running:
            ok, raw_frame = cap.read()
            if not ok or raw_frame is None:
                with state_lock:
                    camera_connected = False
                time.sleep(0.1)
                continue

            with state_lock:
                camera_connected = True

            frame_count += 1
            try:
                frame = cv2.resize(raw_frame, (FRAME_WIDTH, FRAME_HEIGHT))
            except Exception:
                continue

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.perf_counter()
            elapsed = now - program_start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp = int(elapsed * 1000)
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            result = face_model.detect_for_video(mp_image, timestamp)
            found_face = bool(result.face_landmarks)
            alerts = []

            eye_status = "NOT TRACKED"
            mouth_status = "NOT TRACKED"
            head_status = "NOT TRACKED"

            if found_face:
                face = result.face_landmarks[0]
                points = [(int(p.x * width), int(p.y * height)) for p in face]

                left_eye = opening_ratio(points, (33, 133), ((160, 144), (158, 153)))
                right_eye = opening_ratio(points, (362, 263), ((385, 380), (387, 373)))
                eye_ratio = (left_eye + right_eye) / 2.0

                if eye_ratio < EYE_THRESHOLD:
                    if eyes_closed_start is None:
                        eyes_closed_start = now
                    closed_time = now - eyes_closed_start
                    eye_status = f"CLOSED {closed_time:.1f}s"
                    if closed_time >= EYE_CLOSED_SECONDS:
                        alerts.append("POSSIBLE DROWSINESS")
                        if closed_time >= 5.0 and (now - last_drowsy_emit) > 10.0:
                            last_drowsy_emit = now
                            broadcast_event({
                                "event": "Drowsiness",
                                "timestamp": datetime.now().isoformat(),
                                "duration": round(closed_time, 1),
                                "details": f"Sustained eye closure for {closed_time:.1f}s"
                            })
                else:
                    eyes_closed_start = None
                    eye_status = f"OPEN {eye_ratio:.2f}"

                mouth_ratio = opening_ratio(points, (61, 291), ((13, 14), (82, 87)))
                if mouth_ratio > MOUTH_THRESHOLD:
                    if mouth_open_start is None:
                        mouth_open_start = now
                    mouth_time = now - mouth_open_start
                    mouth_status = f"OPEN {mouth_time:.1f}s"
                    if mouth_time >= YAWN_SECONDS:
                        alerts.append("POSSIBLE YAWN")
                        if mouth_time >= 5.0:
                            broadcast_event({
                                "event": "Yawning",
                                "timestamp": datetime.now().isoformat(),
                                "duration": round(mouth_time, 1),
                                "details": f"Sustained yawning for {mouth_time:.1f}s"
                            })
                else:
                    mouth_open_start = None
                    mouth_status = f"NORMAL {mouth_ratio:.2f}"

                yaw, pitch = estimate_head_pose(points, width, height)
                if yaw is not None:
                    if calibration_start is None:
                        calibration_start = now
                    if not calibrated:
                        if now - calibration_start < 2.0:
                            calibration_samples.append((yaw, pitch))
                            head_status = "CALIBRATING"
                        elif calibration_samples:
                            neutral_yaw = float(np.mean([s[0] for s in calibration_samples]))
                            neutral_pitch = float(np.mean([s[1] for s in calibration_samples]))
                            calibrated = True
                    if calibrated:
                        yaw_change = yaw - neutral_yaw
                        pitch_change = pitch - neutral_pitch
                        looking_away = abs(yaw_change) > YAW_THRESHOLD or abs(pitch_change) > PITCH_THRESHOLD
                        if looking_away:
                            if look_away_start is None:
                                look_away_start = now
                            away_time = now - look_away_start
                            head_status = f"AWAY {away_time:.1f}s"
                            if away_time >= DISTRACTION_SECONDS:
                                alerts.append("HEAD TURN DISTRACTION")
                                if away_time >= 5.0 and (now - last_distraction_emit) > 10.0:
                                    last_distraction_emit = now
                                    broadcast_event({
                                        "event": "Head turn distraction",
                                        "timestamp": datetime.now().isoformat(),
                                        "duration": round(away_time, 1),
                                        "details": f"Looking away for {away_time:.1f}s"
                                    })
                        else:
                            look_away_start = None
                            head_status = "FORWARD"
            else:
                eyes_closed_start = None
                mouth_open_start = None
                look_away_start = None

            phone_detected = False
            if phone_model is not None and (frame_count % PHONE_EVERY_N_FRAMES == 0):
                try:
                    preds = phone_model.predict(frame, verbose=False, conf=PHONE_CONFIDENCE, imgsz=640)
                    if preds and preds[0].boxes is not None:
                        for box in preds[0].boxes:
                            if int(box.cls[0].item()) == PHONE_CLASS_ID and float(box.conf[0].item()) >= PHONE_CONFIDENCE:
                                phone_detected = True
                                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
                                put_text(frame, f"PHONE", (x1, max(20, y1 - 7)), RED, 0.5, 2)
                except Exception:
                    pass

            if phone_detected:
                alerts.append("PHONE DETECTED")

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (410, 170), DARK, -1)
            cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
            put_text(frame, "AUTONOMOUS // AI MONITORING", (22, 35), GREEN, 0.55, 2)
            y_pos = 65
            for lbl, val, col in [
                ("FACE", "TRACKED" if found_face else "NOT FOUND", GREEN if found_face else RED),
                ("EYES", eye_status, GREEN),
                ("MOUTH", mouth_status, GREEN),
                ("HEAD", head_status, GREEN),
                ("PHONE", "DETECTED" if phone_detected else "CLEAR", RED if phone_detected else GREEN)
            ]:
                put_text(frame, f"{lbl}: {val}", (22, y_pos), col, 0.43)
                y_pos = y_pos + 20

            if alerts:
                cv2.rectangle(frame, (10, height - 45), (width - 10, height - 10), RED, 2)
                put_text(frame, " | ".join(alerts), (20, height - 20), RED, 0.48, 2)

            success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                with state_lock:
                    latest_jpeg = encoded.tobytes()

        cap.release()
        face_model.close()

def broadcast_event(event_dict):
    if not active_connections:
        return
    import json
    msg = json.dumps(event_dict)
    dead = set()
    for conn in list(active_connections):
        try:
            if main_loop and main_loop.is_running():
                asyncio.run_coroutine_threadsafe(conn.send_text(msg), main_loop)
        except Exception:
            dead.add(conn)
    active_connections.difference_update(dead)

@app.on_event("startup")
def startup_event():
    global main_loop
    try:
        main_loop = asyncio.get_running_loop()
    except Exception:
        main_loop = asyncio.get_event_loop()

    worker = AIEngineWorker(camera_source=DEFAULT_CAMERA, enable_phone=True)
    worker.start()
    print("[INFO] AI Engine worker started in background.", flush=True)

@app.get("/")
def home():
    return {
        "status": "connected",
        "device": "AUTONOMOUS",
        "message": "Python FastAPI server is running with AI engine"
    }

@app.get("/health")
def health():
    with state_lock:
        cam_ok = camera_connected
        ai_ok = ai_ready
    return {
        "status": "ok" if cam_ok else "degraded",
        "connection": "active",
        "camera": "online" if cam_ok else "offline",
        "ai_engine": "ready" if ai_ok else "not_ready",
        "timestamp": datetime.now().isoformat()
    }

def generate_mjpeg():
    while True:
        frame_bytes = None
        with state_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is not None:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                frame_bytes + b'\r\n'
            )
        else:
            time.sleep(0.05)
        time.sleep(0.03)

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    print("[INFO] Flutter WebSocket connected.", flush=True)

    try:
        while True:
            await asyncio.sleep(20)
            await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, ConnectionResetError):
        print("[INFO] Flutter WebSocket disconnected.", flush=True)
    except Exception as e:
        print(f"[WARNING] WebSocket error: {e}", flush=True)
    finally:
        active_connections.discard(websocket)
