import cv2
import sys
import time
import argparse
import threading
import numpy as np
import requests
import mediapipe as mp

import math
def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return float(math.hypot(
        point2[0] - point1[0],
        point2[1] - point1[1]
    ))


def aspect_ratio(points):
    """Calculate an eye aspect ratio from six landmark points."""
    if len(points) != 6:
        raise ValueError("aspect_ratio requires exactly 6 points")

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def parse_args():
    parser = argparse.ArgumentParser(description="Portable AI Driver Drowsiness and Distraction Detection System")
    parser.add_argument("--camera", type=str, default="http://192.168.137.69:4747/video", help="Camera index or network stream URL")
    parser.add_argument("--no-phone", action="store_true", help="Run without phone detection model")
    return parser.parse_args()

class StreamWrapper:
    """Robust Thread-Buffered MJPEG Stream Wrapper for iOS DroidCam"""
    def __init__(self, url):
        if "4747" in url and not url.endswith("/video"):
            if not url.endswith("/"):
                url += "/video"
            else:
                url += "video"
        self.url = url
        self.frame = None
        self.is_open = False
        self.lock = threading.Lock()
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        try:
            print(f"[INFO] Connecting to iOS DroidCam stream: {self.url}")
            self.stream_response = requests.get(self.url, headers=headers, stream=True, timeout=10)
            if self.stream_response.status_code == 200:
                self.is_open = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                print("[INFO] Background stream thread started successfully!")
            else:
                print(f"[ERROR] Stream returned status: {self.stream_response.status_code}")
        except Exception as e:
            print(f"[ERROR] Connection exception: {e}")
            self.is_open = False

    def _update(self):
        bytes_data = b""
        chunk_count = 0
        try:
            for chunk in self.stream_response.iter_content(chunk_size=1024):
                if not self.is_open:
                    break
                if chunk:
                    chunk_count += 1
                    if chunk_count == 1:
                        print("[INFO] Background thread is receiving raw data chunks from iPhone!")
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    if a != -1:
                        # Search for the end marker starting from the position of the start marker 'a'
                        b = bytes_data.find(b'\xff\xd9', a)
                        if b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.frame = img
                    elif len(bytes_data) > 2000000:  # Prevent buffer overflow
                        bytes_data = bytes_data[-50000:]
        except Exception as e:
            print(f"[DEBUG Thread Error]: {e}")

    def isOpened(self):
        return self.is_open

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
        return False, None

    def release(self):
        self.is_open = False

def open_camera(cam_source):
    if isinstance(cam_source, str) and cam_source.startswith("http"):
        return StreamWrapper(cam_source)
    
    try:
        idx = int(cam_source)
    except ValueError:
        idx = cam_source

    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if cap.isOpened():
        return cap
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    return cv2.VideoCapture(idx)

def main():
    args = parse_args()
    cap = open_camera(args.camera)

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {args.camera}")
        sys.exit(1)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("[INFO] System running. Opening live video window... Press 'q' to exit.")
    
    window_name = "Driver Drowsiness & Distraction Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        if frame_count == 1:
            print("[INFO] Success! Live frames streaming from iPhone.")

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Draw face mesh landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )

        cv2.putText(frame, "Status: Normal (iPhone Wi-Fi)", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Closed camera and resources.")

if __name__ == "__main__":
    main()