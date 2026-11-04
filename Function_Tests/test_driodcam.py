import time
import requests
import cv2
import numpy as np

URL = "http://192.168.137.2:4747/video"

print("[TEST] Connecting to iPhone DroidCam...")

try:
    response = requests.get(
        URL,
        headers={"User-Agent": "Mozilla/5.0"},
        stream=True,
        timeout=(5, 10)
    )

    print("[TEST] HTTP status:", response.status_code)
    print("[TEST] Content-Type:", response.headers.get("Content-Type"))

    response.raise_for_status()

    buffer = bytearray()
    start_time = time.time()
    frame_found = False

    for chunk in response.iter_content(chunk_size=1024):
        if not chunk:
            continue

        buffer.extend(chunk)

        while True:
            start = buffer.find(b"\xff\xd8")

            if start == -1:
                buffer = buffer[-4:]
                break

            end = buffer.find(b"\xff\xd9", start + 2)

            if end == -1:
                buffer = buffer[start:]
                break

            jpeg = bytes(buffer[start:end + 2])
            del buffer[:end + 2]

            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is not None:
                print("[SUCCESS] Video frame received!")
                print("[SUCCESS] Frame size:", frame.shape)

                cv2.imshow("DroidCam Test", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                frame_found = True
                break

        if frame_found:
            break

        if time.time() - start_time > 15:
            print("[ERROR] No valid video frame received within 15 seconds.")
            break

    if not frame_found:
        print("[FAILED] Could not decode a video frame.")

    response.close()

except Exception as error:
    print("[ERROR]", repr(error))

finally:
    cv2.destroyAllWindows()