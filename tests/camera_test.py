# Camera diagnostic for the AUTONOMOUS driver-monitoring project.
# This small test checks the camera before we debug the full AI program.

import cv2  # Imports OpenCV for camera access and window display.
import time  # Lets us pause briefly between camera checks.

CAMERA_INDEX = 0  # Starts with camera index 0; change to 1 if DroidCam is another camera.
print("[TEST 1] Starting camera diagnostic...", flush=True)  # Prints immediately so output is not buffered.
print("[TEST 2] OpenCV version:", cv2.__version__, flush=True)  # Shows which OpenCV version is installed.

print("[TEST 3] Trying camera index", CAMERA_INDEX, "with MSMF...", flush=True)  # Reports the camera-opening attempt.
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)  # Opens the camera using Windows Media Foundation.

if not camera.isOpened():  # Checks whether the first camera backend failed.
    print("[INFO] MSMF failed; trying DirectShow...", flush=True)  # Reports the fallback attempt.
    camera.release()  # Releases the failed camera handle.
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Tries the DirectShow backend.

if not camera.isOpened():  # Checks whether both attempts failed.
    print("[ERROR] Camera could not open. Check DroidCam and try CAMERA_INDEX = 1.", flush=True)  # Gives troubleshooting advice.
else:  # Runs when the camera successfully opens.
    print("[TEST 4] Camera opened. Requesting a frame...", flush=True)  # Reports that frame capture is next.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Requests a small camera buffer to reduce delays.
    ok, frame = camera.read()  # Reads one camera frame.
    print("[TEST 5] Frame received:", ok, flush=True)  # Reports whether a valid frame arrived.

    if ok and frame is not None:  # Checks whether the camera produced an image.
        print("[TEST 6] Frame size:", frame.shape, flush=True)  # Shows the image dimensions.
        print("[TEST 7] Opening test window. Press Q or Esc to close.", flush=True)  # Explains the next action.
        while True:  # Keeps the test window visible until the user exits.
            ok, frame = camera.read()  # Reads the next camera frame.
            if not ok or frame is None:  # Checks whether frame capture stopped.
                print("[ERROR] Camera stopped sending frames.", flush=True)  # Reports the capture problem.
                break  # Leaves the display loop.
            cv2.imshow("DroidCam Camera Test", frame)  # Displays the camera image in a window.
            key = cv2.waitKey(1) & 0xFF  # Processes window events and reads keyboard input.
            if key in (ord("q"), 27):  # Checks whether Q or Escape was pressed.
                break  # Closes the test loop.
    else:  # Runs if opening succeeded but no image arrived.
        print("[ERROR] Camera opened but no frame was received.", flush=True)  # Reports a frame-capture failure.

camera.release()  # Releases the camera device.
cv2.destroyAllWindows()  # Closes any OpenCV windows.
print("[DONE] Camera diagnostic finished.", flush=True)  # Confirms that the test ended.
