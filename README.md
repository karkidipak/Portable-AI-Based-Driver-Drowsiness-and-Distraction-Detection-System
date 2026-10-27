# Portable AI-Based Driver Drowsiness and Distraction Detection System

## Current prototype

This repository now includes a laptop/phone-camera software prototype. It is intended for development and demonstration only, not as a certified driver safety device.

### Detection features
- Facial landmarks using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) and sustained eye closure warning
- Mouth opening ratio and sustained yawning warning (heuristic)
- Approximate head pose using OpenCV solvePnP
- Optional cell-phone object detection using pretrained Ultralytics YOLO (COCO class `cell phone`)
- On-screen status, CSV event logging, screenshot capture
- Optional serial alert messages for later ESP32 integration

### Setup on Windows PowerShell

Use a supported Python version for the installed MediaPipe release. If installation fails, use Python 3.11 in a fresh virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run
```powershell
python src/main.py --camera 0
```
Try another camera index if required:
```powershell
python src/main.py --camera 1
```
If the phone is exposed as a network stream, pass its stream URL:
```powershell
python src/main.py --camera "http://PHONE_STREAM_URL"
```
To run without the YOLO phone model:
```powershell
python src/main.py --camera 0 --no-phone
```

The first YOLO run may download `yolov8n.pt`, requiring internet access. Phone detection can be slow on a CPU and can miss phones; it detects a phone object, not whether the driver is using it.

### Controls
- `q`: quit
- `s`: save screenshot to `logs/`

### Optional ESP32 serial integration
Once ESP32 firmware is ready and the correct port is known:
```powershell
python src/main.py --camera 0 --serial-port COM5
```
Replace `COM5` with the actual device port. The Python app sends newline-terminated messages like `ALERT,DROWSINESS|YAWNING`. Agree on and implement the same protocol in the ESP32 firmware before connecting a real buzzer or LED. Serial integration is optional and is not physically tested without the board.

### Test the mathematical helpers
```powershell
python -m unittest discover -s tests -v
```

### Important limitations
- EAR/MAR and head-pose thresholds need calibration for camera placement, face shape, lighting, and individual users.
- Yawning uses a simple mouth-opening heuristic and may produce false positives.
- Head direction is approximate and should be calibrated against neutral posture.
- COCO phone detection detects a phone in the image; it does not confirm phone use by the driver.
- No hardware is physically integrated or validated yet.
- This prototype is not suitable for real driving or safety-critical use.
