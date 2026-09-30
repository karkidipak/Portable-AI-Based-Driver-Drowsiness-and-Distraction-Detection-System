# Hardware

## Planned Hardware

The following components are planned for the portable driver drowsiness and distraction detection system.

| Component       | Planned Model                     | Purpose                                                  |
| --------------- | --------------------------------- | -------------------------------------------------------- |
| Main Computer   | Raspberry Pi 5 Model B – 4GB      | Runs the computer vision and drowsiness detection system |
| Camera          | Raspberry Pi Camera Module 3 Wide | Captures the driver's face                               |
| Microcontroller | ESP32-DevKitC V4 / ESP32-WROOM-32 | Controls the warning system                              |
| Buzzer          | 5V Active Buzzer                  | Provides an audible warning                              |
| LED             | RGB LED Module                    | Provides a visual warning                                |
| Storage         | 64GB microSD Card                 | Stores the operating system and project files            |
| Power           | 10,000mAh USB-C Power Bank        | Provides portable power                                  |
| Enclosure       | Raspberry Pi 5 Compatible Case    | Protects and houses the electronics                      |
| Optional GPS    | NEO-6M GPS Module                 | Records location during warning events                   |

## Hardware System

```text
Camera
   ↓
Raspberry Pi
   ↓
Drowsiness / Distraction Detection
   ↓
ESP32
   ↓
┌─────────────┐
│             │
Buzzer       LED
│             │
└─────────────┘
```

## Status

**Current Status:** Planned

The hardware specifications may be updated during the development and testing stages.

## Hardware Documentation

* `esp32/` — ESP32 code and related files
* `wiring/` — Wiring information
* `circuit_diagram/` — Circuit diagrams

