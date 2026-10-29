# Real-Time Sound Classification with YAMNet

A Python prototype for detecting sound events from a live microphone stream and forwarding the resulting label to an EchoSense endpoint or an ESP32-driven visual indicator.

## Overview

The program monitors mono audio at 16 kHz, estimates short-window RMS energy, and begins classification when the signal rises above a threshold calibrated from a few seconds of background audio. It uses Google's pretrained YAMNet audio event model to score a window and select the highest-scoring class. Low-confidence predictions are mapped to `Noise`.

The repository also includes ESP32 firmware for a 12-pixel WS2812B LED ring. The firmware starts a small HTTP server and sets the ring colour based on sound-label text in a request body.

> **Prototype scope:** this is a real-time sound-event demo, not a validated safety-critical detector. Model confidence is not a guarantee that an event was correctly recognized.

## Features

- Live microphone capture with `sounddevice`
- Background-noise calibration using RMS energy
- Pretrained YAMNet inference through TensorFlow Hub
- Class-confidence thresholding (configured in the Python source)
- HTTP event forwarding to a local EchoSense API endpoint
- ESP32 HTTP endpoint that maps selected sound labels to LED-ring colours

## Architecture

```text
Microphone
   |
   v
Python audio capture (16 kHz, mono)
   |
   +--> RMS energy + background calibration
   |
   v
Audio window --> YAMNet --> label and score
                            |
                 +----------+-----------+
                 |                      |
                 v                      v
        EchoSense API POST        ESP32 HTTP request
        (configured endpoint)     (LED ring colour)
```

The Python script in this repository currently posts event data to `http://127.0.0.1:8000/api/alert`. The ESP32 firmware exposes an HTTP server on port 80. These are separate interfaces; make sure the relevant integration code and network addresses are configured for your deployment.

## Repository contents

| Path | Purpose |
| --- | --- |
| `visual.py` | Microphone capture, RMS-based activity detection, YAMNet inference, and event forwarding |
| `Sound_reactor.ino` | ESP32 Wi-Fi HTTP server and WS2812B LED-ring control |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore configuration |

## Hardware

- ESP32 development board with Wi-Fi
- WS2812B-compatible 12-LED ring
- USB cable and a suitable power source
- Computer with a working microphone for Python inference

### LED mapping

The ESP32 sketch uses data pin `GPIO 5` and `NUM_LEDS = 12`.

| Detected label text | LED colour |
| --- | --- |
| Speech | Blue |
| Bark | Orange |
| Knock | Purple |
| Music | Green |
| Clapping | Yellow |
| Doorbell | Magenta |
| Alarm | Red |
| Noise | Off |

The current sketch matches these labels by checking whether the lower-cased HTTP request body contains the relevant word. Ensure your Python integration sends a request to the ESP32's IP address using a compatible body format. The checked-in `visual.py` currently declares an EchoSense API URL; it does not directly post to the ESP32 endpoint in the active code.

## Requirements

- Python 3.10 or another Python version supported by the installed TensorFlow release
- A microphone input device
- Network access to download the pretrained YAMNet model the first time it runs
- ESP32 board support and the FastLED Arduino library for the hardware component

Install dependencies:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, you can run the environment's Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe visual.py
```

TensorFlow and TensorFlow Hub compatibility depends on your operating system, Python version, and hardware. If installation fails, verify the TensorFlow-supported Python versions and use a compatible environment.

## Run the classifier

From the repository root, with the virtual environment activated:

```bash
python visual.py
```

The program loads YAMNet from TensorFlow Hub, calibrates background noise for approximately three seconds, and then starts listening. It may request microphone permission from your operating system. Press `Ctrl+C` in the terminal to stop.

To change the EchoSense integration endpoint, edit the `ECHOSENSE_API_URL` setting in `visual.py`. To change the confidence threshold, review `MIN_CONFIDENCE`. Adjusting these settings changes program behaviour and should be followed by testing.

## Flash the ESP32 LED firmware

1. Install the Arduino IDE (or a compatible Arduino development environment).
2. Install the ESP32 board package and the **FastLED** library.
3. Open `Sound_reactor.ino`.
4. Replace `YOUR_WIFI_NAME` and `YOUR_WIFI_PASSWORD` with local credentials. Do not commit real credentials.
5. Select the correct ESP32 board and serial port, then upload.
6. Open Serial Monitor at **115200 baud** and note the IP address printed after Wi-Fi connects.
7. Send an HTTP request containing one of the mapped label words to the ESP32 on port 80.

Use a suitable power supply for the LED ring, and connect grounds as required by your board and LED-ring setup. Avoid powering a high-current LED ring directly from an ESP32 GPIO pin.

## Configuration and security

- Do not commit Wi-Fi passwords, API keys, Firebase tokens, or other credentials.
- Keep local endpoints (such as `127.0.0.1`) separate from device IP addresses. `127.0.0.1` means the same machine running the Python process.
- If the classifier and ESP32 are on different devices, use the reachable address of the destination device and ensure both devices can communicate on the network.
- Use authenticated, least-privilege access for deployed endpoints; do not expose unauthenticated alert endpoints to an untrusted network.

## Current limitations

- YAMNet selects from its pretrained AudioSet label vocabulary; it is not a custom-trained classifier for this repository.
- The RMS trigger detects changes in sound energy, not semantic events. Quiet sounds may be missed and loud background noise may trigger a window.
- The confidence threshold is a heuristic and needs evaluation on representative recordings.
- Network forwarding requires the configured service to be running and reachable.
- The checked-in ESP32 sketch expects an HTTP request body containing a recognized label string. Confirm that your active Python integration sends such a request before expecting the LED ring to change.
- The repository does not currently document a reproducible benchmark on a separate test set.

## Troubleshooting

**No microphone input**
- Check OS microphone permissions and the selected input device.
- Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to list available devices.

**YAMNet does not load**
- Check internet access on first launch and confirm TensorFlow/TensorFlow Hub are installed in a compatible Python environment.

**ESP32 does not connect**
- Verify the SSID/password, 2.4 GHz Wi-Fi compatibility, and the IP address printed in Serial Monitor.

**LEDs do not change**
- Confirm the ring's data line is connected to GPIO 5, the ring has adequate power, and the HTTP request reaches the ESP32 on port 80 with a matching label in its body.

**EchoSense API errors**
- Confirm the API service is running at the configured address and port, and that its expected request schema matches the JSON sent by `visual.py`.

## Future improvements

- Move settings into environment variables or a separate local configuration file.
- Add a direct, configurable ESP32 HTTP client path if that is the desired deployment architecture.
- Create a labelled evaluation set and report per-class precision, recall, F1, latency, and false-trigger rate.
- Add automated tests for label mapping and network request failures.

## License

No explicit license file was identified in this repository at the time this README was written. Add a license file before redistributing the code.
