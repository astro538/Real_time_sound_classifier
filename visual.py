import os
import sys
import time
import threading

from collections import deque
from unittest import result

import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import queue
import requests

ESP32_IP = "192.168.29.211"
ECHOSENSE_API_URL = "http://127.0.0.1:8000/api/alert"

# ============================================================
# FIREBASE SETTINGS  ← ADDED
# ============================================================
FIREBASE_URL = "https://rain-dei-default-rtdb.firebaseio.com"
AUTH = "xxeSyW61RLsqJLy9O9pXmHr9RK5NR8Yo7cka2W0G"

# ============================================================
# ESP32 SETTINGS
# ============================================================

#ESP32_URL = f"http://{ESP32_IP}/color"

# ============================================================
# SOUND → LED COLOR
# ============================================================

SOUND_COLORS = {
    "Speech": "blue",
    "Bark": "orange",
    "Knock": "purple",
    "Music": "green",
    "Clapping": "yellow",
    "Doorbell": "magenta",
    "Alarm": "red",
    "Noise": "off"
}


# ============================================================
# SEND SOUND TO ECHOSENSE
# ============================================================

def yamnet_event_type(sound):
    """Translate YAMNet labels into the alert types used by EchoSense."""
    label = sound.lower()

    if "fire alarm" in label or "smoke alarm" in label or "smoke detector" in label:
        return "FIRE"
    if "doorbell" in label or "door bell" in label:
        return "DOORBELL"
    if "knock" in label:
        return "KNOCK"
    if "alarm" in label or "siren" in label:
        return "ALARM"
    return "SOUND"


def send_sound_to_echosense(sound, confidence):
    event_type = yamnet_event_type(sound)

    try:
        response = requests.post(
            ECHOSENSE_API_URL,
            json={
                "event": sound,
                "event_type": event_type,
                "confidence": confidence
            },
            timeout=2
        )

        print(
            f"[ECHOSENSE] Sent: {sound} ({event_type}) | {response.text}"
        )

    except requests.exceptions.RequestException as e:
        print(f"[ECHOSENSE ERROR] {e}")

    # ============================================================
    # SEND TO FIREBASE  ← ADDED (inside function, correct place)
    # ============================================================
    try:
        requests.put(
            f"{FIREBASE_URL}/latest_sound_classification.json?auth={AUTH}",
            json={
                "sound_class": event_type,
                "label": sound,
                "confidence": round(confidence, 4)
            },
            timeout=2
        )
        print(f"[FIREBASE] Pushed: {sound} ({event_type}) confidence={confidence:.2%}")

    except requests.exceptions.RequestException as e:
        print(f"[FIREBASE ERROR] {e}")


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.02
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
CLASSIFICATION_SECONDS = 1.5
CLASSIFICATION_SAMPLES = int(SAMPLE_RATE * CLASSIFICATION_SECONDS)
CALIBRATION_SECONDS = 3.0
ENERGY_MULTIPLIER = 2.5
MIN_THRESHOLD = 0.02
SILENCE_DURATION = 0.2
PRE_TRIGGER_SECONDS = 0.2
PRE_TRIGGER_SAMPLES = int(SAMPLE_RATE * PRE_TRIGGER_SECONDS)
CLASSIFICATION_COOLDOWN = 0.1
ENABLE_TTS = False


# ============================================================
# GLOBAL STATE
# ============================================================

listening = False
processing = False
below_threshold_time = 0.0
MIN_CONFIDENCE = 0.5
last_displayed_class = None
listening = False
processing = False

state_lock = threading.Lock()
processing_lock = threading.Lock()

pre_trigger_buffer = deque(maxlen=PRE_TRIGGER_SAMPLES)
audio_buffer = deque(maxlen=CLASSIFICATION_SAMPLES)
classification_queue = queue.Queue()
stop_event = threading.Event()


# ============================================================
# LOAD YAMNET
# ============================================================

print("\nLoading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")


# ============================================================
# LOAD YAMNET CLASS NAMES
# ============================================================

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
class_names = pd.read_csv(class_map_path)["display_name"].tolist()
print(f"Loaded {len(class_names)} YAMNet classes.")


# ============================================================
# AUDIO ENERGY
# ============================================================

def calculate_rms(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.square(audio)))
    return float(rms)


# ============================================================
# NOISE CALIBRATION
# ============================================================

def calibrate_noise():
    print("\n====================================")
    print("       NOISE CALIBRATION")
    print("====================================")
    print(f"Listening to background noise for {CALIBRATION_SECONDS} seconds...")

    energies = []
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio = indata[:, 0].copy()
        rms = calculate_rms(audio)
        energies.append(rms)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=CHUNK_SIZE, callback=callback):
        while time.time() - start_time < CALIBRATION_SECONDS:
            time.sleep(0.01)

    if not energies:
        print("ERROR: No microphone data received.")
        sys.exit(1)

    noise_mean = np.mean(energies)
    noise_std = np.std(energies)
    threshold = max(MIN_THRESHOLD, noise_mean + ENERGY_MULTIPLIER * noise_std)

    print("\nCalibration complete.")
    print(f"Average noise RMS : {noise_mean:.6f}")
    print(f"Noise deviation    : {noise_std:.6f}")
    print(f"Energy threshold    : {threshold:.6f}")
    print("====================================\n")

    return threshold


# ============================================================
# YAMNET CLASSIFICATION
# ============================================================

def classify_audio(audio):
    try:
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if len(audio) < CLASSIFICATION_SAMPLES:
            audio = np.pad(audio, (0, CLASSIFICATION_SAMPLES - len(audio)))
        elif len(audio) > CLASSIFICATION_SAMPLES:
            audio = audio[:CLASSIFICATION_SAMPLES]

        scores, embeddings, spectrogram = yamnet_model(audio)
        mean_scores = tf.reduce_mean(scores, axis=0)
        best_index = int(tf.argmax(mean_scores))
        best_class = class_names[best_index]
        best_score = float(mean_scores[best_index])

        return best_class, best_score

    except Exception as e:
        print(f"[YAMNET ERROR] {e}")
        return None, 0.0


# ============================================================
# CLASSIFICATION WORKER
# ============================================================

def classification_worker():
    global processing
    global classification_pending
    global last_displayed_class

    print("[WORKER] YAMNet worker started.")

    while not stop_event.is_set():
        try:
            audio = classification_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        with processing_lock:
            processing = True
            try:
                result, confidence = classify_audio(audio)

                if result is not None:
                    if confidence < MIN_CONFIDENCE:
                        result = "Noise"
                    if result == last_displayed_class:
                        continue
                    elif confidence >= MIN_CONFIDENCE:
                        print(f"[RESULT] {result} | confidence = {confidence:.2%}")

                send_sound_to_echosense(result, confidence)
                last_displayed_class = result

            except Exception as e:
                print(f"[WORKER ERROR] {e}")

            finally:
                processing = False
                classification_pending = False
                classification_queue.task_done()


# ============================================================
# MICROPHONE CALLBACK
# ============================================================

threshold = None


def audio_callback(indata, frames, time_info, status):
    global listening
    global below_threshold_time
    global last_displayed_class

    if status:
        print(f"[AUDIO STATUS] {status}")

    audio = indata[:, 0].copy()
    rms = calculate_rms(audio)

    for sample in audio:
        pre_trigger_buffer.append(float(sample))

    if rms >= threshold:
        if not listening:
            print(f"\n[SOUND DETECTED] RMS={rms:.5f} > threshold={threshold:.5f}")
            print("[STATE] listening = True")
            listening = True
            below_threshold_time = 0.0
        else:
            below_threshold_time = 0.0
    else:
        if listening:
            below_threshold_time += CHUNK_DURATION
            if below_threshold_time >= SILENCE_DURATION:
                listening = False
                print('\n[STATE] listening = False')
                below_threshold_time = 0.0
                result = None
                last_displayed_class = None

    if listening:
        for sample in audio:
            audio_buffer.append(float(sample))

        if len(audio_buffer) >= CLASSIFICATION_SAMPLES:
            clip = np.array(list(audio_buffer)[-CLASSIFICATION_SAMPLES:], dtype=np.float32)

            for _ in range(min(CHUNK_SIZE, len(audio_buffer))):
                audio_buffer.popleft()

            if not processing:
                try:
                    classification_queue.put_nowait(clip)
                except Exception as e:
                    print(f"[QUEUE ERROR] {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    global threshold

    print("\n========================================")
    print("       REAL-TIME SOUND CLASSIFIER")
    print("========================================")
    print(f"Sample rate          : {SAMPLE_RATE} Hz")
    print(f"Classification length: {CLASSIFICATION_SECONDS} second")
    print(f"Calibration           : {CALIBRATION_SECONDS} seconds")

    threshold = calibrate_noise()

    worker = threading.Thread(target=classification_worker, daemon=True)
    worker.start()

    print("\n========================================")
    print("       STARTING CONTINUOUS LISTENING")
    print("========================================")
    print("Waiting for sound...")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK_SIZE, callback=audio_callback):
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        print("Program stopped.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
