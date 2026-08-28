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


import requests


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
# SEND SOUND TO ESP32
# ============================================================

def send_sound_to_esp32(sound):

    try:

        url = f"http://{ESP32_IP}/sound"

        response = requests.post(
            url,
            data=str(sound),
            timeout=2
        )

        print(
            f"[ESP32] Sent: {sound} | {response.text}"
        )

    except requests.exceptions.RequestException as e:

        print(f"[ESP32 ERROR] {e}")
# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

# Microphone monitoring chunk.
# 20 ms = 320 samples at 16 kHz.
CHUNK_DURATION = 0.02
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# YAMNet receives 1 second of audio.
CLASSIFICATION_SECONDS = 1.5
CLASSIFICATION_SAMPLES = int(SAMPLE_RATE * CLASSIFICATION_SECONDS)

# Initial noise calibration duration.
CALIBRATION_SECONDS = 3.0

# Threshold multiplier.
# Increase if there are too many false triggers.
ENERGY_MULTIPLIER = 2.5

# Minimum threshold so extremely quiet environments
# don't become hypersensitive.
MIN_THRESHOLD = 0.02

# Sound must remain below threshold for this long
# before listening becomes False.
SILENCE_DURATION = 0.2

# How many seconds of audio to retain before the trigger.
# This gives us some pre-trigger audio.
PRE_TRIGGER_SECONDS = 0.2
PRE_TRIGGER_SAMPLES = int(SAMPLE_RATE * PRE_TRIGGER_SECONDS)

# Minimum time between starting two classification jobs.
CLASSIFICATION_COOLDOWN = 0.1

# Enable/disable TTS.
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

# Locks
state_lock = threading.Lock()
processing_lock = threading.Lock()


# Audio buffers
pre_trigger_buffer = deque(maxlen=PRE_TRIGGER_SAMPLES)

audio_buffer = deque(maxlen=CLASSIFICATION_SAMPLES)

# Used to communicate audio from microphone to classifier.
classification_queue = queue.Queue()
stop_event = threading.Event()


# ============================================================
# LOAD YAMNET
# ============================================================

print("\nLoading YAMNet...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print("YAMNet loaded.")


# ============================================================
# LOAD YAMNET CLASS NAMES
# ============================================================

class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")

class_names = pd.read_csv(class_map_path)["display_name"].tolist()

print(f"Loaded {len(class_names)} YAMNet classes.")


# ============================================================
# TTS
# ============================================================



# ============================================================
# AUDIO ENERGY
# ============================================================

def calculate_rms(audio):

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) == 0:
        return 0.0

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )

    return float(rms)


# ============================================================
# NOISE CALIBRATION
# ============================================================

def calibrate_noise():

    print("\n====================================")
    print("       NOISE CALIBRATION")
    print("====================================")

    print(
        f"Listening to background noise for "
        f"{CALIBRATION_SECONDS} seconds..."
    )

    energies = []

    start_time = time.time()

    def callback(indata, frames, time_info, status):

        if status:
            print(status)

        audio = indata[:, 0].copy()

        rms = calculate_rms(audio)

        energies.append(rms)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=callback
    ):

        while time.time() - start_time < CALIBRATION_SECONDS:
            time.sleep(0.01)

    if not energies:

        print("ERROR: No microphone data received.")

        sys.exit(1)

    noise_mean = np.mean(energies)
    noise_std = np.std(energies)

    # Use both average noise and variation.
    threshold = max(
        MIN_THRESHOLD,
        noise_mean + ENERGY_MULTIPLIER * noise_std
    )

    print("\nCalibration complete.")

    print(
        f"Average noise RMS : {noise_mean:.6f}"
    )

    print(
        f"Noise deviation    : {noise_std:.6f}"
    )

    print(
        f"Energy threshold    : {threshold:.6f}"
    )

    print("====================================\n")

    return threshold


# ============================================================
# YAMNET CLASSIFICATION
# ============================================================

def classify_audio(audio):

    try:

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        audio = audio.flatten()

        if len(audio) < CLASSIFICATION_SAMPLES:

            audio = np.pad(
                audio,
                (
                    0,
                    CLASSIFICATION_SAMPLES - len(audio)
                )
            )

        elif len(audio) > CLASSIFICATION_SAMPLES:

            audio = audio[:CLASSIFICATION_SAMPLES]

        # Run YAMNet
        scores, embeddings, spectrogram = yamnet_model(audio)

        # Average scores across the audio frames
        mean_scores = tf.reduce_mean(
            scores,
            axis=0
        )

        # Find BEST class only
        best_index = int(
            tf.argmax(mean_scores)
        )

        best_class = class_names[best_index]

        best_score = float(
            mean_scores[best_index]
        )

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

            audio = classification_queue.get(
                timeout=0.1
            )

        except queue.Empty:

            continue

        with processing_lock:

            processing = True

            try:

                #print("[WORKER] Received 1-second audio.")

                result, confidence = classify_audio(
                    audio
                )

                if result is not None:

                    if confidence < MIN_CONFIDENCE:


                        result = "Noise"
                    

                    if result == last_displayed_class:

                        continue;
                    

                    elif confidence >= MIN_CONFIDENCE:

                        print(
                        f"[RESULT] {result} "
                        f"| confidence = {confidence:.2%}"
                    )
                    
                send_sound_to_esp32(result)
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

        print(
            f"[AUDIO STATUS] {status}"
        )

    # Copy microphone samples.
    audio = indata[:, 0].copy()

    # --------------------------------------------------------
    # Calculate current energy
    # --------------------------------------------------------

    rms = calculate_rms(audio)

    # --------------------------------------------------------
    # Keep a small pre-trigger buffer
    # --------------------------------------------------------

    for sample in audio:

        pre_trigger_buffer.append(
            float(sample)
        )

    # --------------------------------------------------------
    # Check threshold
    # --------------------------------------------------------

    if rms >= threshold:

        # Sound detected.
        if not listening:

            print(
                f"\n[SOUND DETECTED] "
                f"RMS={rms:.5f} "
                f"> threshold={threshold:.5f}"
            )

            print(
                "[STATE] listening = True"
            )

            listening = True

            below_threshold_time = 0.0

        else:

            # Still hearing sound.
            below_threshold_time = 0.0

    else:

        # Below threshold.
        if listening:

            below_threshold_time += CHUNK_DURATION

            # Don't immediately stop on a tiny dip.
            if below_threshold_time >= SILENCE_DURATION:

                listening = False
                print('\n[STATE] listening = False')

                below_threshold_time = 0.0
                result= None
                last_displayed_class = None


            

    # --------------------------------------------------------
    # If listening, add audio to classification buffer
    # --------------------------------------------------------

    if listening:

        for sample in audio:

            audio_buffer.append(
                float(sample)
            )

        # ----------------------------------------------------
        # Once we have 1 second, send it to YAMNet
        # ----------------------------------------------------

        if len(audio_buffer) >= CLASSIFICATION_SAMPLES:

            # Copy exactly one second.
            clip = np.array(
                list(audio_buffer)[
                    -CLASSIFICATION_SAMPLES:
                ],
                dtype=np.float32
            )

            # Remove the oldest samples corresponding to
            # one classification window.
            for _ in range(
                min(
                    CHUNK_SIZE,
                    len(audio_buffer)
                )
            ):

                audio_buffer.popleft()

            # Don't queue multiple copies while one is
            # already waiting/being processed.
            if not processing:

                try:

                    classification_queue.put_nowait(
                        clip
                    )


                except Exception as e:

                    print(
                        f"[QUEUE ERROR] {e}"
                    )


# ============================================================
# MAIN
# ============================================================

def main():

    global threshold

    print("\n========================================")
    print("       REAL-TIME SOUND CLASSIFIER")
    print("========================================")

    print(
        f"Sample rate          : {SAMPLE_RATE} Hz"
    )

    print(
        f"Classification length: "
        f"{CLASSIFICATION_SECONDS} second"
    )

    print(
        f"Calibration           : "
        f"{CALIBRATION_SECONDS} seconds"
    )

    # --------------------------------------------------------
    # 1. Calibrate
    # --------------------------------------------------------

    threshold = calibrate_noise()

    # --------------------------------------------------------
    # 2. Start YAMNet worker
    # --------------------------------------------------------

    worker = threading.Thread(
        target=classification_worker,
        daemon=True
    )

    worker.start()

    # --------------------------------------------------------
    # 3. Start continuous microphone stream
    # --------------------------------------------------------

    print("\n========================================")
    print("       STARTING CONTINUOUS LISTENING")
    print("========================================")

    print(
        "Waiting for sound..."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback
        ):

            while True:

                time.sleep(0.1)

    except KeyboardInterrupt:

        print(
            "\nStopping..."
        )

    finally:

        stop_event.set()

        print(
            "Program stopped."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()