"""Quick headless check that the webcam + hand detection work.

Run this before main.py to confirm the camera opens and MediaPipe sees your
hand. It grabs a few dozen frames, prints how many hands were detected and
which gesture was read — no GUI window needed, so it also works to diagnose
whether a problem is with the camera or with the display.

    python3 camera_check.py
"""

import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rps import classify

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
FRAMES = 60


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit(
            "Camera did NOT open. On macOS, grant camera access to your "
            "terminal app: System Settings > Privacy & Security > Camera."
        )
    print("Camera opened. Grabbing frames — hold a hand up to the camera...")

    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)

    seen_hand = 0
    gestures = {}
    for i in range(FRAMES):
        ok, frame = cap.read()
        if not ok:
            print(f"frame {i}: capture failed")
            continue
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect_for_video(img, int(time.perf_counter() * 1000))
        if res.hand_landmarks:
            seen_hand += 1
            g = classify(res.hand_landmarks[0]) or "ambiguous"
            gestures[g] = gestures.get(g, 0) + 1
        time.sleep(0.03)

    landmarker.close()
    cap.release()

    print(f"\nFrames grabbed: {FRAMES}")
    print(f"Frames with a hand: {seen_hand}")
    print(f"Gestures read: {gestures or 'none'}")
    if seen_hand == 0:
        print(
            "\nNo hand detected. Make sure your hand is well-lit and fully "
            "in frame, then rerun."
        )
    else:
        print("\nAll good — run:  python3 main.py")


if __name__ == "__main__":
    main()
