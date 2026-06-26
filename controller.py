import requests
import io
import time
from PIL import Image
import numpy as np
from vis import predict_gap

BASE = "http://127.0.0.1:5000"


import cv2

def get_frame():
    try:
        r = requests.get(f"{BASE}/frame", timeout=2)
        img_array = np.frombuffer(r.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except:
        print("Frame fetch failed.")
        return None


def get_status():
    return requests.get(f"{BASE}/status", timeout=2).json()


def move(x, y, z):
    requests.post(
        f"{BASE}/control",
        json={
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "yaw": 0.0
        }
    )


def reset():
    requests.post(f"{BASE}/reset")


reset()
time.sleep(1)

last_gap_x = 0.0
last_gap_y = 0.0
last_dist = 2.0

while True:
    status = get_status()
    frame = get_frame()

    pos = status["position"]

    gap_x, gap_y, dist = predict_gap(frame)

    if gap_x == 0.0 and gap_y == 0.0:
        gap_x = last_gap_x
        gap_y = last_gap_y
        dist = last_dist
    else:
        last_gap_x = gap_x
        last_gap_y = gap_y
        last_dist = dist

    # Move ABSOLUTE into future space
    target_x = pos["x"] + (gap_x * (0.9 + 0.1 * dist))

    # IMPORTANT:
    # Don't use current y
    # Use future y
    target_y = pos["y"] + min(0.10, dist * 0.02)

    # Use gap_y to center vertically
    target_z = 1.25 + (gap_y * 0.05)
    target_z = max(1.1, min(1.35, target_z))

    print("Gap:", gap_x, gap_y, dist)
    print("Target:", target_x, target_y, target_z)

    move(target_x, target_y, target_z)

    if status["crashed"]:
        print("Resetting...")
        reset()
        time.sleep(1)

    time.sleep(0.03)