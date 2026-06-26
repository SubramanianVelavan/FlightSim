import requests
import time
import cv2
import numpy as np
from vis import predict_gap

BASE = "http://127.0.0.1:5000"

# ── tunables ─────────────────────────────────────────────────────────────────
CRUISE_ALTITUDE   = 2.0     # y: target flight height (matches drone start_pos y=2.0)
LATERAL_GAIN      = 3.5     # how aggressively to correct left/right (x)
ALTITUDE_GAIN     = 1.5     # how aggressively to correct altitude (y)
FORWARD_SPEED     = 0.35    # z units added per loop tick when gate is visible
FORWARD_SEARCH    = 0.08    # z units added when no gate detected (creep forward)
LOOP_DT           = 0.05    # seconds between control ticks
# ─────────────────────────────────────────────────────────────────────────────


def get_frame():
    try:
        r = requests.get(f"{BASE}/frame", timeout=2)
        img_array = np.frombuffer(r.content, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Frame fetch failed:", e)
        return None


def get_status():
    return requests.get(f"{BASE}/status", timeout=2).json()


def move(x, y, z):
    """
    World axes (from README / physics.py):
        x = left / right
        y = altitude  (ground ≈ 0.3, ceiling ≈ 14)
        z = forward depth  (gates are at z = 15, 30, 45 …)
    """
    requests.post(
        f"{BASE}/control",
        json={"x": float(x), "y": float(y), "z": float(z), "yaw": 0.0}
    )


def reset():
    requests.post(f"{BASE}/reset")


# ── startup ───────────────────────────────────────────────────────────────────
reset()
time.sleep(1.0)

last_gap_x = 0.0
last_gap_y = 0.0
gate_visible = False

# ── main loop ─────────────────────────────────────────────────────────────────
while True:
    status = get_status()

    if status["crashed"]:
        print("Crashed — resetting …")
        reset()
        time.sleep(1.0)
        last_gap_x = 0.0
        last_gap_y = 0.0
        gate_visible = False
        continue

    frame = get_frame()
    if frame is None:
        time.sleep(LOOP_DT)
        continue

    pos = status["position"]
    gap_x, gap_y, dist = predict_gap(frame)

    if gap_x == 0.0 and gap_y == 0.0:
        # No detection — hold last known lateral/altitude offset, creep forward
        gap_x = last_gap_x
        gap_y = last_gap_y
        gate_visible = False
    else:
        last_gap_x = gap_x
        last_gap_y = gap_y
        gate_visible = True

    # ── x: steer left/right toward gate center ────────────────────────────
    # gap_x is in [-1, 1]: negative = gate is left of centre → move left (lower x)
    target_x = pos["x"] + gap_x * LATERAL_GAIN

    # ── y: correct altitude toward gate center ────────────────────────────
    # gap_y is in [-1, 1]: positive = gate center is ABOVE camera center → fly up
    target_y = CRUISE_ALTITUDE + gap_y * ALTITUDE_GAIN

    # ── z: always push forward; scale with distance so we slow down a bit
    # when very close (dist≈1) and go faster when far (dist≈5+)
    forward_step = FORWARD_SPEED * min(dist, 3.0) if gate_visible else FORWARD_SEARCH
    target_z = pos["z"] + forward_step

    print(f"gap=({gap_x:+.2f}, {gap_y:+.2f}) dist={dist:.1f} | "
          f"pos=({pos['x']:.1f},{pos['y']:.1f},{pos['z']:.1f}) | "
          f"tgt=({target_x:.1f},{target_y:.1f},{target_z:.1f})")

    move(target_x, target_y, target_z)

    time.sleep(LOOP_DT)