import requests
import time
import cv2
import numpy as np
from vis import predict_gap

BASE = "http://127.0.0.1:5000"

# Physics constants (must match physics.py)
GRAVITY    = 4.0
ATTRACT_K  = 3.0
# To hover at any altitude, target_y must be GRAVITY/ATTRACT_K above current pos
HOVER_BIAS = GRAVITY / ATTRACT_K   # = 1.333 — always add this to target_y

# ── tunables ──────────────────────────────────────────────────────────────────
CRUISE_ALTITUDE = 2.5   # default flight altitude when no gate detected

# Lateral gain: gap_x * LATERAL_GAIN added to current x each tick
LATERAL_GAIN    = 0.20

# Altitude gain: how much above cruise to push per unit of gap_y
ALTITUDE_GAIN   = 0.8

# Forward speed
Z_STEP_SEARCH   = 0.12   # creep forward when gate not visible
Z_STEP_MIN      = 0.10   # minimum forward push when gate visible (close)
Z_STEP_MAX      = 0.25   # maximum forward push when gate visible (far)

LOOP_DT         = 0.05
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
    requests.post(
        f"{BASE}/control",
        json={"x": float(x), "y": float(y), "z": float(z), "yaw": 0.0}
    )


def reset():
    requests.post(f"{BASE}/reset")


# ── startup ───────────────────────────────────────────────────────────────────
reset()
time.sleep(1.0)

last_gap_x   = 0.0
last_gap_y   = 0.0
gate_visible = False

# ── main loop ─────────────────────────────────────────────────────────────────
while True:
    try:
        status = get_status()
    except Exception as e:
        print("Status fetch failed:", e)
        time.sleep(0.5)
        continue

    if status["crashed"]:
        print("Crashed — resetting …")
        reset()
        time.sleep(1.0)
        last_gap_x   = 0.0
        last_gap_y   = 0.0
        gate_visible = False
        continue

    frame = get_frame()
    if frame is None:
        time.sleep(LOOP_DT)
        continue

    pos = status["position"]
    gap_x, gap_y, dist = predict_gap(frame)

    if gap_x == 0.0 and gap_y == 0.0:
        gap_x        = last_gap_x
        gap_y        = last_gap_y
        gate_visible = False
    else:
        last_gap_x   = gap_x
        last_gap_y   = gap_y
        gate_visible = True

    # ── x: incremental lateral correction ────────────────────────────────
    target_x = pos["x"] + gap_x * LATERAL_GAIN

    # ── y: MUST include hover bias to fight gravity ───────────────────────
    # The spring needs target_y = current_y + HOVER_BIAS just to hold altitude.
    # Then we add gap_y * ALTITUDE_GAIN on top to actually steer toward gate center.
    if gate_visible:
        altitude_correction = gap_y * ALTITUDE_GAIN
    else:
        # Drift back toward cruise altitude
        altitude_correction = (CRUISE_ALTITUDE - pos["y"]) * 0.3

    target_y = pos["y"] + HOVER_BIAS + altitude_correction

    # Safety clamp: never command below ground+margin or above ceiling
    target_y = max(1.0, min(12.0, target_y))

    # ── z: forward thrust, slow down if badly off-center ─────────────────
    if gate_visible:
        z_step = Z_STEP_MIN + (Z_STEP_MAX - Z_STEP_MIN) * min((dist - 1.0) / 4.0, 1.0)
        lateral_error = abs(gap_x) + abs(gap_y)
        if lateral_error > 0.5:
            z_step *= 0.6   # slow down to let steering catch up
    else:
        z_step = Z_STEP_SEARCH

    target_z = pos["z"] + z_step

    print(f"{'SEE' if gate_visible else '   '} "
          f"gap=({gap_x:+.2f},{gap_y:+.2f}) dist={dist:.1f} | "
          f"pos=({pos['x']:+.2f},{pos['y']:.2f},{pos['z']:.2f}) | "
          f"tgt=({target_x:+.2f},{target_y:.2f},{target_z:.2f})")

    move(target_x, target_y, target_z)

    time.sleep(LOOP_DT)