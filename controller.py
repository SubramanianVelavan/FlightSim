"""
Autonomy controller: fetches the FPV frame, perceives the gate (YOLO + SAM 3D
via vis.py), and steers the drone through it over the sim's REST API.

Run the sim first (python main.py in the sim repo), then:  python controller.py
Point it at another machine with:  SIM_BASE=http://<host>:5000 python controller.py
"""

import os
import time

import cv2
import numpy as np
import requests

import vis

BASE = os.environ.get("SIM_BASE", "http://127.0.0.1:5000")

# Physics constants (mirror physics.py). To hold altitude the spring needs the
# target_y biased GRAVITY/ATTRACT_K above current y — that cancels gravity.
GRAVITY    = 4.0
ATTRACT_K  = 3.0
HOVER_BIAS = GRAVITY / ATTRACT_K   # ≈ 1.333

# ── tunables ──────────────────────────────────────────────────────────────────
CRUISE_ALTITUDE = 2.5
LATERAL_GAIN    = 0.20    # gap_x -> lateral target nudge
ALTITUDE_GAIN   = 0.8     # gap_y -> altitude nudge
Z_STEP_SEARCH   = 0.12    # creep forward when no gate is visible
Z_STEP_MIN      = 0.10    # forward push when gate is close
Z_STEP_MAX      = 0.25    # forward push when gate is far
LOOP_DT         = 0.05
# ─────────────────────────────────────────────────────────────────────────────

session = requests.Session()


def get_frame():
    try:
        r = session.get(f"{BASE}/frame", timeout=2)
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print("Frame fetch failed:", e)
        return None


def get_status():
    return session.get(f"{BASE}/status", timeout=2).json()


def move(x, y, z):
    session.post(f"{BASE}/control",
                 json={"x": float(x), "y": float(y), "z": float(z), "yaw": 0.0},
                 timeout=2)


def reset():
    session.post(f"{BASE}/reset", timeout=2)


def load_intrinsics():
    """Pull real camera focal length + gate width from the sim so vis.py's
    distance estimate is exact. Harmless no-op against an older sim."""
    try:
        g = session.get(f"{BASE}/gates", timeout=2).json()
        cam = g.get("camera", {})
        gw = g.get("gates", [{}])[0].get("width") if g.get("gates") else None
        vis.set_intrinsics(focal_px=cam.get("focal_px"), frame_w=cam.get("width"),
                           frame_h=cam.get("height"), gate_width_m=gw)
        print(f"[ctl] intrinsics: focal={cam.get('focal_px')} gate_w={gw}")
    except Exception as e:
        print(f"[ctl] /gates not available ({e}); using vis.py defaults")


# ── startup ───────────────────────────────────────────────────────────────────
reset()
time.sleep(1.0)
load_intrinsics()

last_gap_x = last_gap_y = 0.0

# ── main loop ─────────────────────────────────────────────────────────────────
while True:
    try:
        status = get_status()
    except Exception as e:
        print("Status fetch failed:", e)
        time.sleep(0.5)
        continue

    if status.get("crashed"):
        print("Crashed — resetting …")
        reset()
        time.sleep(1.0)
        last_gap_x = last_gap_y = 0.0
        continue

    frame = get_frame()
    if frame is None:
        time.sleep(LOOP_DT)
        continue

    pos = status["position"]
    det = vis.detect(frame)

    if det.found:
        gap_x, gap_y = det.gap_x, det.gap_y
        dist = det.distance
        last_gap_x, last_gap_y = gap_x, gap_y
    else:
        gap_x, gap_y = last_gap_x, last_gap_y
        dist = 3.0

    # ── x: incremental lateral correction toward gate centre ──────────────
    target_x = pos["x"] + gap_x * LATERAL_GAIN

    # ── y: hover bias (fight gravity) + steer toward gate centre ──────────
    if det.found:
        altitude_correction = gap_y * ALTITUDE_GAIN
    else:
        altitude_correction = (CRUISE_ALTITUDE - pos["y"]) * 0.3
    target_y = pos["y"] + HOVER_BIAS + altitude_correction
    target_y = max(1.0, min(12.0, target_y))   # safety clamp

    # ── z: forward thrust, ease off when badly off-centre ─────────────────
    if det.found:
        z_step = Z_STEP_MIN + (Z_STEP_MAX - Z_STEP_MIN) * min((dist - 1.0) / 4.0, 1.0)
        if abs(gap_x) + abs(gap_y) > 0.5:
            z_step *= 0.6
    else:
        z_step = Z_STEP_SEARCH
    target_z = pos["z"] + z_step

    print(f"[{det.source:8}] gap=({gap_x:+.2f},{gap_y:+.2f}) dist={dist:4.1f}m "
          f"pos=({pos['x']:+.2f},{pos['y']:.2f},{pos['z']:.2f}) "
          f"spd={status.get('speed', 0):.2f} passed={status.get('gates_passed', 0)}")

    move(target_x, target_y, target_z)
    time.sleep(LOOP_DT)
