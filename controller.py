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
# NOTE: these gains now multiply a world-space offset (gap * distance), not
# a raw screen fraction — 1.0 means "aim squarely at the gate's estimated
# position", <1.0 damps it slightly to stay stable under detection noise.
LATERAL_GAIN    = 0.85    # world-space lateral target gain
ALTITUDE_GAIN   = 0.85    # world-space vertical target gain
Z_STEP_SEARCH   = 0.12    # creep forward when no gate is visible
Z_STEP_MIN      = 0.10    # forward push when gate is close
Z_STEP_MAX      = 0.25    # forward push when gate is far
LOOP_DT         = 0.05

LOCK_MATCH_MAX_PX   = 140    # px; how far a candidate can be from the locked gate's
                              # last screen position and still count as "the same gate"
LOCK_RELEASE_DIST   = 1.3    # m; inside this range we treat the locked gate as passed
LATERAL_CUTOFF_DIST = 2.0    # m; inside this range, stop steering and fly straight through
GAP_SMOOTH_ALPHA    = 0.35   # EMA weight on the newest gap sample (0-1, higher = less smoothing)
MAX_COAST_FRAMES    = 20     # ~1s at LOOP_DT=0.05; force-unlock if we've been coasting
                              # this long with no reliable match (gate passed/left the frame)
# physics.py's spring (ATTRACT_K=3.0, DRAG=2.5) is underdamped
# (DRAG^2 - 4*ATTRACT_K < 0), so any target that JUMPS causes the drone to
# overshoot and ring before settling. The distance-scaled target above can
# jump several metres in one frame (e.g. right after locking a far-off
# gate), which was exciting that ringing and sending the drone past
# centreline into the gate frame. Rate-limiting how fast the commanded
# target itself is allowed to move turns that jump into a smooth ramp —
# gentle on the spring — while still reaching the correct position well
# before the drone arrives at the gate.
MAX_TARGET_STEP_X  = 0.5     # m per loop iteration (~10 m/s of target motion)
MAX_TARGET_STEP_Y  = 0.35    # m per loop iteration
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


def _rate_limit(current, desired, max_step):
    """Move `current` toward `desired` by at most `max_step` this call.

    Converts a target that would otherwise jump straight to `desired` (a
    step input to physics.py's underdamped spring) into a bounded ramp —
    see MAX_TARGET_STEP_X/Y above."""
    delta = desired - current
    if delta > max_step:
        delta = max_step
    elif delta < -max_step:
        delta = -max_step
    return current + delta


def pick_gate(cands, locked, locked_center_px):
    """Choose which detected gate to track this frame.

    - Not locked: take the nearest gate. vis.candidates() is already sorted
      largest-box-first (== nearest-first), so cands[0] is the closest gate
      currently in view — preferring proximity over confidence.
    - Locked: pick whichever candidate is spatially closest to where the
      locked gate was last seen, so a second (possibly larger-looking) gate
      further down the track can't steal the target mid-pass. If nothing is
      close enough (gate briefly clipped by the frame edge while flying
      through it), return None so the caller coasts on the last known
      reading instead of re-acquiring a different gate.
    """
    if not cands:
        return None
    if not locked or locked_center_px is None:
        return cands[0]
    best, best_d = None, None
    for c in cands:
        d = ((c.center_px[0] - locked_center_px[0]) ** 2 +
             (c.center_px[1] - locked_center_px[1]) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best, best_d = c, d
    return best if best_d is not None and best_d <= LOCK_MATCH_MAX_PX else None


# ── startup ───────────────────────────────────────────────────────────────────
reset()
time.sleep(1.0)
load_intrinsics()

locked = False
locked_center_px = None
smoothed_gap_x = smoothed_gap_y = 0.0
last_gap_x = last_gap_y = 0.0
last_dist = 3.0
coast_frames = 0
_start_status = get_status()
target_x_state = _start_status["position"]["x"]
target_y_state = _start_status["position"]["y"]

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
        locked = False
        locked_center_px = None
        smoothed_gap_x = smoothed_gap_y = 0.0
        last_gap_x = last_gap_y = 0.0
        last_dist = 3.0
        coast_frames = 0
        _reset_status = get_status()
        target_x_state = _reset_status["position"]["x"]
        target_y_state = _reset_status["position"]["y"]
        continue

    frame = get_frame()
    if frame is None:
        time.sleep(LOOP_DT)
        continue

    pos = status["position"]

    cands = vis.candidates(frame)
    chosen = pick_gate(cands, locked, locked_center_px)

    if chosen is not None and not chosen.clipped:
        gap_x, gap_y, dist = chosen.gap_x, chosen.gap_y, chosen.distance
        det_source = chosen.source
        locked = True
        locked_center_px = chosen.center_px
        last_gap_x, last_gap_y, last_dist = gap_x, gap_y, dist
        coast_frames = 0
    elif chosen is not None and chosen.clipped:
        # bbox touches the frame border -> the gate is only partially in
        # view (almost always because we're now very close to it). Its
        # apparent width is truncated, which inflates the computed distance
        # and its centre is skewed toward whichever side is still visible —
        # exactly the bad reading that was causing the late steer-and-clip.
        # Keep tracking it (update locked_center_px so matching doesn't
        # break) but don't trust its numbers: coast on the last good gap and
        # clamp distance down so the lateral cutoff below is guaranteed to
        # engage, since a clipped box is itself proof we're already close.
        locked = True
        locked_center_px = chosen.center_px
        gap_x, gap_y = last_gap_x, last_gap_y
        dist = min(last_dist, LATERAL_CUTOFF_DIST)
        det_source = chosen.source + "*clip"
        coast_frames = 0
    elif locked:
        # locked gate not matched this frame (occlusion / momentarily out of
        # frame while flying through it) -> coast on the last reading rather
        # than snapping onto whatever else is in view. If this drags on too
        # long the gate is behind us, not just briefly hidden -> release the
        # lock so the next visible gate can be acquired (requirement 2).
        gap_x, gap_y, dist = last_gap_x, last_gap_y, last_dist
        det_source = "coast"
        coast_frames += 1
        if coast_frames > MAX_COAST_FRAMES:
            locked = False
            locked_center_px = None
            coast_frames = 0
    else:
        gap_x, gap_y, dist = 0.0, 0.0, 3.0
        det_source = "none"

    have_gate = det_source != "none"

    # exponential smoothing to damp per-frame pixel jitter in the raw gap
    smoothed_gap_x = GAP_SMOOTH_ALPHA * gap_x + (1 - GAP_SMOOTH_ALPHA) * smoothed_gap_x
    smoothed_gap_y = GAP_SMOOTH_ALPHA * gap_y + (1 - GAP_SMOOTH_ALPHA) * smoothed_gap_y

    # unlock once we've effectively reached the gate plane -> next iteration
    # is free to lock onto whichever gate is now nearest
    if locked and dist <= LOCK_RELEASE_DIST:
        locked = False
        locked_center_px = None

    # ── x: lateral correction toward gate centre, in *world* units ─────────
    # gap_x is a normalised screen fraction, not a distance — multiplying it
    # by a small constant gain gave a fixed ~0.2m nudge regardless of how far
    # off-axis the gate actually was, which was nowhere near enough for a
    # gate sitting far to one side (it ran out of forward distance before
    # catching up laterally). Converting through the pinhole model instead:
    # with FOV_DEG=90 the horizontal conversion is exact —
    #   world_x_offset = gap_x * distance
    # — a gate that's 4m to the right at 12m out now produces a target ~4m
    # to the right, instead of taking dozens of frames to crawl there at a
    # flat 0.2m/step.
    if dist <= LATERAL_CUTOFF_DIST:
        # this close, further steering fights the spring physics into an
        # overshoot instead of helping — hold heading and fly straight through
        desired_x = pos["x"]
    else:
        desired_x = pos["x"] + smoothed_gap_x * dist * LATERAL_GAIN
    # Rate-limit the *commanded* target rather than jumping straight to
    # desired_x — physics.py's spring is underdamped, so a target that jumps
    # several metres in one frame causes the drone to overshoot past
    # centreline and clip the gate frame on the way through. Ramping the
    # target instead keeps the spring's input smooth while still reaching
    # the correct position well before the drone arrives at the gate.
    target_x_state = _rate_limit(target_x_state, desired_x, MAX_TARGET_STEP_X)
    target_x = target_x_state

    # ── y: hover bias (fight gravity) + steer toward gate centre ──────────
    # Same world-unit conversion for the vertical axis. The vertical FOV
    # isn't equal to the horizontal FOV (frame is 640x480, not square), so
    # the conversion factor picks up the frame's height/width ratio.
    if have_gate:
        vertical_scale = vis.FRAME_H / float(vis.FRAME_W)
        altitude_correction = smoothed_gap_y * dist * vertical_scale * ALTITUDE_GAIN
    else:
        altitude_correction = (CRUISE_ALTITUDE - pos["y"]) * 0.3
    desired_y = pos["y"] + HOVER_BIAS + altitude_correction
    desired_y = max(1.0, min(12.0, desired_y))   # safety clamp
    target_y_state = _rate_limit(target_y_state, desired_y, MAX_TARGET_STEP_Y)
    target_y = target_y_state

    # ── z: forward thrust, ease off when badly off-centre ─────────────────
    if have_gate:
        z_step = Z_STEP_MIN + (Z_STEP_MAX - Z_STEP_MIN) * min((dist - 1.0) / 4.0, 1.0)
        if abs(smoothed_gap_x) + abs(smoothed_gap_y) > 0.5:
            z_step *= 0.6
    else:
        z_step = Z_STEP_SEARCH
    target_z = pos["z"] + z_step

    print(f"[{det_source:8}] gap=({smoothed_gap_x:+.2f},{smoothed_gap_y:+.2f}) dist={dist:4.1f}m "
          f"pos=({pos['x']:+.2f},{pos['y']:.2f},{pos['z']:.2f}) "
          f"spd={status.get('speed', 0):.2f} passed={status.get('gates_passed', 0)}")

    move(target_x, target_y, target_z)
    time.sleep(LOOP_DT)