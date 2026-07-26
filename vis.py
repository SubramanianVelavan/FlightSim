"""
Obstacle (gate) perception for the FPV sim.

Pipeline, best → fallback:

    1. YOLO  (ultralytics)  detects the gate  -> bounding box
    2. SAM 3D (Meta)        segments the gate  -> precise mask + centre
    3. Analytic distance    from the camera intrinsics + known gate width

Every stage degrades gracefully so the whole thing still runs on a plain
CPU box with no model weights:

    - No ultralytics / no weights  -> classical HSV "orange gate" detector
    - No SAM                       -> centre/size taken from the box/contour

Distance is computed from the pinhole camera model, which is exact here
because we know both the camera focal length and the real gate width
(served by the sim at GET /gates):

        distance_m = real_width_m * focal_px / apparent_width_px

SAM 3D, when present, only sharpens the mask (hence the apparent width and
centre) — it does not change the distance *formula*. If you'd rather use
SAM 3D's own metric depth, drop it into Detection.distance in _refine_with_sam.

Configuration (environment variables, all optional):

    YOLO_WEIGHTS   path to a trained YOLO .pt   (default: bundled / skip)
    YOLO_CONF      detection confidence          (default: 0.15)
    SAM_CHECKPOINT path to a SAM / SAM-3D ckpt   (default: skip SAM)
    SAM_MODEL_CFG  SAM model config name         (default: sam2 hiera-large)
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

# ── camera / gate priors (overwritten by set_intrinsics() from /gates) ──────────
FOCAL_PX      = 320.0   # focal length in px for a 640-wide frame (sim default)
FRAME_W       = 640
FRAME_H       = 480
GATE_WIDTH_M  = 5.0     # real gate opening width (half_w 2.5 * 2)

YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.15"))


@dataclass
class Detection:
    found: bool = False
    gap_x: float = 0.0          # normalised horizontal offset of centre  [-1, 1]
    gap_y: float = 0.0          # normalised vertical offset of centre    [-1, 1] (+ = up)
    center_px: tuple = (0.0, 0.0)
    width_px: float = 0.0       # apparent gate width in pixels
    distance: float = 2.0       # metres, camera POV -> obstacle centre
    conf: float = 0.0
    source: str = "none"        # "yolo+sam" | "yolo" | "hsv" | "none"


def set_intrinsics(focal_px=None, frame_w=None, frame_h=None, gate_width_m=None):
    """Feed real camera intrinsics + gate size (from the sim's /gates endpoint)."""
    global FOCAL_PX, FRAME_W, FRAME_H, GATE_WIDTH_M
    if focal_px:     FOCAL_PX = float(focal_px)
    if frame_w:      FRAME_W = int(frame_w)
    if frame_h:      FRAME_H = int(frame_h)
    if gate_width_m: GATE_WIDTH_M = float(gate_width_m)


# ── lazy model handles ──────────────────────────────────────────────────────────
_yolo = None
_yolo_tried = False
_sam = None
_sam_tried = False


def _load_yolo():
    global _yolo, _yolo_tried
    if _yolo_tried:
        return _yolo
    _yolo_tried = True
    weights = os.environ.get("YOLO_WEIGHTS")
    if not weights or not os.path.exists(weights):
        print("[vis] YOLO weights not found (set YOLO_WEIGHTS) — using HSV fallback")
        return None
    try:
        from ultralytics import YOLO
        _yolo = YOLO(weights)
        print(f"[vis] YOLO loaded: {weights}")
    except Exception as e:  # ultralytics missing / load error
        print(f"[vis] YOLO unavailable ({e}) — using HSV fallback")
        _yolo = None
    return _yolo


def _load_sam():
    global _sam, _sam_tried
    if _sam_tried:
        return _sam
    _sam_tried = True
    ckpt = os.environ.get("SAM_CHECKPOINT")
    if not ckpt or not os.path.exists(ckpt):
        return None
    try:
        # Meta SAM 3D / SAM 2 image predictor. Package/entrypoint names vary by
        # release; this targets the sam2 image-predictor API that SAM 3D ships.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg = os.environ.get("SAM_MODEL_CFG", "sam2_hiera_l.yaml")
        model = build_sam2(cfg, ckpt)
        _sam = SAM2ImagePredictor(model)
        print(f"[vis] SAM loaded: {ckpt}")
    except Exception as e:
        print(f"[vis] SAM unavailable ({e}) — centre from box/contour only")
        _sam = None
    return _sam


# ── detection stages ─────────────────────────────────────────────────────────────
def _detect_yolo(frame):
    """Return (x1, y1, x2, y2, conf) of the most prominent gate, or None."""
    model = _load_yolo()
    if model is None:
        return None
    results = model(frame, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(np.argmax(areas))
    x1, y1, x2, y2 = xyxy[i]
    return float(x1), float(y1), float(x2), float(y2), float(confs[i])


def _detect_hsv(frame):
    """Classical fallback: the sim's gates render bright orange. Find the largest
    orange contour and return its bounding box. Returns (x1,y1,x2,y2,conf) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # orange ~ (255,140,0) BGR -> hue band around 15-30 in OpenCV's 0-179 scale
    mask = cv2.inRange(hsv, (8, 120, 120), (30, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 80:
        return None
    x, y, w, h = cv2.boundingRect(c)
    # confidence ~ how much of the box the orange fills (gate frames are hollow,
    # so this is naturally moderate, which is fine).
    fill = cv2.contourArea(c) / max(1.0, w * h)
    return float(x), float(y), float(x + w), float(y + h), float(min(1.0, fill + 0.3))


def _detect_yolo_all(frame):
    """Return every gate box the model sees this frame (not just the largest).

    Needed so the controller can look at *all* gates simultaneously in view
    and decide for itself which one to keep tracking (gate-lock), instead of
    vis.py silently collapsing to a single 'best' box every call."""
    model = _load_yolo()
    if model is None:
        return []
    results = model(frame, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    return [(float(x1), float(y1), float(x2), float(y2), float(c))
            for (x1, y1, x2, y2), c in zip(xyxy, confs)]


def _detect_hsv_all(frame):
    """Classical fallback: return every orange-gate contour box in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (8, 120, 120), (30, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < 80:
            continue
        x, y, w, h = cv2.boundingRect(c)
        fill = cv2.contourArea(c) / max(1.0, w * h)
        out.append((float(x), float(y), float(x + w), float(y + h), float(min(1.0, fill + 0.3))))
    return out


def _box_to_detection(frame, box, source):
    """Turn a raw (x1,y1,x2,y2,conf) box into a Detection (no SAM refine — SAM
    is single-target and only makes sense once we already know which gate we
    care about, so multi-gate candidates() skips it for speed)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2, conf = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width_px = max(1.0, x2 - x1)
    focal = FOCAL_PX * (w / float(FRAME_W))
    distance = GATE_WIDTH_M * focal / width_px
    gap_x = (cx - w / 2.0) / (w / 2.0)
    gap_y = (h / 2.0 - cy) / (h / 2.0)
    return Detection(found=True, gap_x=gap_x, gap_y=gap_y, center_px=(cx, cy),
                      width_px=width_px, distance=distance, conf=conf, source=source)


def candidates(frame):
    """Every visible gate this frame, nearest (largest box) first.

    Unlike detect(), which collapses to a single 'best' box, this exposes
    every candidate so the controller can implement gate-lock: match against
    the previously-locked gate's position instead of always grabbing
    whichever box happens to look biggest/most confident this frame.
    """
    boxes = _detect_yolo_all(frame)
    source = "yolo"
    if not boxes:
        boxes = _detect_hsv_all(frame)
        source = "hsv"
    if not boxes:
        return []
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)  # largest = nearest
    return [_box_to_detection(frame, b, source) for b in boxes]


def _refine_with_sam(frame, box):
    """Use SAM (box prompt) to get a tight mask; return (cx, cy, width_px) or None."""
    predictor = _load_sam()
    if predictor is None:
        return None
    try:
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            box=np.array(box[:4], dtype=np.float32), multimask_output=False
        )
        mask = masks[0].astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        cx, cy = float(xs.mean()), float(ys.mean())
        width_px = float(xs.max() - xs.min())
        return cx, cy, width_px
    except Exception as e:
        print(f"[vis] SAM refine failed ({e})")
        return None


def detect(frame):
    """Run the full perception pipeline on a BGR frame -> Detection."""
    h, w = frame.shape[:2]

    box = _detect_yolo(frame)
    source = "yolo"
    if box is None:
        box = _detect_hsv(frame)
        source = "hsv"
    if box is None:
        return Detection(found=False)

    x1, y1, x2, y2, conf = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width_px = x2 - x1

    sam = _refine_with_sam(frame, box)
    if sam is not None:
        cx, cy, width_px = sam
        source += "+sam"

    # Distance from the pinhole model. Scale focal to *this* frame's width in case
    # the frame isn't the reference 640 wide.
    focal = FOCAL_PX * (w / float(FRAME_W))
    width_px = max(1.0, width_px)
    distance = GATE_WIDTH_M * focal / width_px

    gap_x = (cx - w / 2.0) / (w / 2.0)
    gap_y = (h / 2.0 - cy) / (h / 2.0)

    return Detection(
        found=True, gap_x=gap_x, gap_y=gap_y, center_px=(cx, cy),
        width_px=width_px, distance=distance, conf=conf, source=source,
    )


def predict_gap(frame):
    """Backward-compatible shim: returns (gap_x, gap_y, distance)."""
    d = detect(frame)
    if not d.found:
        return 0.0, 0.0, 2.0
    return d.gap_x, d.gap_y, d.distance