"""Quick perception smoke-test: run the full vis.py pipeline on a saved frame.

    python framestest.py   # grabs a frame.jpg from the running sim
    python check.py        # runs YOLO+SAM (or HSV fallback) on it
"""

import sys

import cv2

import vis

path = sys.argv[1] if len(sys.argv) > 1 else "frame.jpg"
frame = cv2.imread(path)
if frame is None:
    raise SystemExit(f"could not read {path}")

det = vis.detect(frame)
print(det)

if det.found:
    cx, cy = map(int, det.center_px)
    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"{det.source} {det.distance:.1f}m", (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("detect_out.jpg", frame)
    print("annotated -> detect_out.jpg")
