from ultralytics import YOLO
import numpy as np
import cv2 

model = YOLO(r"C:\intern\runs\detect\train-12\weights\best.pt")

def predict_gap(frame):
    small = cv2.resize(frame, (320, 240))
    results = model(small, conf=0.15)   

    boxes = results[0].boxes.xyxy.cpu().numpy()

    print("Detections:", len(boxes))

    if len(boxes) == 0:
        return 0.0, 0.0, 2.0

    best_box = max(
        boxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
    )

    x1, y1, x2, y2 = map(int, best_box)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    gap_x = (cx - frame.shape[1]/2) / (frame.shape[1]/2)
    gap_y = (frame.shape[0]/2 - cy) / (frame.shape[0]/2)

    box_width = x2 - x1
    dist = max(1.0, 300 / box_width)

    return gap_x, gap_y, dist