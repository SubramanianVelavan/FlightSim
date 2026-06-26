from ultralytics import YOLO

model = YOLO(r"C:\intern\runs\detect\train-12\weights\best.pt")

results = model("frame.jpg")

results[0].show()

print(results[0].boxes.xyxy)
print(results[0].boxes.cls)