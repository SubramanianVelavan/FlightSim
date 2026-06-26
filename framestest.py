# test_frame.py
import requests

BASE = "http://127.0.0.1:5000"

frame = requests.get(f"{BASE}/frame")

with open("frame.jpg", "wb") as f:
    f.write(frame.content)

print("saved")

