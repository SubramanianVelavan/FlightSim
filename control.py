import requests
import time

BASE = "http://127.0.0.1:5000"

# move forward slowly
for i in range(20):
    requests.post(
        f"{BASE}/control",
        json={
            "x": 0.0,
            "y": i * 0.5,
            "z": 1.5,
            "yaw": 0.0
        }
    )
    time.sleep(0.1)

print("Done")