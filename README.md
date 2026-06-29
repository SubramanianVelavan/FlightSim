# FlightSim — Computer-Vision Autonomy Client

The **vision + autonomy** half of the FPV drone project. It does **not** contain the
simulator's logic — it connects to the sim (the `sim` repo) over its REST API,
perceives the gate, and flies the drone through it.

```
sim  (pygame + Flask)  ──/frame──►  controller.py ──► vis.detect() ──► /control
        ▲   server                     this repo (the eyes + brain)        │
        └──────────────────────────────  /status, /gates  ◄───────────────┘
```

## Run

1. Start the **sim** first (in the `sim` repo): `python main.py` — it serves on
   `http://127.0.0.1:5000`.
2. Then, here:
   ```
   pip install -r requirements.txt
   python controller.py
   ```
   Different machine? Point at the sim's host:
   ```
   SIM_BASE=http://<sim-host>:5000 python controller.py
   ```

## Perception pipeline (`vis.py`)

`vis.detect(frame)` returns a `Detection` with the gate's **centre**, **apparent
width**, and **distance from the camera POV** — best backend wins, each stage
degrades gracefully:

1. **YOLO** (ultralytics) detects the gate → bounding box.
2. **SAM 3D** (Meta) segments it with the box as a prompt → tight mask → precise
   centre + width.
3. **Distance** from the pinhole model, exact here because we know the focal
   length and the real gate width (both fetched from the sim's `/gates`):

   ```
   distance_m = gate_width_m * focal_px / apparent_width_px
   ```

If `ultralytics`/weights are absent it falls back to a classical **HSV orange-gate
detector**; if SAM is absent the centre/width come from the box. So it runs on a
plain CPU box with zero models, and lights up YOLO + SAM 3D automatically on the
GPU box once the weights are present.

### Configuration (env vars, all optional)

| Variable         | Meaning                                  | Default              |
|------------------|------------------------------------------|----------------------|
| `SIM_BASE`       | Sim REST base URL                        | `http://127.0.0.1:5000` |
| `YOLO_WEIGHTS`   | Path to trained YOLO `.pt`               | *(skip → HSV)*       |
| `YOLO_CONF`      | Detection confidence                     | `0.15`               |
| `SAM_CHECKPOINT` | Path to SAM / SAM-3D checkpoint          | *(skip SAM)*         |
| `SAM_MODEL_CFG`  | SAM model config name                    | `sam2_hiera_l.yaml`  |

Example (GPU box):
```
set YOLO_WEIGHTS=C:\intern\runs\detect\train-12\weights\best.pt
set SAM_CHECKPOINT=C:\models\sam2_hiera_large.pt
python controller.py
```

## Files

- `vis.py` — perception: YOLO + SAM 3D + analytic distance (with HSV fallback).
- `controller.py` — autonomy loop (perceive → steer through gate).
- `framestest.py` — save one `frame.jpg` from the running sim.
- `check.py` — run the perception pipeline on a saved image (`detect_out.jpg`).
- `control.py` — minimal scratch test of the `/control` endpoint.

> **Note:** this repo also still carries a *copy* of the simulator engine
> (`main.py`, `physics.py`, `renderer.py`, …) from when it was forked. The
> maintained simulator now lives in the `sim` repo (with telemetry + the
> `/gates` endpoint this client uses). Run the sim from there and treat this
> folder as the vision client.
