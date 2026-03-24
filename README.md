![Project Thumbnail](milad-fakurian-sin5WZzF1U0-unsplash.jpg)

# 🎬 1-Minute Live Demo
https://drive.google.com/file/d/1ADcJ2MZzeHz0UhCKbCBBpWKSe56JtM7-/view?usp=sharing


## 📑 Table of Contents

### Overview
- [Summary](#summary)
- [What this project demonstrates](#what-this-project-demonstrates)
- [How to run it (entrypoints)](#how-to-run-it-entrypoints)

### Environment & Structure
- [Tested Environment](#tested-environment)
- [Repository Overview](#repository-overview)

### Architecture
- [Core Engine Architecture](#core-engine-architecture-enginepy)
- [Configuration Cheatsheet](#configuration-cheatsheet)

### Performance
- [Performance Notes & Tips](#performance-notes--tips)

### Setup
- [Install dependencies](#1-install-dependencies)
- [MySQL setup](#2-mysql-setup)
- [Download / place model weights](#3-download--place-model-weights)

### Training
- [Training the YOLO License Plate Detector](#training-the-yolo-license-plate-detector)

### Usage
- [Inference Entry Points](#inference-entry-points)

### Advanced
- [Extending the Project](#extending-the-project)
  

# Summary
This is a **full, production‑style license plate recognition system**, not just a single model.

- **End‑to‑end flow**: raw video → **YOLO detection** → **ByteTrack multi‑object tracking** → **plate OCR with strict post‑filtering** → **MySQL persistence + cropped plate images**.
- **Real‑time aware**: explicit frame pacing + catch‑up/drop‑frame logic to keep live streams aligned (avoid latency build‑up).
- **Tracking stability & data coherency**: dropped frames can split the same physical plate into multiple `track_id`s; the engine chooses top candidate by best-size for each track, de‑duplicates `best_text` across different track ids, and commits only after the plate disappears from all candidates for about **90 frames**.
- **Robust OCR layer**: grayscale plate preprocessing (2x upscale + sharpen), OCR throttling, `OCR_CONF` min-confidence filtering, and strict text rules (unique characters, digit/letter balance, minimum length).
- **Data‑ready outputs**: structured DB schema plus run‑organized plate crops so detections are easy to query, debug, and feed into dashboards or downstream services.

### What this project demonstrates

- **System‑level computer vision**: integrates detection, tracking, OCR, and storage with attention to latency, quality, and failure modes.
- **Real engineering trade‑offs options**:
  - Real‑time vs offline processing (pacing vs throughput, when to drop frames).
  - Connection pooling vs single connections for different workloads.
  - Practical pre‑processing (2x upscale + grayscale + sharpen; CLAHE tested but disabled) tuned for OCR robustness.
- **Extensible architecture**: pluggable OCR backend and tracker config; DB write path that can be swapped for APIs, queues, or other sinks.

### How to run it (entrypoints)

- **Live GUI demo** – overlay + FPS + DB + crops: `python stream_inference.py`  
- **Headless real‑time stream** – no window, DB + crops: `python stream_headless_inference.py`  
- **Offline batch (file → file)** – annotated video + DB + crops: `python offline_batch_inference.py`  

---
## Tested Environment

- Python 3.11
- PyTorch 2.6.0 (CUDA 12.4)
- GPU: NVIDIA (CUDA 12.x compatible)
- OS: Linux / Windows

## Repository Overview

- **`requirements.txt`** – Core dependencies including PyTorch (CUDA 12.4 builds). Install with `pip install -r requirements.txt`.
- **`environment.yaml`** – Conda env `lp`: Python 3.11, PyTorch 2.6.0 and CUDA 12.4 (via pytorch-cuda) plus all required dependencies. Use with `conda env create -f environment.yaml` for a single-command setup.
- `train.py` – YOLO training entrypoint for the plate detector.
- `engine.py` – **core engine**: model/DB initialization, frame loop, pacing, detection, ByteTrack ID handling, OCR, temporal filtering, DB inserts, and result image export.
- `stream_inference.py` – interactive streaming script that wraps `engine.run` with a GUI (FPS + boxes + text).
- `offline_batch_inference.py` – file‑to‑file batch processing via `engine.run`, uses DB connection pooling and writes annotated video to `OUT_VIDEO`.
- `stream_headless_inference.py` – real‑time streaming inference via `engine.run` without any GUI.
- `model/custom_bytetrackv2.yaml` – ByteTrack tracker configuration used by `engine.run`.
- `runs/detect/train9/weights/last.pt` – default detector checkpoint path after training (path is configurable in the front‑end scripts).
- `utils/export_TensorRT.py` – helper for exporting models to TensorRT (optional).
---

## Core Engine Architecture (`engine.py`)

The `run(...)` function in `engine.py` is the **single source of truth** for the pipeline; all front‑ends just configure and call it.

### 1. Model and DB initialization

- **YOLO detector + ByteTrack**:
  - Loaded via `YOLO(model_weights)` in `init_models`.
  - Tracking is done with `model.track(frame, tracker=tracker, persist=True, imgsz=imgsz, conf=conf, verbose=False)[0]`.
  - `persist=True` keeps **track IDs** stable across frames.
  - `model/custom_bytetrackv2.yaml` uses `track_buffer: 90` so tracks tolerate longer gaps during dropped-frame catch-up.
- **OCR model**:
  - Uses `fast-plate-ocr` (`LicensePlateRecognizer`) for fast CPU OCR.
  - Instantiated as `LicensePlateRecognizer(ocr_model_name)`.
  - `ocr.run(plate_gray)` may return a list or string; the engine normalizes to a string, uppercases it, and strips non‑alphanumeric characters.
  - If `OCR_CONF` is set, low-confidence OCR candidates are rejected using the mean per-character confidence.
- **DB connection** (`init_db`):
  - Either a single `mysql.connector.connect(..., autocommit=True)` or a `MySQLConnectionPool` when `db_pool=True`.

### 2. Frame pacing and timing

To support both **real‑time streaming** and **offline batch** processing:

- `get_src_fps(cap)` – pulls source FPS, defaults to `30.0` if invalid.
- `pace_video(t0, frame_i, frame_period)` – sleeps so frame `i` aligns with wall‑clock time since `t0`.
- `behind_catchup(cap, t0, frame_i, frame_period)` – when processing lags behind, repeatedly calls `cap.grab()` to drop frames until delay is within one frame.

Effect:

- `realtime=True` → pacing + frame dropping to approximate real time.
- `realtime=False` → no waiting, maximum throughput.

### 3. Detection, tracking, and plate cropping

Each frame:

1. `results = model.track(...)`  
2. Extract:
   - `cls` – class indices
   - `xyxy` – bounding boxes
   - `ids` – track IDs via `extract_ids_numpy`, which handles missing / weird `boxes.id` cases.
3. For each detection:
   - Skip non‑plate classes: `if c != 0: continue`.
   - Clamp box to `[0, w] × [0, h]`.
   - Compute width/height and area; discard if below `min_plate_w` or `min_plate_h`.
   - Optionally draw a green rectangle when `draw_gui=True` or a writer is active.

### 4. OCR pre‑processing and quality control

**Pre‑processing** (`preprocess_plate_for_ocr`):

- Upscale crop (x2).
- Convert to grayscale, then back to 3-channel BGR (keeps preprocessing benefits while matching OCR input expectations).
- Sharpen with Gaussian blur + weighted subtraction (`cv2.addWeighted`).
- Note: CLAHE was tested but performed slightly worse, so it is disabled in the current engine.

**OCR and filtering** (`ocr_plate_text`):

- Call `ocr.run(plate_gray)`, normalize:
  - Uppercase.
  - Strip to alphanumeric.
- If `OCR_CONF > 0`, reject candidates when the mean character confidence is below the threshold.
- Reject candidates when:
  - **≤ 2 unique characters**.
  - All characters are digits **or** all are letters.
  - Length `< min_ocr_chars_len`.

Only survivors pass into the temporal cache.

### 5. Temporal aggregation and “best sample” selection

`ocr_cache` (per `track_id`) stores:

- `best_text`, `best_width`, `best_crop`
- `last_frame`, `last_ocr_frame`

**OCR throttling**:

- Run OCR only when `(frame_i - cached["last_ocr_frame"]) >= ocr_every_frames`.

**Best sample update**:

- If a valid candidate exists:
  - Compute relative width gain:
    - `diff = (bw - old_w) / float(old_w)` if `old_w > 0` else `1.0`.
  - Update when:
    - `old_w == 0`, or
    - `bw > old_w and diff > area_eps_ratio`.

This keeps the sharpest / largest view as the canonical sample.

### 6. DB insertion and cache flushing
The writes are event-based and de-duplicated across track IDs:

- Periodically, when some cached candidate has been missing for `>= 90` frames, the engine groups `ocr_cache` entries by their current `best_text`.
- For each `plate_text` group, it waits until **all** `track_id`s in that group have been missing for `>= 90` frames (deadzone).
- Then it commits exactly one row:
  - Pick the best `track_id` by largest `best_width` (tie-breaker: more recent `last_frame`).
  - Save the best crop to disk and insert into DB via `insert_plate(...)`.
  - Remove all `track_id`s in that `plate_text` group from `ocr_cache`.

On shutdown, any remaining tracks with `best_text` are flushed the same way.

DB errors are logged as `[DB ERROR] ...` and do **not** abort the loop.

### 7. Visualization & output

- If `draw_gui=True`:
  - Draw bounding boxes and plate text boxes.
  - Overlay smoothed FPS.
  - Resize for display using `fit_for_screen(...)` and show in a window named `"plate"`.
  - Exit cleanly on **Esc**.
- If `save_video_path` is set:
  - Initialize `cv2.VideoWriter` at original resolution and FPS.
  - Write frames with all overlays.

On exit, all OpenCV handles and DB resources are released.

---


## Configuration Cheatsheet

- **Detector & Tracker**
  - `MODEL_WEIGHTS` – YOLO weights path in each front‑end script.
  - `TRACKER` – ByteTrack config YAML (`model/custom_bytetrackv2.yaml`).
  - `CONF` – detection confidence threshold.
  - `IMGSZ` – inference resolution (higher is more accurate but slower).
- **OCR**
  - `OCR_MODEL_NAME` – OCR model name for `LicensePlateRecognizer`.
  - `OCR_CONF` – minimum OCR confidence (mean per-character confidence) to accept a candidate.
  - `MIN_OCR_CHARS_LEN` – minimum length of accepted OCR string.
  - `OCR_EVERY_FRAMES` – throttle period for OCR attempts per `track_id`.
  - `AREA_EPS_RATIO` – how much larger the plate crop must be to replace the current best sample.
  - `MIN_PLATE_W`, `MIN_PLATE_H` – minimum plate crop size (in pixels) to even attempt OCR.
- **DB**
  - `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_TABLE`.
  - `db_pool` and `db_pool_size` (engine parameters) to toggle and size connection pooling.
- **Runtime & Output**
  - `draw_gui` – whether to draw GUI overlays and show a live window.
  - `realtime` – whether to pace frames and drop when lagging.
  - `save_video_path` – path for annotated output video, or `None`.
  - `result_images_root` – parent folder where auto‑incremented run directories of cropped plates are stored.

---

## Performance Notes & Tips

- **GPU vs CPU**: For high‑resolution (`IMGSZ` ≥ 1408) and multi‑object tracking, GPU is strongly recommended. On CPU you may need to:
  - Lower `IMGSZ` (e.g., 640)
  - Increase `OCR_EVERY_FRAMES` to reduce OCR load
  - Possibly disable real‑time pacing (`realtime=False`) for offline batches.
- **DB throughput**:
  - For offline processing, prefer `db_pool=True` in `offline_batch_inference.py`.
  - Keep `DB_TABLE` indexed on `ts` and/or `plate_text` for faster querying.
- **OCR quality**:
  - The **pre‑processing pipeline** (2x upscale + grayscale + sharpen) is tuned for typical plate imagery; depending on your region/plates you may want to adjust crop scaling and sharpening strength.
  - `MIN_OCR_CHARS_LEN` and uniqueness / digit/letter checks are conservative filters to reduce false positives; you can relax them in controlled environments.

---

### 1. Install dependencies

You can use **Conda** (recommended) or **pip only**. Both are pinned so the project runs the same on your machine.


#### Option A: Conda 

The repo includes an **`environment.yaml`** that builds the `lp` env. It uses **PyTorch 2.6.0** with **CUDA 12.4** support.

```bash
conda env create -f environment.yaml
conda activate lp
```

- **CUDA**: If your driver uses an older toolkit, edit the yaml: e.g. `pytorch-cuda=12.1` or `pytorch-cuda=11.8` for CUDA 11.
- **OCR**: Uses **`onnxruntime`** (CPU) by default. For this pipeline, CPU OCR gave me better FPS than GPU OCR


#### Option B: Pip only (`requirements.txt`)

If you already have Python 3.11 or prefer a venv you could use:

```bash
pip install -r requirements.txt
```

- **`requirements.txt`** pins all core dependencies: `ultralytics`, `fast-plate-ocr`, `mysql-connector-python`, `opencv-python`, `numpy`, `protobuf`, `onnxruntime`. PyTorch and torchvision (CUDA 12.4 builds) included, so no separate PyTorch installation is required.
- You can run inference on CPU only, but FPS will be very low. Also the OCR runs on CPU via `onnxruntime` and does not require CUDA.


> **Note**: In `engine.py`, `ORT_TENSORRT_UNAVAILABLE=1` is set so the OCR model does not try to load TensorRT (avoids slow startup and DLL issues). You can change this if you use TensorRT and onnxruntime-gpu on purpose.

### 2. MySQL setup

Create a database and a table for decoded plate records. The **current engine** (`engine.py`) inserts:

- `track_id` – integer track identifier (ByteTrack ID)
- `plate_text` – normalized plate text
- `best_width` – width (in pixels) of the best plate crop used for DB insert
- `ts` – timestamp at insert time
- `image_path` – path to stored cropped plate image

An example table schema:

```sql
CREATE TABLE plates (
  id INT AUTO_INCREMENT PRIMARY KEY,
  track_id INT NOT NULL,
  plate_text VARCHAR(32) NOT NULL,
  best_width INT NOT NULL,
  ts DATETIME NOT NULL,
  image_path VARCHAR(255) NOT NULL
);
```

Configure connectivity by editing DB constants in the front‑end scripts:

- `stream_inference.py`, `stream_headless_inference.py`, `offline_batch_inference.py`

### 3. Download / place model weights

By default, the detector is expected at:

- `runs/detect/train9/weights/last.pt`

You can either:

- Train your own detector with `train.py` (see below), or
- Drop in a compatible YOLO weights file and point the config constants to it.

The OCR model name is configured by `ocr_model_name` in the entrypoints (passed into `LicensePlateRecognizer(...)`):

- `stream_inference.py` defaults to `"cct-s-v2-global-model"`.
- `stream_headless_inference.py` and `offline_batch_inference.py` default to `"cct-s-v1-global-model"` (change `OCR_MODEL_NAME` in those scripts if you want v2 everywhere).

Make sure the OCR backend you are using supports this model name.

---

## Training the YOLO License Plate Detector

The training script is intentionally minimal and uses Ultralytics’ high‑level API:

- **Script**: `train.py`
- **Key hyperparameters**:
  - `data="data/data.yaml"` – path to your YOLO dataset configuration
  - `epochs=50`
  - `patience=7`
  - `batch=14`
  - `imgsz=960`
  - `save_period=3` – saves model weights every N epochs

Run:

```bash
python train.py
```

This will train a model and write checkpoints under a `runs/detect/...` directory (by default `runs/detect/train9/weights/last.pt` is used later in inference scripts).

> **Assumption**: Class index **0** in your dataset corresponds to license plates. The engine explicitly filters detections with `if c != 0: continue`, so plates must be class 0.

---

## Inference Entry Points

### 1. `stream_inference.py` – interactive streaming (GUI)

**Purpose**:  
Video in → real‑time YOLO + ByteTrack + OCR → live overlay window + MySQL + plate crops.

- **Key configs**:
  - `TRACKER` – e.g. `model/custom_bytetrackv2.yaml`
  - `VID_IN` – path to input video or camera index
  - `RESULT_IMAGES_ROOT` – root folder for cropped plate images
  - `DISPLAY_W`, `DISPLAY_H` – GUI display resolution
  - `CONF`, `IMGSZ`, `MODEL_WEIGHTS` – YOLO inference settings
  - `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_TABLE`
  - `OCR_MODEL_NAME`, `MIN_OCR_CHARS_LEN`, `OCR_EVERY_FRAMES`, `AREA_EPS_RATIO`
  - `MIN_PLATE_W`, `MIN_PLATE_H`

**To run**:

```bash
python stream_inference.py
```

Press **Esc** to exit.

Internally wraps:

```python
run(
    draw_gui=True,
    realtime=True,
    save_video_path=None,
    ...
)
```

### 2. `offline_batch_inference.py` – offline batch, file‑to‑file

**Purpose**:  
Process a video file **as fast as possible** (no GUI) into:

- An **annotated output video** (`OUT_VIDEO`)
- Plate crops in `RESULT_IMAGES_ROOT` under an auto‑incremented run directory
- Plate records in MySQL

Key behavior:

- `draw_gui=False`, `realtime=False` → no pacing, full‑speed processing.
- `db_pool=True`, `db_pool_size=DB_POOL_SIZE` → MySQL connection pooling for high insert throughput.

Run:

```bash
python offline_batch_inference.py
```

### 3. `stream_headless_inference.py` – online, headless streaming

**Purpose**:  
Real‑time inference **without** GUI or output video:

- Detection + tracking
- OCR
- DB inserts
- Plate crop export

Key behavior:

- `draw_gui=False`, `realtime=True`, `save_video_path=None`.
- Uses the same `ocr_cache` and DB insertion strategy as other modes.

Run:

```bash
python stream_headless_inference.py
```

Configure **DB credentials** before running (they are left blank in the template).

---

## Extending the Project

- **Different OCR backends**: As long as you provide a `LicensePlateRecognizer`‑like object with a `.run(image_gray)` method returning text (or list of text), the engine will work. Implement your own wrapper and update `init_models`.
- **Alternative tracking**: You can experiment with different tracker configs in the `model/` directory and pass them via the `tracker` argument to `run(...)`.
- **Additional outputs**:
  - Add JSON/CSV logging alongside DB inserts.
  - Push events to a message queue or REST API in `insert_plate` if you need streaming integration with other systems.
