![Project Thumbnail](milad-fakurian-sin5WZzF1U0-unsplash.jpg)

# 🎬 2-Minute Live Demo
https://drive.google.com/file/d/17nol3fuDRJbh-lqV7N5qbJxSnosGdA_I

# Summary
This is a **full, production‑style license plate recognition system**, not just a single model.

- **End‑to‑end flow**: raw video → **YOLO detection** → **ByteTrack multi‑object tracking** → **plate OCR with strict post‑filtering** → **MySQL persistence + cropped plate images**.
- **Real‑time aware**: explicit frame pacing and catch‑up logic to keep live streams in sync by **dropping frames intelligently** instead of building latency.
- **Robust OCR layer**: event based updates, per‑track temporal aggregation, “best frame” selection, and strong text rules (unique‑character checks, digit/letter balance, minimum length) to cut down false reads and compensate for any frames dropped during live sync.
- **Data‑ready outputs**: structured DB schema plus run‑organized plate crops so detections are easy to query, debug, and feed into dashboards or downstream services.

### What this project demonstrates

- **System‑level computer vision**: integrates detection, tracking, OCR, and storage with attention to latency, quality, and failure modes.
- **Real engineering trade‑offs**:
  - Real‑time vs offline processing (pacing vs throughput, when to drop frames).
  - Connection pooling vs single connections for different workloads.
  - Practical pre‑processing (CLAHE + sharpening) tuned for OCR robustness.
- **Extensible architecture**: pluggable OCR backend and tracker config; DB write path that can be swapped for APIs, queues, or other sinks.

### How to run it (entrypoints)

- **Live GUI demo** – overlay + FPS + DB + crops: `python stream_inference.py`  
- **Headless real‑time stream** – no window, DB + crops: `python stream_headless_inference.py`  
- **Offline batch (file → file)** – annotated video + DB + crops: `python offline_headless_inference.py`  

---
## Core Engine Architecture (`engine.py`)

The `run(...)` function in `engine.py` is the **single source of truth** for the pipeline; all front‑ends just configure and call it.

### 1. Model and DB initialization

- **YOLO detector + ByteTrack**:
  - Loaded via `YOLO(model_weights)` in `init_models`.
  - Tracking is done with `model.track(frame, tracker=tracker, persist=True, imgsz=imgsz, conf=conf, verbose=False)[0]`.
  - `persist=True` keeps **track IDs** stable across frames.
- **OCR model**:
  - Instantiated as `LicensePlateRecognizer(ocr_model_name)`.
  - `ocr.run(plate_bgr)` may return a list or string; the engine normalizes to a string, uppercases it, and strips non‑alphanumeric characters.
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

- Upscale crop (e.g. `fx=1.5, fy=1.5`).
- Convert to grayscale.
- Apply **CLAHE** for local contrast normalization.
- Blur + weighted subtraction (`cv2.addWeighted`) for sharpening.

**OCR and filtering** (`ocr_plate_text`):

- Call `ocr.run(plate_bgr)`, normalize:
  - Uppercase.
  - Strip to alphanumeric.
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
The writes are event-based (saving the best result after the tracked plate disappears for about 90 frames):

- Iterate `ocr_cache`:
  - For each `track_id` with non‑empty `best_text` and `frame_i - last_frame >= 90`:
    - Save crop to disk: `save_best_crop(run_dir, tid, best_text, data["best_crop"])`.
    - Insert into DB via `insert_plate(...)` with `track_id`, `plate_text`, `best_width`, `image_path`, and timestamp.
    - Remove `tid` from cache.

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
## Repository Overview

- **`requirements.txt`** – Pip dependencies (ultralytics, fast-plate-ocr, mysql-connector-python, opencv-python, numpy, protobuf, onnxruntime). Use with `pip install -r requirements.txt` if you already have PyTorch/CUDA.
- **`environment.yaml`** – Conda env `lp`: Python 3.10, PyTorch 2.10 + CUDA 13.0, and the same pip deps. Use with `conda env create -f environment.yaml` for a single-command setup.
- `train.py` – YOLO training entrypoint for the plate detector.
- `engine.py` – **core engine**: model/DB initialization, frame loop, pacing, detection, ByteTrack ID handling, OCR, temporal filtering, DB inserts, and result image export.
- `stream_inference.py` – interactive streaming script that wraps `engine.run` with a GUI (FPS + boxes + text).
- `offline_headless_inference.py` – file‑to‑file batch processing via `engine.run`, uses DB connection pooling and writes annotated video to `OUT_VIDEO`.
- `stream_headless_inference.py` – real‑time streaming inference via `engine.run` without any GUI.
- `model/custom_bytetrackv2.yaml` – ByteTrack tracker configuration used by `engine.run`.
- `runs/detect/train9/weights/last.pt` – default detector checkpoint path after training (path is configurable in the front‑end scripts).
- `utils/export_TensorRT.py` – helper for exporting models to TensorRT (optional).
---

## Configuration Cheatsheet

- **Detector & Tracker**
  - `MODEL_WEIGHTS` – YOLO weights path in each front‑end script.
  - `TRACKER` – ByteTrack config YAML (`model/custom_bytetrackv2.yaml`).
  - `CONF` – detection confidence threshold.
  - `IMGSZ` – inference resolution (higher is more accurate but slower).
- **OCR**
  - `OCR_MODEL_NAME` – OCR model name for `LicensePlateRecognizer`.
  - `MIN_OCR_CHARS_LEN` – minimum length of accepted OCR string.
  - `OCR_EVERY_FRAMES` – frames between OCR attempts for the same track.
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
  - Lower `IMGSZ` (e.g., 960)
  - Increase `OCR_EVERY_FRAMES` to reduce OCR load
  - Possibly disable real‑time pacing (`realtime=False`) for offline batches.
- **DB throughput**:
  - For offline processing, prefer `db_pool=True` in `offline_headless_inference.py`.
  - Keep `DB_TABLE` indexed on `ts` and/or `plate_text` for faster querying.
- **OCR quality**:
  - The **pre‑processing pipeline** (CLAHE + sharpening) is tuned for typical plate imagery; depending on your region/plates you may want to adjust contrast/sharpening kernels.
  - `MIN_OCR_CHARS_LEN` and uniqueness / digit/letter checks are conservative filters to reduce false positives; you can relax them in controlled environments.

---

### 1. Install dependencies

You can use **Conda** (recommended) or **pip only**. Both are pinned so the project runs the same on your machine.

#### Option A: Conda (recommended, includes CUDA for YOLO)

The repo includes an **`environment.yaml`** that defines the `lp` env with Python 3.10, PyTorch 2.10, CUDA 13.0, and all pip dependencies in one go.

```bash
conda env create -f environment.yaml
conda activate lp
```

- **CUDA**: Needed for **YOLO** (training and real-time inference). The yaml uses `pytorch-cuda=13.0` (CUDA 13.x). If your driver uses an older toolkit, edit the yaml: e.g. `pytorch-cuda=12.1` for CUDA 12 or `pytorch-cuda=11.8` for CUDA 11.
- **OCR**: Uses **`onnxruntime`** (CPU) by default. For this pipeline, CPU OCR gave me better FPS than GPU OCR


#### Option B: Pip only (`requirements.txt`)

If you prefer a venv or already have Python 3.10+ and PyTorch (with or without CUDA):

```bash
pip install -r requirements.txt
```

- **`requirements.txt`** pins: `ultralytics`, `fast-plate-ocr`, `mysql-connector-python`, `opencv-python`, `numpy`, `protobuf`, `onnxruntime`. It does **not** install PyTorch or CUDA; 
- **When CUDA is needed**: For **training** (`train.py`) and for **smooth real-time inference** (YOLO detection), a GPU with CUDA is strongly recommended. You can run inference on CPU only, but FPS will be lower. OCR runs on CPU via `onnxruntime` and does not require CUDA.


> **Note**: In `engine.py`, `ORT_TENSORRT_UNAVAILABLE=1` is set so the OCR model does not try to load TensorRT (avoids slow startup and DLL issues). You can change this if you use TensorRT on purpose.

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

- `stream_inference.py`, `stream_headless_inference.py`, `offline_headless_inference.py`

### 3. Download / place model weights

By default, the detector is expected at:

- `runs/detect/train9/weights/last.pt`

You can either:

- Train your own detector with `train.py` (see below), or
- Drop in a compatible YOLO weights file and point the config constants to it.

The OCR model name is configured as:

- `"cct-s-v1-global-model"` in the entrypoints, via `LicensePlateRecognizer("cct-s-v1-global-model")`

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

### 2. `offline_headless_inference.py` – offline batch, file‑to‑file

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
python offline_headless_inference.py
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

- **Different OCR backends**: As long as you provide a `LicensePlateRecognizer`‑like object with a `.run(image_bgr)` method returning text (or list of text), the engine will work. Implement your own wrapper and update `init_models`.
- **Alternative tracking**: You can experiment with different tracker configs in the `model/` directory and pass them via the `tracker` argument to `run(...)`.
- **Additional outputs**:
  - Add JSON/CSV logging alongside DB inserts.
  - Push events to a message queue or REST API in `insert_plate` if you need streaming integration with other systems.

## Deduplication Technique for the realtime-sync (Concept):

I thought of this after publishing the project but didn't try it. it sounds very solid to me:

- create a new cache variable that periodically (each 15 frames) searches for different tracking IDs with the same best_text in our ocr_cache (this can happen when the model was tracking a license plate, then dropped it because of frame drops or something similar, then tracked it again with a new ID)

- if found, temporarily halt the first dropped track from committing, wait until the second one finishes tracking (with the final 90 frame buffer), and commit the best one out of them. Same idea if there are more than 2.

(the ocr_cache already auto-cleans when it commits, so we shouldn't have any noticeable performance loss from the searches)
