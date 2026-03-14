import os
import numpy as np
import cv2
import time
from datetime import datetime

import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1" # this disables TensorRT (if available), the ocr model takes little bit long to load with it  

def init_models(model_weights, ocr_model_name):
    model = YOLO(model_weights)
    ocr = LicensePlateRecognizer(ocr_model_name)
    return model, ocr   

def init_db(host, port, user, password, database, pool=False, pool_size=8):
    if pool:
        return MySQLConnectionPool(pool_name="lp_pool", pool_size=int(pool_size), host=host, port=port,
                                    user=user, password=password, database=database, autocommit=True)

    conn = mysql.connector.connect(host=host,port=port,user=user,password=password,database=database)
    conn.autocommit = True
    return conn

def get_src_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 0 else 30.0

def pace_video(t0, frame_i, frame_period):
    target_t = t0 + frame_i * frame_period
    now_abs = time.perf_counter()
    if now_abs < target_t:
        time.sleep(target_t - now_abs)

def behind_catchup(cap, t0, frame_i, frame_period):
    expected_time = frame_i * frame_period
    now_time = time.perf_counter() - t0
    while now_time > expected_time + frame_period:
        if not cap.grab():
            break
        frame_i += 1
        expected_time = frame_i * frame_period
        now_time = time.perf_counter() - t0
    return frame_i

def extract_ids_numpy(boxes_cpu, n): # i made this robust because i had some problems with the ids
    ids_t = boxes_cpu.id
    if ids_t is None:
        return np.arange(n, dtype=np.int32)
    if hasattr(ids_t, "int"):
        return ids_t.int().cpu().numpy().astype(np.int32)
    return ids_t.astype(np.int32)

def preprocess_plate_for_ocr(plate_bgr): # upscale + CLAHE + sharpness
    plate = cv2.resize(plate_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) 
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0,0), 1.0)  # sharpness (blur then subtract blur from original) 
    gray = cv2.addWeighted(gray, 1.1, blur, -0.1, 0)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def save_best_crop(run_dir, track_id, plate_text, crop_bgr):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return ""
    fname = f"{int(track_id)}_{_safe_name(plate_text)}.jpg"
    path = os.path.join(run_dir, fname)
    try:
        cv2.imwrite(path, crop_bgr)
        return path
    except Exception:
        return ""

def _next_run_dir(root_dir): # this is just for each run we make a new sub folder /n for the saved cropped images
    os.makedirs(root_dir, exist_ok=True)
    max_n = -1
    for name in os.listdir(root_dir):
        if name.isdigit():
            max_n = max(max_n, int(name))
    run_dir = os.path.join(root_dir, str(max_n + 1))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _safe_name(s): # this just for windows file naming (cropped imagese names)
    s = "".join(ch for ch in str(s) if ch.isalnum() or ch in ("-", "_"))
    return s[:40] if len(s) > 40 else s

def insert_plate(db, track_id, plate_text, best_width, image_path, table):
    ts = datetime.now()
    conn = None
    try:
        conn = db.get_connection() if hasattr(db, "get_connection") else db
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table} (track_id, plate_text, best_width, ts, image_path) VALUES (%s, %s, %s, %s, %s)",
                (int(track_id), plate_text, int(best_width), ts, image_path)
            )
    except Exception as e:
        # don't kill the stream if DB fails, but show the error
        print(f"[DB ERROR] Failed to insert plate {plate_text} (track {track_id}): {e}")
    finally:
        if conn is not None and conn is not db:
            try:
                conn.close()
            except Exception:
                pass

def ocr_plate_text(ocr, plate_bgr, min_ocr_chars_len):
    text = ocr.run(plate_bgr)
    if isinstance(text, list):
        text = text[0]
    candidate = "".join(ch for ch in str(text).upper() if ch.isalnum())

    # reject results with less than 2 unique characters
    if len(set(candidate)) <= 2:
        return ""

    # reject results that are all digits or all letters
    digits = sum(c.isdigit() for c in candidate)
    letters = sum(c.isalpha() for c in candidate)
    if digits == len(candidate) or letters == len(candidate):
        return ""

    if len(candidate) < min_ocr_chars_len:
        return ""

    return candidate

def draw_text_box(frame, text, x1, y1, x2, y2):
    bw = x2 - x1
    bh = y2 - y1

    text_box_h = bh
    text_x1 = x1
    text_x2 = x2
    text_y2 = y1 - 10
    text_y1 = text_y2 - text_box_h
    if text_y1 < 0:
        return

    cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), (255, 255, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    while tw > bw - 10 and scale > 0.3:
        scale -= 0.1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    tx = text_x1 + (bw - tw) // 2
    ty = text_y1 + (text_box_h + th) // 2
    cv2.putText(frame, text, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

def draw_fps(frame, fps):
    fps_txt = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    (tw, th), base = cv2.getTextSize(fps_txt, font, scale, thick)
    x, y = 10, 10 + th
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + base + 6), (0, 0, 0), -1)
    cv2.putText(frame, fps_txt, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def fit_for_screen(frame, max_w=1280, max_h=720):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def run(*, show_gui=True, realtime=True, save_video_path=None, result_images_root="result_images",
    vid_in, tracker, model_weights, ocr_model_name, conf, imgsz, min_ocr_chars_len,
    ocr_every_frames, area_eps_ratio, min_plate_w, min_plate_h, db_host, db_port, db_user, db_password, db_name, db_table,
    db_pool=False, db_pool_size=8, display_w=None, display_h=None
    ):
    model, ocr = init_models(model_weights=model_weights, ocr_model_name=ocr_model_name)
    db = init_db(host=db_host, port=db_port, user=db_user, password=db_password, database=db_name, pool=db_pool, pool_size=db_pool_size)
    run_dir = _next_run_dir(result_images_root)

    cap = cv2.VideoCapture(vid_in)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid_in}")

    src_fps = get_src_fps(cap)
    frame_period = 1.0 / float(src_fps)
    frame_i = 0
    t0 = time.perf_counter()

    writer = None
    if save_video_path:
        out_dir = os.path.dirname(save_video_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video_path, fourcc, src_fps, (w, h))

    ocr_cache = {}  # track_id {"best_text": str, "best_width": int, "best_crop": np.ndarray / None, "last_frame": int, "last_ocr_frame": int}

    loop_last = time.perf_counter()
    fps = 0.0
    fps_alpha = 0.9
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now_s = time.perf_counter()
        h, w = frame.shape[:2]
        frame_i += 1

        if realtime:
            pace_video(t0, frame_i, frame_period)
            frame_i = behind_catchup(cap, t0, frame_i, frame_period)

        results = model.track(frame, tracker=tracker, persist=True, imgsz=imgsz, conf=conf, verbose=False)[0]
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            b = boxes.cpu()
            cls = b.cls.int().numpy()
            xyxy = b.xyxy.int().numpy()
            ids = extract_ids_numpy(b, len(cls))
            for i, (c, bb) in enumerate(zip(cls, xyxy)):
                if c != 0:
                    continue

                track_id = int(ids[i])
                x1, y1, x2, y2 = bb
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bw = x2 - x1
                bh = y2 - y1
                area = int(bw * bh)
                if bw < min_plate_w or bh < min_plate_h:
                    if track_id not in ocr_cache:
                        ocr_cache[track_id] = {"best_text": "", "best_width": 0, "best_crop": None, "last_frame": frame_i, "last_ocr_frame": frame_i}
                    else:
                        ocr_cache[track_id]["last_frame"] = frame_i
                    continue

                if show_gui or writer is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cached = ocr_cache.get(track_id)
                if cached is None:
                    cached = {"best_text": "", "best_width": 0, "best_crop": None, "last_frame": frame_i, "last_ocr_frame": frame_i}
                    ocr_cache[track_id] = cached
                else:
                    cached["last_frame"] = frame_i

                do_ocr = (frame_i - cached["last_ocr_frame"]) >= ocr_every_frames
                if do_ocr:
                    plate = frame[y1:y2, x1:x2]
                    if plate.size != 0:
                        plate = preprocess_plate_for_ocr(plate)
                        candidate = ocr_plate_text(ocr, plate, min_ocr_chars_len=min_ocr_chars_len)
                    else:
                        candidate = ""

                    if candidate:
                        old_w = cached["best_width"]
                        diff = (bw - old_w) / float(old_w) if old_w > 0 else 1.0
                        if old_w == 0 or (bw > old_w and diff > area_eps_ratio):
                            cached["best_width"] = bw
                            cached["best_text"] = candidate
                            cached["best_crop"] = frame[y1:y2, x1:x2].copy()

                    cached["last_ocr_frame"] = frame_i

                text = cached["best_text"]
                if not text:
                    continue
                if show_gui or writer is not None:
                    draw_text_box(frame, text, x1, y1, x2, y2)

        # after processing detections, insert plates whose track has been gone for >= 90 frames
        to_delete = []
        for tid, data in ocr_cache.items():
            best_text = data["best_text"]
            if not best_text:
                continue
            last_frame = data["last_frame"]
            if frame_i - last_frame >= 90:
                image_path = save_best_crop(run_dir, tid, best_text, data["best_crop"])
                insert_plate(db, track_id=tid, plate_text=best_text, best_width=data["best_width"], image_path=image_path, table=db_table)
                to_delete.append(tid)

        for tid in to_delete:
            ocr_cache.pop(tid, None)

        # fps counter
        loop_now = time.perf_counter()
        dt = loop_now - loop_last
        loop_last = loop_now
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        fps = inst_fps if fps == 0 else fps_alpha * fps + (1.0 - fps_alpha) * inst_fps

        if show_gui:
            draw_fps(frame, fps)
            show = fit_for_screen(frame, display_w or 1920, display_h or 1080)
            cv2.imshow("plate", show)
            if cv2.pollKey() & 0xFF == 27:
                break

        if writer is not None:
            writer.write(frame)

    # flush any remaining tracks that never hit the 90 frame gap
    for tid, data in ocr_cache.items():
        best_text = data["best_text"]
        if best_text:
            image_path = save_best_crop(run_dir, tid, best_text, data["best_crop"])
            insert_plate(db, track_id=tid, plate_text=best_text, best_width=data["best_width"], image_path=image_path, table=db_table)

    cap.release()
    if writer is not None:
        writer.release()
    if show_gui:
        cv2.destroyAllWindows()
    try:
        if hasattr(db, "close"):
            db.close()
    except Exception:
        pass
