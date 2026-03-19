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

def extract_ids_numpy(boxes_cpu, n): # i made this safer because i had some problems with the ids
    ids_t = boxes_cpu.id
    if ids_t is None:
        return np.arange(n, dtype=np.int32)
    try:
        return ids_t.cpu().numpy().astype(np.int32)
    except AttributeError:
        return ids_t.astype(np.int32)

def preprocess_plate_for_ocr(plate_bgr): # upscale + CLAHE + sharpness
    plate = cv2.resize(plate_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    #clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(6,6)) # commented because performed little bit worse than without it
    #gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0,0), 1.0)  # sharpness (blur then subtract blur from original) 
    gray = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)

    return gray

def save_best_crop(run_dir, track_id, plate_text, crop):
    if crop is None or crop.size == 0:
        return ""
    fname = f"{int(track_id)}_{_safe_name(plate_text)}.jpg"
    path = os.path.join(run_dir, fname)
    try:
        cv2.imwrite(path, crop)
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

def _safe_name(s): # left it as a function if we would want to change the safety logic later, we already write safely
    return s[:40]

def insert_plate(db, track_id, plate_text, best_width, image_path, table):
    ts = datetime.now()
    conn = None
    try:
        conn = db.get_connection() if hasattr(db, "get_connection") else db  # pooling else normal
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
            except Exception as e:
                print(f"Error closing DB connection: {e}")

def ocr_plate_text(ocr, plate_gray, min_ocr_chars_len, min_ocr_conf=0.0):
    preds = ocr.run(plate_gray, return_confidence=True)
    if not preds:
        return ""
    p0 = preds[0]
    raw_text = p0.plate

    # reject low confidence OCR
    if min_ocr_conf > 0.0:
        probs = p0.char_probs 
        if probs is not None:
            mean_conf = sum(probs) / float(len(probs))
            if mean_conf < min_ocr_conf:
                return ""
    candidate = "".join(ch for ch in raw_text.upper() if ch.isalnum())

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
    vid_in, tracker, model_weights, ocr_model_name, conf, imgsz, ocr_conf, min_ocr_chars_len,
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

    try:
        while True:
            if realtime:
                frame_i = behind_catchup(cap, t0, frame_i, frame_period)

            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            frame_i += 1 
        
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
                        candidate = ""   

                        if cached["best_width"] == 0: # first instance always take
                            plate = frame[y1:y2, x1:x2]
                            plate = preprocess_plate_for_ocr(plate)
                            candidate = ocr_plate_text(ocr, plate, min_ocr_chars_len=min_ocr_chars_len, min_ocr_conf=ocr_conf)
                            cached["last_ocr_frame"] = frame_i

                        else: # else event-based write
                            old_w = cached["best_width"]
                            diff = (bw - old_w) / float(old_w) if old_w > 0 else 1.0
                            if old_w == 0 or (bw > old_w and diff > area_eps_ratio):
                                plate = frame[y1:y2, x1:x2]
                                plate = preprocess_plate_for_ocr(plate)
                                candidate = ocr_plate_text(ocr, plate, min_ocr_chars_len=min_ocr_chars_len, min_ocr_conf=ocr_conf)
                                cached["last_ocr_frame"] = frame_i

                        if candidate:
                            cached["best_width"] = bw
                            cached["best_text"] = candidate
                            cached["best_crop"] = plate.copy()

                    text = cached["best_text"]
                    if not text:
                        continue
                    if show_gui or writer is not None:
                        draw_text_box(frame, text, x1, y1, x2, y2)

            # insert plates only when "all" tracks with same best_text have been gone for >= 90 frames
            if ocr_cache:
                any_candidate = False
                for data in ocr_cache.values():
                    if frame_i - data["last_frame"] >= 90:
                        any_candidate = True
                        break

                if any_candidate:
                    by_text = {}
                    for tid, data in ocr_cache.items():
                        txt = data["best_text"]
                        if not txt:
                            continue
                        if txt not in by_text:
                            by_text[txt] = []
                        by_text[txt].append(tid)

                    to_delete = []
                    for txt, tids in by_text.items():
                        all_done = True
                        for t in tids:
                            last_frame = ocr_cache[t]["last_frame"]
                            if frame_i - last_frame < 90:
                                all_done = False
                                break
                        if not all_done:
                            continue

                        best_tid = None
                        best_w = -1
                        best_last_frame = -1
                        for t in tids:
                            d = ocr_cache[t]
                            w = d["best_width"]
                            lf = d["last_frame"]
                            if w > best_w or (w == best_w and lf > best_last_frame):
                                best_w = w
                                best_last_frame = lf
                                best_tid = t

                        if best_tid is None:
                            for t in tids:
                                to_delete.append(t)
                            continue

                        d_best = ocr_cache[best_tid]
                        image_path = save_best_crop(run_dir, best_tid, txt, d_best["best_crop"])
                        insert_plate(
                            db,
                            track_id=best_tid,
                            plate_text=txt,
                            best_width=d_best["best_width"],
                            image_path=image_path,
                            table=db_table,
                        )

                        for t in tids:
                            to_delete.append(t)

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

            if realtime:
                pace_video(t0, frame_i, frame_period)

        # flush any remaining tracks that never hit the 90 frame gap:
        if ocr_cache:
            by_text = {}
            for tid, data in ocr_cache.items():
                txt = data["best_text"]
                if not txt: continue
                if txt not in by_text: by_text[txt] = []
                by_text[txt].append(tid)

            for txt, tids in by_text.items():
                best_tid = None
                best_w = -1
                for t in tids:
                    if ocr_cache[t]["best_width"] > best_w:
                        best_w = ocr_cache[t]["best_width"]
                        best_tid = t
                if best_tid:
                    d_best = ocr_cache[best_tid]
                    image_path = save_best_crop(run_dir, best_tid, txt, d_best["best_crop"])
                    insert_plate(db, best_tid, txt, d_best["best_width"], image_path, db_table)

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_gui:
            cv2.destroyAllWindows()
        try:
            db.close()
        except:
            pass
