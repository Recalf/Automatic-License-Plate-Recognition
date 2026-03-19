from engine import run

# CONFIG
VID_IN = "vid5.mp4"
OUT_VIDEO = "out/offline_result.mp4"
RESULT_IMAGES_ROOT = "result_images"

# YOLO
CONF = 0.25
IMGSZ = 1920
MODEL_WEIGHTS = "runs/detect/train9/weights/last.pt"
TRACKER = "model/custom_bytetrackv2.yaml"

# DB
DB_HOST = 
DB_PORT = 
DB_USER = 
DB_PASSWORD = 
DB_NAME = 
DB_TABLE = 

DB_POOL_SIZE = 8 # for offline speed, DB can get fast

# OCR
OCR_MODEL_NAME = "cct-s-v2-global-model" # xs version: cct-s-v2-global-model
OCR_CONF = 0.75 # threshold
MIN_OCR_CHARS_LEN = 5
OCR_EVERY_FRAMES = 5 # we have 2 checks to do ocr, this first, then if size (with margin eps) bigger
AREA_EPS_RATIO = 0.03  # n% bigger size margin to do ocr again
MIN_PLATE_W = 25  # size threshold for ocr
MIN_PLATE_H = 25


def main():
    run(
        show_gui=False,
        realtime=False,  # full speed (no wait)
        save_video_path=OUT_VIDEO,
        vid_in=VID_IN,
        tracker=TRACKER,
        model_weights=MODEL_WEIGHTS,
        ocr_model_name=OCR_MODEL_NAME,
        conf=CONF,
        imgsz=IMGSZ,
        ocr_conf=OCR_CONF,
        min_ocr_chars_len=MIN_OCR_CHARS_LEN,
        ocr_every_frames=OCR_EVERY_FRAMES,
        area_eps_ratio=AREA_EPS_RATIO,
        min_plate_w=MIN_PLATE_W,
        min_plate_h=MIN_PLATE_H,
        db_host=DB_HOST,
        db_port=DB_PORT,
        db_user=DB_USER,
        db_password=DB_PASSWORD,
        db_name=DB_NAME,
        db_table=DB_TABLE,
        db_pool=True,
        db_pool_size=DB_POOL_SIZE,
        result_images_root=RESULT_IMAGES_ROOT,
    )


if __name__ == "__main__":
    main()

