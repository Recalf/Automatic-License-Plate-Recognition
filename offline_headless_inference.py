from engine import run

# CONFIG
VID_IN = "test/vid3.mp4"
OUT_VIDEO = "out/offline_result.mp4"
RESULT_IMAGES_ROOT = "result_images"

# YOLO
CONF = 0.25
IMGSZ = 1280
MODEL_WEIGHTS = "runs/detect/train9/weights/last.pt"
TRACKER = "model/custom_bytetrackv2.yaml"

# DB
DB_HOST = "localhost"
DB_PORT = 
DB_USER = 
DB_PASSWORD = 
DB_NAME = "license_plate_db"
DB_TABLE = "plates"

# OCR
OCR_MODEL_NAME = "cct-s-v1-global-model"
MIN_OCR_CHARS_LEN = 5
OCR_EVERY_FRAMES = 15  
AREA_EPS_RATIO = 0.05  # n% bigger size margin to do ocr again
MIN_PLATE_W = 55  # how many pixels to not do ocr
MIN_PLATE_H = 20

# for offline speed, DB can get fast
DB_POOL_SIZE = 8


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

