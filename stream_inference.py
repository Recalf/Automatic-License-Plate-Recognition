from engine import run

# CONFIG
VID_IN = "vid5.mp4"
RESULT_IMAGES_ROOT = "result_images"

# inference view
DISPLAY_W = 1920
DISPLAY_H = 1080

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
MIN_PLATE_W = 55  # size threshold for ocr
MIN_PLATE_H = 20

def main():
    run(
        show_gui=True, # if this is False and save_video_path=None we wont draw anything (faster)
        realtime=True, # handles realtime processed stream/video with catchup and stream pacing 
                       # (because we drop frames its better to make it False if you're processing an offline video with show_gui=False and save_video_path=None for best consistency)
        save_video_path=None,
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
        db_pool=False,
        result_images_root=RESULT_IMAGES_ROOT,
        display_w=DISPLAY_W,
        display_h=DISPLAY_H,
    )


if __name__ == "__main__":
    main()

