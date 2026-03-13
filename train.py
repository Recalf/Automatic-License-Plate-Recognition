from ultralytics import YOLO

def main():
    model = YOLO("yolo26s.pt")
    model.train(data="data/data.yaml", 
                epochs=50,
                patience=7,
                batch=14,
                imgsz=960,
                save_period=3, 
                # kept default mosaic (1.0, last 10 epochs 0.0)
                ) 

if __name__ == "__main__":
    main()      
