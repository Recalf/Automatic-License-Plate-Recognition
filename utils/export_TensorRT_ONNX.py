from ultralytics import YOLO

m = YOLO("runs/detect/train9/weights/last.pt")
m.export(format="engine", imgsz=1280, half=True, device=0) # TensorRT
# m.export(format="onnx", imgsz=1280)  # ONNX