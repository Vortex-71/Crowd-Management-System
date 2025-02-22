from ultralytics import YOLO

def train():
    # 1) Load your previously trained weights instead of yolov8m.pt
    model = YOLO("C:/Code/Crowd-Management-System/runs/detect/train2/weights/best.pt")

    # 2) Train further using those weights as a starting point (not from scratch)
    model.train(
        data="C:/Code/Crowd-Management-System/dataset/data.yaml",
        epochs=120,
        imgsz=640,
        batch=8,
        device="cuda",
        lr0=0.0005,
        half=True
    )

if __name__ == "__main__":
    train()
