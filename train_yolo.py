from ultralytics import YOLO

def train():
    model = YOLO("yolov8m.pt")  # Load YOLO model
    model.train(data="/content/Crowd-Management-System/dataset/data.yaml", epochs=120, imgsz=640, batch=8, device="cuda", lr0=0.0005, resume=True, half=True)

if __name__ == "__main__":
    train()
