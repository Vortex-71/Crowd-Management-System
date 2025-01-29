from ultralytics import YOLO

def train():
    model = YOLO("yolov8m.pt")  # Load YOLO model
    model.train(data="/content/Ultron_Hackathon/dataset/data.yaml", epochs=120, imgsz=512, batch=4, device="cuda", half=True)

if __name__ == "__main__":
    train()
