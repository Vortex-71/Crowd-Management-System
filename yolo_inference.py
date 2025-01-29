import torch
from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_path='yolov8m.pt'):
        """
        Load YOLO model and set it to run on GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path).to(self.device)

    def process_video(self, input_path, output_path):
        import cv2
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO Inference on GPU
            results = self.model.predict(frame, conf=0.3, device=self.device)

            for r in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls_id = map(int, r[:6])
                if cls_id == 0:  # Person class
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        return output_path
