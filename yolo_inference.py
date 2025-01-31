import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import requests
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# Configuration
CROWD_THRESHOLD = 30
ALERT_DELAY = 15
MOTION_THRESHOLD = 5000
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1334966466365358130/GQdP-llWP1QABcdDrxMuEmCQW9L5T0rf1dp0m_eRVrxTmh3I_NT6q3N39qvYF6urZXOT"
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

ROLLING_WINDOW = 30
ENABLE_CROWD_PREDICTION = True
PREDICT_FRAMES_AHEAD = 30

tracker = DeepSort(max_age=30, embedder="mobilenet")

class YOLOInference:
    def __init__(self, model_path='yolov8m.pt'):
        # Choose best available device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load YOLO
        self.model = YOLO(model_path).to(self.device)
        print(f"[INFO] YOLO model loaded on device: {self.device}")

        self.last_alert_time = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.density_map = None
        self.latest_frame = None

        # Rolling window data
        self.frame_indices = []
        self.crowd_counts = []

    def send_discord_alert(self, count, labeled_snapshot=None, raw_snapshot=None):
        """
        Sends an alert message and any available images to Discord via webhook.
        Respects the ALERT_DELAY to avoid spamming.
        """
        current_time = time.time()
        if (current_time - self.last_alert_time) < ALERT_DELAY:
            return
        self.last_alert_time = current_time

        # 1) Text payload
        message = {
            "content": (
                f"🚨 **Alert! High Crowd Density Detected!** 🚨\n"
                f"👥 **Crowd Count:** {count}\n"
                f"⚠️ **Immediate Action Required!**"
            )
        }
        
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(DISCORD_WEBHOOK_URL, json=message, headers=headers)
            print("[DEBUG] Discord Message Response:", resp.status_code, resp.text)
            if resp.status_code not in (200, 204):
                print(f"[ERROR] Discord (message) responded with: {resp.text}")
        except Exception as e:
            print(f"[ERROR] Exception sending Discord alert message: {e}")
            return

        # 2) If snapshots exist, send them
        files = {}
        if labeled_snapshot and os.path.exists(labeled_snapshot):
            files["file1"] = open(labeled_snapshot, "rb")
        if raw_snapshot and os.path.exists(raw_snapshot):
            files["file2"] = open(raw_snapshot, "rb")

        if files:
            try:
                resp_files = requests.post(DISCORD_WEBHOOK_URL, files=files)
                print("[DEBUG] Discord Files Response:", resp_files.status_code, resp_files.text)
                if resp_files.status_code not in (200, 204):
                    print(f"[ERROR] Discord (files) responded with: {resp_files.text}")
            except Exception as e:
                print(f"[ERROR] Exception sending Discord snapshots: {e}")
            finally:
                for f in files.values():
                    f.close()

    def _smooth_count(self, current_count, current_frame_idx):
        self.frame_indices.append(current_frame_idx)
        self.crowd_counts.append(current_count)

        if len(self.frame_indices) > ROLLING_WINDOW:
            self.frame_indices.pop(0)
            self.crowd_counts.pop(0)

        return int(round(np.mean(self.crowd_counts)))

    def _predict_future_crowd(self):
        if len(self.frame_indices) < 5:
            return None

        X = np.array(self.frame_indices).reshape(-1, 1)
        y = np.array(self.crowd_counts)

        model = LinearRegression()
        model.fit(X, y)

        last_frame = self.frame_indices[-1]
        future_frame = last_frame + PREDICT_FRAMES_AHEAD
        predicted_value = model.predict([[future_frame]])
        return int(round(predicted_value[0]))

    def process_video(self, input_path, output_path):
        print(f"[INFO] Opening video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25
            print(f"[WARN] Invalid FPS detected. Defaulting to: {fps} fps")

        self.density_map = np.zeros((height, width), dtype=np.float32)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] No more frames. Processing complete.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[DEBUG] Processed {frame_count} frames...")

            fgmask = self.fgbg.apply(frame)
            motion_pixels = np.count_nonzero(fgmask)
            significant_motion = (motion_pixels > MOTION_THRESHOLD)

            results = self.model.predict(frame, conf=0.3, device=self.device)
            yolo_boxes = results[0].boxes.data.cpu().numpy()

            persons = [box for box in yolo_boxes if int(box[5]) == 0]

            detections = []
            for box in persons:
                x1, y1, x2, y2, conf, class_id = box
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

            tracks = tracker.update_tracks(detections, frame=frame)
            raw_count = len(tracks)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(self.density_map, (cx, cy), 25, (1.0,), thickness=-1)

            smoothed_count = self._smooth_count(raw_count, frame_count)

            predicted_future_count = None
            if ENABLE_CROWD_PREDICTION:
                predicted_future_count = self._predict_future_crowd()

            # Draw bounding boxes
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(frame, f"People Count (Smoothed): {smoothed_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            if ENABLE_CROWD_PREDICTION and predicted_future_count is not None:
                cv2.putText(frame, f"Predicted Next: {predicted_future_count}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            density_norm = cv2.normalize(
                self.density_map, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

            alpha = 0.5
            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

            # Alert if threshold exceeded
            if smoothed_count > CROWD_THRESHOLD and significant_motion:
                print(f"🚨 ALERT: Crowd count {smoothed_count} exceeds threshold!")
                raw_frame = frame.copy()

                timestamp = int(time.time())
                labeled_snapshot_path = os.path.join(SNAPSHOT_FOLDER, f"alert_labeled_{timestamp}.jpg")
                raw_snapshot_path = os.path.join(SNAPSHOT_FOLDER, f"alert_raw_{timestamp}.jpg")

                cv2.imwrite(labeled_snapshot_path, overlay)
                cv2.imwrite(raw_snapshot_path, raw_frame)

                self.send_discord_alert(smoothed_count, labeled_snapshot_path, raw_snapshot_path)

            out.write(overlay)
            self.latest_frame = overlay

        cap.release()
        out.release()
        print(f"[INFO] Finished processing. Output saved to: {output_path}")


if __name__ == "__main__":
    yolo_infer = YOLOInference("yolov8m.pt")
    yolo_infer.process_video("input_video.mp4", "output_video.mp4")
