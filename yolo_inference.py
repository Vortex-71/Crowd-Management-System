import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import requests
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CROWD_THRESHOLD = 30            # For Discord alerts (overall count threshold)
ALERT_DELAY = 15                # Minimum seconds between alerts
MOTION_THRESHOLD = 5000         # For background subtraction
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1334259674580516979/pH92tTp_wnYG2a5j6KNgPHHHmbQRC7Hs8L01KANDKiQrw7iE4jPa6iuWqauLY1G6DqoD"
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

ROLLING_WINDOW = 30             # For smoothing crowd count
ENABLE_CROWD_PREDICTION = True  
PREDICT_FRAMES_AHEAD = 30       # Not used in grid analysis

# --- Grid-based crowd analysis settings ---
ENABLE_GRID_ANALYSIS = True
NUM_GRID_ROWS = 6               # Grid rows
NUM_GRID_COLS = 6               # Grid cols
GRID_CELL_THRESHOLD = 3         # If cell count >= this, mark it

# DeepSort tracker
tracker = DeepSort(max_age=30, embedder="mobilenet")


class YOLOInference:
    def __init__(self, model_path='yolov8m.pt'):
        # Load YOLO on best available device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        print(f"[INFO] YOLO model loaded on device: {self.device}")

        self.last_alert_time = 0
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.density_map = None
        self.latest_frame = None

        self.frame_indices = []
        self.crowd_counts = []

        # For toggling heatmap overlay
        self.enable_heat_map = False

        # For zooming into a grid cell (set via /set_zoom or /enhanced_view)
        self.zoom_row = None
        self.zoom_col = None

        # We'll store the last full overlay frame (with boxes, grid, etc.)
        self.last_processed_overlay = None

        # --- Initialize the DNN Super Resolution object ---
        # self.sr = dnn_superres.DnnSuperResImpl_create()
        # sr_model_path = "models/EDSR_x2.pb"  # adjust if needed
        # self.sr.readModel(sr_model_path)
        # self.sr.setModel("edsr", 2)  # Use EDSR with 2× upscaling
        # print("[INFO] EDSR x2 super-resolution model loaded!")

    def set_heatmap_enabled(self, state: bool):
        self.enable_heat_map = state
        print(f"[INFO] Heat map enabled: {self.enable_heat_map}")

    def set_zoom_cell(self, row: int, col: int):
        """Set the grid cell (row, col) to magnify; if negative, reset."""
        if row < 0 or col < 0:
            self.zoom_row = None
            self.zoom_col = None
            print("[INFO] Zoom reset (no cell selected).")
        else:
            self.zoom_row = row
            self.zoom_col = col
            print(f"[INFO] Zoom cell set to row={row}, col={col}")

    def _crop_and_zoom_cell(self, overlay_frame):
        if self.zoom_row is None or self.zoom_col is None:
            return None

        h, w, _ = overlay_frame.shape
        cell_w = w // NUM_GRID_COLS
        cell_h = h // NUM_GRID_ROWS

        x1 = self.zoom_col * cell_w
        y1 = self.zoom_row * cell_h
        x2 = x1 + cell_w
        y2 = y1 + cell_h

        if x1 >= x2 or y1 >= y2 or x2 > w or y2 > h:
            return None

        subimg = overlay_frame[y1:y2, x1:x2].copy()

        # Optionally do super-resolution:
        # sr_enhanced = self.sr.upsample(subimg)
        # return sr_enhanced

        # or naive upscaling:
        zoom_factor = 2
        subimg_zoomed = cv2.resize(subimg, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        return subimg_zoomed

    def send_discord_alert(self, count, labeled_snapshot=None, raw_snapshot=None):
        current_time = time.time()
        if (current_time - self.last_alert_time) < ALERT_DELAY:
            return
        self.last_alert_time = current_time

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

    def _grid_crowd_analysis(self, frame, tracks, num_rows, num_cols, cell_threshold):
        height, width, _ = frame.shape
        cell_w = width // num_cols
        cell_h = height // num_rows
        grid_counts = np.zeros((num_rows, num_cols), dtype=int)

        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            col = min(cx // cell_w, num_cols - 1)
            row = min(cy // cell_h, num_rows - 1)
            grid_counts[row, col] += 1

        for r in range(1, num_rows):
            y = r * cell_h
            cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
        for c in range(1, num_cols):
            x = c * cell_w
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)

        for r in range(num_rows):
            for c in range(num_cols):
                if grid_counts[r, c] >= cell_threshold:
                    top_left = (c * cell_w, r * cell_h)
                    bottom_right = ((c + 1) * cell_w, (r + 1) * cell_h)
                    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
                    cv2.putText(frame, f"{grid_counts[r, c]}", (top_left[0] + 5, top_left[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, grid_counts

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

            if self.enable_heat_map:
                density_norm = cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX)
                density_norm = density_norm.astype(np.uint8)
                heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
                alpha = 0.5
                overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
            else:
                overlay = frame.copy()

            if ENABLE_GRID_ANALYSIS:
                overlay, grid_counts = self._grid_crowd_analysis(
                    overlay, tracks, NUM_GRID_ROWS, NUM_GRID_COLS, GRID_CELL_THRESHOLD
                )

            self.last_processed_overlay = overlay.copy()
            out.write(overlay)
            self.latest_frame = overlay

        cap.release()
        out.release()
        print(f"[INFO] Finished processing. Output saved to: {output_path}")

    def get_zoomed_subimage(self):
        if self.last_processed_overlay is None:
            return None
        return self._crop_and_zoom_cell(self.last_processed_overlay)


if __name__ == "__main__":
    yolo_infer = YOLOInference("yolov8m.pt")
    yolo_infer.process_video("input_video.mp4", "output_video.mp4")
