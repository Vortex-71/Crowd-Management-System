import cv2
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import requests
import os
import numpy as np

# Alert Threshold
CROWD_THRESHOLD = 30  
ALERT_DELAY = 15  # Minimum time (seconds) between consecutive alerts

# Motion Detection Threshold
MOTION_THRESHOLD = 5000  # Total number of foreground (moving) pixels required to consider motion significant

# Discord Webhook
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_SERVER_WEBHOOK_URL"

# Output directory for alert snapshots
SNAPSHOT_FOLDER = "snapshots"
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, embedder="mobilenet")

class YOLOInference:
    def __init__(self, model_path='yolov8m.pt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.last_alert_time = 0  # Timestamp of last alert

        # Background subtractor for motion detection
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Will hold the accumulated heatmap
        self.density_map = None
        
        # Store latest processed frame for live preview
        self.latest_frame = None
        
        print(f"[INFO] YOLO model loaded on device: {self.device}")

    def send_discord_alert(self, count, labeled_snapshot=None, raw_snapshot=None):
        """
        Send an alert to Discord with labeled & raw image attachments.
        Includes try/except to catch errors and logs the response.
        """
        current_time = time.time()
        
        # Ensure alert delay is respected
        if current_time - self.last_alert_time < ALERT_DELAY:
            print("[DEBUG] Alert suppressed due to ALERT_DELAY.")
            return  

        self.last_alert_time = current_time  # Update last alert time
        
        message = {
            "content": (
                f"🚨 **Alert! High Crowd Density Detected!** 🚨\n"
                f"👥 **Crowd Count:** {count}\n"
                f"⚠️ **Immediate Action Required!**"
            )
        }
        
        print("[DEBUG] Sending alert to Discord...")
        try:
            response_msg = requests.post(DISCORD_WEBHOOK_URL, json=message)
            print(f"[DEBUG] Discord text message status: {response_msg.status_code}")
            if response_msg.status_code not in (200, 204):
                print(f"[ERROR] Discord response text: {response_msg.text}")
        except Exception as e:
            print(f"[ERROR] Exception when sending Discord alert message: {e}")

        # Prepare snapshot files (if they exist)
        files = {}
        if labeled_snapshot and os.path.exists(labeled_snapshot):
            files["file1"] = open(labeled_snapshot, "rb")
        if raw_snapshot and os.path.exists(raw_snapshot):
            files["file2"] = open(raw_snapshot, "rb")

        if files:
            print("[DEBUG] Sending snapshots to Discord...")
            try:
                response_files = requests.post(DISCORD_WEBHOOK_URL, files=files)
                print(f"[DEBUG] Discord file upload status: {response_files.status_code}")
                if response_files.status_code not in (200, 204):
                    print(f"[ERROR] Discord response text: {response_files.text}")
            except Exception as e:
                print(f"[ERROR] Exception when sending Discord snapshots: {e}")

    def process_video(self, input_path, output_path):
        """
        Processes a video file, detects people, adds heatmap & motion detection,
        and writes output to an MP4. Also updates self.latest_frame for live preview.
        """
        print(f"[INFO] Opening video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            print("[WARN] FPS not detected, defaulting to 25.")
            fps = 25

        # Initialize density map to zeros
        self.density_map = np.zeros((height, width), dtype=np.float32)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] No more frames to read, ending processing.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[DEBUG] Processed {frame_count} frames so far...")

            raw_frame = frame.copy()  # store a raw copy before drawing labels

            # ---------------------------
            # 1) Motion Detection Check
            # ---------------------------
            fgmask = self.fgbg.apply(frame)
            motion_pixels = np.count_nonzero(fgmask)
            significant_motion = (motion_pixels > MOTION_THRESHOLD)

            # Debug prints
            # Uncomment if you want to see motion pixel counts each frame:
            # print(f"[DEBUG] Frame: {frame_count}, motion_pixels: {motion_pixels}")

            # -----------------------------------
            # 2) Person Detection with YOLOv8
            # -----------------------------------
            results = self.model.predict(frame, conf=0.3, device=self.device)
            persons = results[0].boxes.data.cpu().numpy()

            # Filter only 'person' class (YOLOv8 index for 'person' is 0)
            persons = [box for box in persons if int(box[5]) == 0]

            detections = []
            for box in persons:
                x1, y1, x2, y2, conf, cls_id = box
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

            # -----------------------------------
            # 3) Tracking with DeepSORT
            # -----------------------------------
            tracks = tracker.update_tracks(detections, frame=frame)
            count = len(tracks)

            # -----------------------------------
            # 4) Update Heatmap
            # -----------------------------------
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Draw a small circle on the density map
                cv2.circle(self.density_map, (cx, cy), 25, (1.0,), thickness=-1)

            # -----------------------------------
            # 5) Draw bounding boxes & info on frame
            # -----------------------------------
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(frame, f"People Count: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # -----------------------------------
            # 6) Generate Heatmap Overlay
            # -----------------------------------
            density_norm = cv2.normalize(self.density_map, None, 0, 255, cv2.NORM_MINMAX)
            density_norm = density_norm.astype(np.uint8)
            heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

            # Blend heatmap onto the frame
            alpha = 0.5
            overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

            # -----------------------------------
            # 7) Trigger Alert (if needed)
            # -----------------------------------
            if count > CROWD_THRESHOLD and significant_motion:
                print(f"🚨 ALERT: Crowd count {count} exceeds threshold!")
                
                # Save labeled snapshot
                labeled_snapshot_filename = f"alert_labeled_{int(time.time())}.jpg"
                labeled_snapshot_path = os.path.join(SNAPSHOT_FOLDER, labeled_snapshot_filename)
                cv2.imwrite(labeled_snapshot_path, overlay)  # overlay or frame

                # Save raw snapshot (no labels)
                raw_snapshot_filename = f"alert_raw_{int(time.time())}.jpg"
                raw_snapshot_path = os.path.join(SNAPSHOT_FOLDER, raw_snapshot_filename)
                cv2.imwrite(raw_snapshot_path, raw_frame)

                # Send Discord alert
                self.send_discord_alert(count, labeled_snapshot_path, raw_snapshot_path)

            # -----------------------------------
            # 8) Write output frame and store latest_frame
            # -----------------------------------
            out.write(overlay)
            self.latest_frame = overlay  # For live preview usage

        cap.release()
        out.release()
        print(f"[INFO] Finished processing. Output saved to: {output_path}")
