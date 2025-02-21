import os
import cv2
import numpy as np

# -----------------------------------------------------------------------------
# ADJUST THESE PATHS AND PARAMETERS
# -----------------------------------------------------------------------------
JHU_DATASET_PATH = r"C:/Users/Vikas/Downloads/Compressed/jhu_crowd_v2.0" 
OUTPUT_YOLO_PATH = r"C:/Code/Crowd-Management-System/dataset"

# Size in pixels to create around each (x,y). 
# If BOX_HALF_SIZE=10, final box width=20 and height=20.
BOX_HALF_SIZE = 10

# Which subfolders to convert (assuming "train" and "val" exist).
SUBSETS = ["train", "val"]
# -----------------------------------------------------------------------------

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def point_to_yolo_bbox(image_shape, x_center, y_center, half_size):
    """
    Given an (x,y) *point* in absolute pixels plus a half-box size,
    produce a YOLOv8 bounding box line in the format:
        class x_center_norm y_center_norm width_norm height_norm
    """
    img_h, img_w = image_shape[:2]
    
    # Compute corner coordinates in absolute pixels
    x_min = clamp(x_center - half_size, 0, img_w - 1)
    x_max = clamp(x_center + half_size, 0, img_w - 1)
    y_min = clamp(y_center - half_size, 0, img_h - 1)
    y_max = clamp(y_center + half_size, 0, img_h - 1)

    # Convert corners -> YOLO [x_center, y_center, width, height], normalized
    bb_width = x_max - x_min
    bb_height = y_max - y_min
    x_c = x_min + bb_width / 2.0
    y_c = y_min + bb_height / 2.0

    x_c_norm = x_c / float(img_w)
    y_c_norm = y_c / float(img_h)
    w_norm = bb_width / float(img_w)
    h_norm = bb_height / float(img_h)

    # Class = 0 (person), then the 4 normalized values
    return f"0 {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

# Create YOLO folders
os.makedirs(f"{OUTPUT_YOLO_PATH}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_YOLO_PATH}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_YOLO_PATH}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_YOLO_PATH}/labels/val", exist_ok=True)

for subset in SUBSETS:
    images_dir = os.path.join(JHU_DATASET_PATH, subset, "images")
    gt_dir     = os.path.join(JHU_DATASET_PATH, subset, "gt")
    
    # Iterate over images in JHU
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith(".jpg"):
            continue
        
        # Derive matching GT file
        image_id = os.path.splitext(image_file)[0]
        gt_file_path = os.path.join(gt_dir, image_id + ".txt")
        image_path   = os.path.join(images_dir, image_file)
        
        # Read image to get its shape
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: could not read image {image_file}")
            continue
        
        img_h, img_w = image.shape[:2]

        # Read point annotations
        yolo_lines = []
        if os.path.exists(gt_file_path):
            with open(gt_file_path, "r") as f_gt:
                for line in f_gt:
                    # Typically JHU lines have x, y (one point per line).
                    # If your lines differ, adjust accordingly.
                    vals = line.strip().split()
                    if len(vals) < 2:
                        continue  # skip malformed
                    x_point = float(vals[-2])  # last but one
                    y_point = float(vals[-1])  # last
                    # Convert each point -> YOLO bounding box
                    label_str = point_to_yolo_bbox((img_h, img_w), x_point, y_point, BOX_HALF_SIZE)
                    yolo_lines.append(label_str)
        else:
            print(f"Warning: no GT file found for {image_file}")

        # Write YOLO label file
        yolo_label_path = os.path.join(OUTPUT_YOLO_PATH, "labels", subset, image_id + ".txt")
        with open(yolo_label_path, "w") as f_out:
            f_out.write("\n".join(yolo_lines))

        # Copy (or re‐write) image into YOLO folder
        yolo_image_path = os.path.join(OUTPUT_YOLO_PATH, "images", subset, image_file)
        cv2.imwrite(yolo_image_path, image)

print("✅ Done. Your dataset is now in YOLOv8 format with synthetic bounding boxes.")
