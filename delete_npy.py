import os

# Replace with your actual dataset path:
DATASET_PATH = r"C:/Code/Crowd-Management-System/dataset"

for split in ["train", "val"]:
    folder_path = os.path.join(DATASET_PATH, "images", split)
    
    # Walk through subfolders (if any) as well
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".npy"):
                file_path = os.path.join(root, file)
                print(f"Deleting {file_path}")
                os.remove(file_path)

print("All .npy files removed.")
