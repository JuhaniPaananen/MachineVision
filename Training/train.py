from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

script_dir = os.path.dirname(os.path.abspath(__file__))

model.train(
    data="C:/path_to/ultralytics/datasetfolder/data.yaml", # Example
    epochs=50,
    imgsz=640,
    project=script_dir,  # Set project directory to the script location
    name="train_results"  # Folder name for this training run
)