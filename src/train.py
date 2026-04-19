import os
import kagglehub
from ultralytics import YOLO

if __name__ == "__main__":
    # Download and prepare dataset
    dataset_path = kagglehub.dataset_download("barkataliarbab/udacity-self-driving-car-obstacles-dataset")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "custom_yolo.yaml")

    # Load model
    model = YOLO(config_path, task="detect")


    # Transfer pretrained YOLO neck + head weights
    model.load("yolov8n.pt")
        
    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=512,
        optimizer="AdamW",
        device=0  # Use GPU (device 0 is the first GPU, 'cuda' for auto-select)
    )