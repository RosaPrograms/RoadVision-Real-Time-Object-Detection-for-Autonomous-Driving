import os
import kagglehub
from ultralytics import YOLO

if __name__ == "__main__":
    # Download and prepare dataset
    dataset_path = kagglehub.dataset_download(
        "barkataliarbab/udacity-self-driving-car-obstacles-dataset")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "custom_yolo.yaml")

    # ref: https://github.com/ultralytics/ultralytics/blob/main/README.md & https://docs.ultralytics.com/usage/python/
    # Load model
    # detect task for object detection
    model = YOLO(config_path, task="detect")

    # Transfer pretrained YOLO neck + head weights
    model.load("yolov8n.pt")

    # device is auto detected (GPU if available, else CPU): add device=0 for GPU, device='cpu' for CPU
    # save; training checkpoints and final model weights.
    model.train(
        data=yaml_path,
        epochs=10,
        imgsz=512,
        optimizer="AdamW",
        # freeze backbone layers (first 10 layers) to retain pretrained features
        freeze=10,
        lr0=0.001,  # initial learning rate
        seed=42,
        save=True,
    )
