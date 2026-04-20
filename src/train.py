import os
import kagglehub
from ultralytics import YOLO

if __name__ == "__main__":
    # Download and prepare dataset
    dataset_path = kagglehub.dataset_download("barkataliarbab/udacity-self-driving-car-obstacles-dataset")
    yaml_path = os.path.join(dataset_path, "data.yaml")
    config_path = os.path.join(os.path.dirname(__file__), "custom_yolo.yaml")

    # ref: https://github.com/ultralytics/ultralytics/blob/main/README.md & https://docs.ultralytics.com/usage/python/
    # Load model
    model = YOLO(config_path, task="detect") # detect task for object detection


    # Transfer pretrained YOLO neck + head weights
    model.load("yolov8n.pt")
   
    # save; training checkpoints and final model weights.
    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=512,
        optimizer="AdamW",
        device=0,  # auto-detect GPU (set to 'cpu' to force CPU)
        amp=False,   # disable automatic mixed precision - fixes NaN loss issues with custom backbone
        verbose=False # supresses terminal output
    )