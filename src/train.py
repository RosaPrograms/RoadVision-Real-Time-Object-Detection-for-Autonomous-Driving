import os
import shutil
import kagglehub
from ultralytics import YOLO

if __name__ == "__main__":
    # Download and prepare dataset
    cache_path = kagglehub.dataset_download("barkataliarbab/udacity-self-driving-car-obstacles-dataset")

    print("Copying images and labels to Data directory...")
    shutil.copytree(os.path.join(cache_path, "export", "images"), "./data/images", dirs_exist_ok=True)
    shutil.copytree(os.path.join(cache_path, "export", "labels"), "./data/labels", dirs_exist_ok=True)

    config_path = os.path.join(os.path.dirname(__file__), "custom_yolo.yaml")

    # ref: https://github.com/ultralytics/ultralytics/blob/main/README.md & https://docs.ultralytics.com/usage/python/
    # Load model
    model = YOLO(config_path, task="detect") # detect task for object detection

    # Transfer pretrained YOLO neck + head weights
    model.load("yolov8n.pt")
   
    # save; training checkpoints and final model weights.
    model.train(
        data="./data.yaml",
        epochs=5,
        imgsz=512,
        optimizer="AdamW",
        device=0,  # auto-detect GPU (set to 'cpu' to force CPU)
        amp=False,   # disable automatic mixed precision - fixes NaN loss issues with custom backbone
        verbose=False # supresses terminal output
    )