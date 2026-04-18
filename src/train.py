import os
import yaml
import kagglehub
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Download and prepare dataset
dataset_path = kagglehub.dataset_download("barkataliarbab/udacity-self-driving-car-obstacles-dataset")

# Fix the data.yaml paths to point to the correct locations
yaml_path = os.path.join(dataset_path, "data.yaml")
with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# Update paths to use the export subdirectory
data_config['path'] = dataset_path
data_config['train'] = 'export/images'  # All images are in export/images for this dataset
data_config['val'] = 'export/images'    # Use same for validation 
data_config['test'] = 'export/images'   # Use same for testing

# Write updated config
with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f)

model.train(
    data=yaml_path,
    epochs=1,
    imgsz=512,
    optimizer="AdamW",
    device=0  # Use GPU (device 0 is the first GPU, 'cuda' for auto-select)
)