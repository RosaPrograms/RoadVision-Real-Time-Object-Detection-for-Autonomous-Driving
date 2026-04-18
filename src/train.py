import os
import kagglehub
from torchvision import transforms
from dataset import RoadDataset


# Download latest version
path = kagglehub.dataset_download("barkataliarbab/udacity-self-driving-car-obstacles-dataset")

print("Path to dataset files:", path)

image_dir = os.path.join(path, "export", "images")
label_dir = os.path.join(path, "export", "labels")

# 3. Load dataset
dataset = RoadDataset(image_dir, label_dir)

# 4. Test one sample
img, target = dataset[0]

print(img)
print(target)