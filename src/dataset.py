import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class RoadDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Label
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    c, x, y, w, h = map(float, line.split())
                    boxes.append([x, y, w, h])
                    labels.append(int(c))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        return image, target