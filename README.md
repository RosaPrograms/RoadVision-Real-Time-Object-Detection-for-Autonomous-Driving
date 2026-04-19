# RoadVision

Real-Time Object Detection for Autonomous Driving

Missouri S&T — Deep Learning Final Project  
John Wheeler · Heriberto Rosa · Joshua Kroft · Dylan Seabaugh

---

## What This Project Is

RoadVision is a YOLO-based object detection system trained to identify road-critical objects in real-world driving scenarios. The model is designed to detect 11 classes relevant to autonomous driving: bikers, cars, pedestrians, trucks, and several traffic light variants.

The project uses the Udacity Self-Driving Car Obstacles Dataset, which contains approximately 30,000 labeled dashcam images pre-split into train, validation, and test sets by Roboflow.

---

## How We Implemented It

The core idea was to build a custom CNN backbone from scratch, then attach it to a pretrained YOLOv8 detection neck and head for fine-tuning. This gave us hands-on implementation of the feature extraction architecture while leveraging pretrained weights for the YOLO detection logic.

### Custom Backbone

We implemented a Darknet-inspired CNN in `model.py`. The backbone is built from three components stacked together into a progressive feature hierarchy:

- `ConvBNLeaky` — the fundamental unit, combining a Conv2d layer, BatchNorm, and LeakyReLU activation
- `DarknetResidualBlock` — a bottleneck residual block (1x1 conv → 3x3 conv → skip connection) that reduces parameters while preserving spatial features
- `DarknetStage` — a downsampling stage combining a strided convolution with N residual blocks

The full backbone takes a 512x512 RGB image and outputs three feature maps at different spatial scales: P3 (64x64), P4 (32x32), and P5 (16x16). These three scales feed into the YOLO neck for multi-scale detection — small objects are detected from P3, medium from P4, and large from P5.

### YOLO Neck and Head

The detection neck and head are defined in `custom_yolo.yaml` using the Ultralytics config format. The neck is a Feature Pyramid Network (FPN) that fuses the three backbone feature maps through a top-down path (upsampling and concatenation) and a bottom-up path (downsampling and concatenation), allowing semantic information from deep layers to mix with spatial detail from shallow layers. The detection head runs across all three scales simultaneously, predicting bounding boxes, objectness scores, and class probabilities for each grid cell.

### Training

The model is trained in `train.py` using transfer learning. The architecture is built from `custom_yolo.yaml` and pretrained YOLOv8n weights are loaded into the neck and head via `model.load("yolov8n.pt")`. Training uses the AdamW optimizer.

## Setup

### Requirements

- Python 3.12
- NVIDIA GPU strongly recommended
- Kaggle account for dataset download

### Install

Clone the repository and create a virtual environment:

```bash
python -m venv .roadvision
.roadvision\Scripts\activate        # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Kaggle Dataset

The dataset downloads automatically via `kagglehub` on first run.

### Run

```bash
python src/train.py
```
