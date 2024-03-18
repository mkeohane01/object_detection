# Comparative Analysis of YOLO and Faster R-CNN for Object Detection in Household Environments

## Overview
This project presents a  comparative analysis between two leading object detection models, YOLOv5 and Faster R-CNN (implemented via Detectron2), within the context of detecting household items. The primary objective was to evaluate and compare the performance of these models on a custom dataset comprising images of laptops, drinks, and utensils. The evaluation criteria included mean Average Precision (mAP), inference speed, and model size.

## Dataset
The dataset consists of approximately 350 images, evenly distributed among the three classes: utensils, drinks, and laptops. These images were collected and annotated by hand and data is stored in data/

## Methodology
The project involved several key steps, starting from data collection and annotation, followed by model training using transfer learning, and finally, evaluation based on the predefined criteria. Both models were pretrained on COCO dataset and fine-tuned on our custom dataset using transfer learning to adapt to the specific task of household item detection.

### Model Architectures
- **YOLOv5:** - https://github.com/ultralytics/yolov5
- **Detectron2 (Faster R-CNN):** - https://github.com/facebookresearch/detectron2

### Training Environment
All models were trained on Google Colab utilizing a T4 GPU. Training notebooks are found in notebooks/

## Results
The comparative analysis revealed distinct strengths and weaknesses for each model. Detectron2 showcased superior object detection precision but at the cost of increased model size and slower inference speed compared to YOLOv5. YOLOv5, although slightly less accurate, demonstrated remarkable efficiency, making it suitable for real-time object detection applications.

| Model       | mAP50 | mAP50-95 | Inf. Time (s/iter) | Size (MB) |
|-------------|-------|----------|--------------------|-----------|
| Detectron2  | 0.934 | 0.589    | 0.0860             | 213       |
| YOLOv5s     | 0.785 | 0.483    | 0.0151             | 13.7      |
| YOLOv5l     | 0.765 | 0.469    | 0.0356             | 88.5      |

## Conclusion
Both YOLOv5 and Detectron2 are powerful tools for object detection, each with its own advantages. YOLO is much faster and cost efficient while Faster-RCNN has better performance.

---

*Note: This project was conducted by Mike Keohane for the AIPI590-CV course, focusing on the practical application of computer vision techniques in everyday scenarios.*