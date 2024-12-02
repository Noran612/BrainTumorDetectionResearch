# Brain Tumor Detection Using MRI Scans with CAM-Guided Attention

## Overview

This repository contains the implementation of a deep learning model designed for brain tumor detection using Magnetic Resonance Imaging (MRI) scans. The model utilizes **Class Activation Maps (CAM)** to provide interpretable insights and guide channel attention, enabling accurate and explainable predictions.

### Features
- **Binary Classification**: Tumor vs. No Tumor.
- **Interpretable Predictions**: CAM visualizations highlight critical regions in MRI scans for model decisions.
- **Dataset Preprocessing**: Cropping, resizing, and normalization techniques are used to enhance model performance.
- **Attention Mechanism**: CAM-guided attention improves feature extraction and model generalization.

---

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Results and Visualizations](#results-and-visualizations)
- [How to Use](#how-to-use)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## Dataset
The model was trained and evaluated using a dataset of **2,065 MRI images** (after augmentation) sourced from:
- [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- [Abhranta Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri)

### Data Preprocessing
1. **Cropping**: Removes irrelevant background information.
2. **Resizing**: Standardizes all images to a fixed size of `240x240`.
3. **Normalization**: Scales pixel values to [0, 1] for numerical stability.
4. **Augmentation**: Increases dataset size to improve model generalization.

---

## Model Architecture

| **Layer**             | **Details**                                  | **Output Shape**     |
|------------------------|----------------------------------------------|----------------------|
| Input                 | MRI image (3 channels, 240x240)             | (3, 240, 240)       |
| Conv2D                | 32 filters, kernel size=5, stride=1, padding=2 | (32, 240, 240)      |
| BatchNorm2D           | Batch normalization on 32 channels          | (32, 240, 240)      |
| ReLU Activation       | Introduces non-linearity                    | (32, 240, 240)      |
| MaxPool2D             | Kernel size=4, stride=4                     | (32, 60, 60)        |
| CAM-Guided Attention  | Refines feature maps                        | (32, 60, 60)        |
| MaxPool2D             | Kernel size=4, stride=4                     | (32, 15, 15)        |
| Flatten               | Converts features into 1D vector            | (32 * 15 * 15)      |
| Fully Connected       | Sigmoid activation for binary classification | (1)                 |

---

## Training Pipeline

- **Optimizer**: Adam with a learning rate of `0.001`.
- **Loss Function**: Binary Cross Entropy Loss (BCELoss) for probability-based classification.
- **Progressive Attention**: CAM-guided attention is introduced after two epochs to stabilize training.
- **Metrics Tracked**:
  - Training Loss
  - Validation Loss
  - Training Accuracy
  - Validation Accuracy

---

## Results and Visualizations

### Performance Metrics
- **Validation Accuracy**: ~98.12%
- **Test Accuracy**: ~94.87%

---

## How to Use

### Clone Repository
```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain-tumor-detection
```

### Run the python notebook on google colab or jupyter

## Future Work
1. Extend to **multiclass classification** (e.g., distinguishing between benign and malignant tumors).
2. Integrate additional interpretability techniques like **Grad-CAM++**.
3. Optimize the model for **real-time inference** on lightweight devices.

---

## Acknowledgments
- **CAM Research**: Inspired by [Woo et al., 2018](https://arxiv.org/pdf/1807.06521v2).
- **Dataset Contributors**: Kaggle community for publicly available MRI datasets.


