# Video Classification Using 3D CNN

## Overview

A deep learning model for classifying videos using a 3D Convolutional Neural Network (CNN) architecture. The model is designed to detect and classify activities in video sequences.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Configuration](#model-configuration)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- 3D CNN architecture with multiple parallel processing branches
- Support for video frame processing and batching
- Real-time data augmentation
- Integration with Weights & Biases (wandb) for experiment tracking
- Comprehensive evaluation metrics including accuracy, precision, recall, and F1 score
- ROC and Precision-Recall curve visualization

## Requirements

```txt
tensorflow
opencv-python==4.5.2.52
opencv-python-headless==4.5.2.52
wandb
numpy
pandas
matplotlib
scikit-learn
```

## Model Architecture

The model uses a complex 3D CNN architecture with:

- Multiple convolutional branches for feature extraction
- Batch normalization layers
- Various activation functions (LeakyReLU, PReLU, ReLU)
- Dropout layers for regularization
- Dense layers for final classification

### Key Components

1. Input processing branch
2. Dual parallel processing paths
3. Feature concatenation
4. Global pooling
5. Dense classification layers

## Dataset Structure

```
Dataset/
├── Train/
│   ├── Class1/
│   │   └── *.mp4
│   └── Class2/
│       └── *.mp4
└── Val/
    ├── Class1/
    │   └── *.mp4
    └── Class2/
        └── *.mp4
```

## Usage

### Training

```python
# Initialize wandb
wandb.init(project='your_project_name')

# Prepare datasets
train_ds = tf.data.Dataset.from_generator(...)
val_ds = tf.data.Dataset.from_generator(...)

# Train the model
model.fit(train_ds,
          epochs=config.epochs,
          validation_data=val_ds,
          callbacks=[wandb_callbacks])
```

### Inference

```python
# Load the model
model = tf.keras.models.load_model("path_to_model.keras")

# Prepare video
video_clip = prepare_video_by_skipping_frames("video_path.mp4")

# Get predictions
predictions = model.predict(np.array([video_clip]))
```

## Evaluation Metrics

The model's performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Precision-Recall Curves
- ROC Curves

## Model Configuration

Default configuration:

| Parameter | Value |
|-----------|--------|
| Learning rate | 0.001 |
| Number of epochs | 60 |
| Loss function | binary_crossentropy |
| Early stopping patience | 40 |
| LR reduction on plateau | True |
