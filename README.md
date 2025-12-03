# CIFAR-10 Image Classification with CNN

This repository contains a **Convolutional Neural Network (CNN)** model built in **TensorFlow/Keras** for classifying images from the **CIFAR-10 dataset**. The model uses **Batch Normalization**, **Dropout**, and a **Cosine Decay learning rate schedule** to improve performance and prevent overfitting.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Features](#features)
* [Usage](#usage)
* [Results](#results)
* [Future Improvements](#future-improvements)

---

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal of this project is to train a deep CNN model to classify these images accurately.

The project implements:

* Convolutional layers with **ReLU** activation
* **Batch Normalization** after each convolution
* **Dropout** to prevent overfitting
* **Cosine Decay Learning Rate** scheduler
* Training and evaluation of the model
* Visualization of **predictions** and **confusion matrix**

---

## Dataset

CIFAR-10 classes:

```
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

* Training images: 50,000
* Test images: 10,000

The dataset is directly loaded using `tensorflow.keras.datasets.cifar10`.

---

## Model Architecture

```
Input: 32x32x3 image
Conv2D -> BatchNorm -> ReLU
Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout
Conv2D -> BatchNorm -> ReLU
Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout
Conv2D -> BatchNorm -> ReLU -> MaxPooling -> Dropout
Flatten -> Dense(128) -> BatchNorm -> Dropout
Dense(10) -> Softmax
```

* Optimizer: **Adam**
* Loss: **Sparse Categorical Crossentropy**
* Metrics: **Accuracy**

---

## Features

* **Batch Normalization:** Helps stabilize and speed up training.
* **Dropout:** Reduces overfitting.
* **Cosine Decay Learning Rate:** Adjusts learning rate during training for better convergence.
* **Visualization:** Display test images with predicted vs actual labels.
* **Confusion Matrix:** Shows model performance across all classes.

---

## Usage

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install tensorflow matplotlib scikit-learn seaborn
```

3. Run the training script:

```bash
python train_cifar10_cnn.py
```

4. Evaluate the model and visualize predictions/confusion matrix:

```python
# In Python or Jupyter Notebook
from evaluate import show_predictions, plot_confusion_matrix
```

---

## Results

* Achieved **Test Accuracy:** ~0.74 – 0.82 (depending on training)
* Confusion matrix identifies misclassified classes
* Model predictions can be visualized alongside true labels

---

## Future Improvements

* Implement **ResNet or VGG** architectures for better accuracy
* Use **data augmentation** to improve generalization
* Apply **learning rate schedulers** like `ReduceLROnPlateau` or `OneCycleLR`
* Hyperparameter tuning for **dropout rates**, **batch size**, and **learning rate**


