# CNN-Based Image Classification Using TensorFlow

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras for binary image classification. The model processes RGB images of size 256 × 256 and predicts one of two classes using a sigmoid activation function.

The architecture combines convolutional layers, batch normalization, max pooling, global average pooling, and fully connected layers to learn image features and perform classification.

## Model Architecture

The CNN consists of the following components:

- **Input Layer:** Accepts RGB images of shape `(256, 256, 3)`.
- **Convolutional Block 1:** 32 filters with a 3 × 3 kernel, followed by batch normalization and max pooling.
- **Convolutional Block 2:** 64 filters with a 3 × 3 kernel, followed by batch normalization and max pooling.
- **Convolutional Block 3:** 128 filters with a 3 × 3 kernel, followed by batch normalization and max pooling.
- **Global Average Pooling:** Reduces spatial feature maps to a compact feature representation.
- **Dense Layer 1:** 128 neurons with ReLU activation and dropout of 0.4.
- **Dense Layer 2:** 64 neurons with ReLU activation and dropout of 0.3.
- **Output Layer:** One neuron with sigmoid activation for binary classification.

L2 regularization is applied to convolutional and dense layers to help reduce overfitting.

## Technologies Used

- Python
- TensorFlow
- Keras
- SciPy
- Matplotlib

## Key Features

- Three convolutional blocks for hierarchical image feature extraction.
- Batch normalization to support stable training.
- Max pooling for spatial downsampling.
- Global average pooling to reduce the number of parameters compared with flattening the feature maps.
- Dropout for regularization.
- L2 kernel regularization to help control model complexity.
- Sigmoid output for binary classification.
