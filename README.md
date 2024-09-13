
# Deepfake Detection Using 3-Ensembled CNN Model

## Overview

This repository contains the code and sample dataset for a deepfake detection project using a 3-ensembled customised sequential Convolutional Neural Network (CNN) model. The project aims to identify deepfake images by analyzing their features and detecting inconsistencies introduced by deepfake generation techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contact](#contact)

## Project Structure

```
deepfake-detection/
├── dataset/
│   ├── real/
│   └── fake/
├── deepfakedetection.py
├── README.md
```

- `dataset/`: Contains sample dataset with real and fake images.
- `deepfakedetection.py`: The main script containing the complete code for preprocessing, training, evaluation, and detection.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies.

## Installation

To run this project, you need to have Python installed. Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

## Usage

1. **Run the detection script**: The script will preprocess the dataset, train the models, evaluate them, and perform detection.

```bash
python deepfakedetection.py
```

## Dataset

The sample dataset contains real and fake images. You can find the dataset in the `dataset/` directory. For a larger dataset, you may consider using publicly available deepfake datasets such as FaceForensics++, Celeb-DF, or DFDC.

## Model Architecture

The deepfake detection model consists of three CNN models that are ensembled to improve accuracy. Each model is trained independently and their outputs are combined to make the final prediction.

### Model 1

- **Input Layer**: 256x256x3 images
- **Convolution Layer**: 32 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Convolution Layer**: 64 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Convolution Layer**: 128 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Flatten Layer**
- **Dense Layer**: 128 units, ReLU activation, L2 regularization
- **Dropout Layer**: 20%
- **Dense Layer**: 64 units, ReLU activation, L2 regularization
- **Dropout Layer**: 20%
- **Output Layer**: 1 unit, sigmoid activation

### Model 2

- **Input Layer**: 256x256x3 images
- **Convolution Layer**: 32 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Convolution Layer**: 64 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 4x4, stride 2
- **Flatten Layer**
- **Dense Layer**: 64 units, ReLU activation, L2 regularization
- **Dropout Layer**: 20%
- **Dense Layer**: 32 units, ReLU activation, L2 regularization
- **Dropout Layer**: 20%
- **Output Layer**: 1 unit, sigmoid activation

### Model 3

- **Input Layer**: 256x256x3 images
- **Convolution Layer**: 32 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Convolution Layer**: 64 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Convolution Layer**: 128 filters, kernel size 3x3, ReLU activation
- **Batch Normalization**
- **Max Pooling Layer**: pool size 2x2, stride 2
- **Flatten Layer**
- **Dense Layer**: 128 units, ReLU activation, L2 regularization
- **Dropout Layer**: 20%
- **Dense Layer**: 32 units, ReLU activation, L2 regularization
- **Dropout Layer**: 20%
- **Output Layer**: 1 unit, sigmoid activation

## Results

The ensembled model achieves a high accuracy in detecting deepfakes. Below are the performance metrics:

- **Accuracy**: 70%

## Contact

For any questions or suggestions, please open an issue or contact [Hanush M](mailto:mhanush@gmail.com) [Manoj S](mailto:manoj12a2004@gmail.com) [Prakash](mailto:pprakash45984@gmail.com) [Vikkram V](mailto:mpoco3027@gmail.com).

---
