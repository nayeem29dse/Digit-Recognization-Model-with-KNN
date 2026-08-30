# Digit Recognition Model with KNN

A handwritten digit recognition project built with the **K-Nearest Neighbors (KNN)** algorithm, trained and evaluated on the MNIST dataset, then tested against a custom hand-drawn digit image.

## Overview

This project walks through the full pipeline of a simple image classification task:

1. Load and explore the MNIST dataset of handwritten digits (0–9)
2. Preprocess the images (normalization, flattening)
3. Train a KNN classifier on the training set
4. Evaluate accuracy on the held-out test set
5. Save the trained model
6. Load the saved model and test it on a real, custom digit image (`sp1.png`)

## Repository Contents

| File | Description |
|---|---|
| `modeling.ipynb` | Loads MNIST, preprocesses the data, trains the KNN classifier, evaluates accuracy, and saves the model (`KNN_Model.pkl`) |
| `model_testing.ipynb` | Loads the saved model and runs a prediction on a custom test image |
| `sp1.png` | Sample handwritten digit image used for testing the trained model |

## How It Works

**Training (`modeling.ipynb`)**
- Loads the MNIST dataset via `tensorflow.keras.datasets`
- Normalizes pixel values to the `[0, 1]` range
- Flattens each 28×28 image into a 784-length vector
- Trains a `KNeighborsClassifier` (`k = 3`) from scikit-learn
- Evaluates the model on the test set, reaching **~97% accuracy**
- Serializes the trained model to `KNN_Model.pkl` using `pickle`

**Testing on a custom image (`model_testing.ipynb`)**
- Reads a real image (`sp1.png`) in grayscale using OpenCV
- Inverts the colors (`cv2.bitwise_not`) so the digit matches MNIST's white-on-black format
- Resizes the image to 28×28 to match the model's expected input shape
- Normalizes and flattens the image
- Loads the saved model and predicts the digit

## Requirements

- Python 3
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `scikit-learn`
- `opencv-python`
- `pickle` (standard library)

Install dependencies:

```bash
pip install numpy matplotlib seaborn tensorflow scikit-learn opencv-python
```

## Usage

1. Run `modeling.ipynb` end-to-end to train the KNN model on MNIST. This produces a `KNN_Model.pkl` file in the working directory.
2. Run `model_testing.ipynb` to load `KNN_Model.pkl` and test it on `sp1.png` (or swap in your own hand-drawn digit image, saved as a similar grayscale PNG).

## Results

The trained KNN classifier (k = 3) achieves approximately **97% accuracy** on the MNIST test set.

## Author

**Mehedee Hasan Nyeem**

**Data Science Student**
