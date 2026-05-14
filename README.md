# Material Stream Identification System

This project implements a feature-based machine learning system for classifying waste materials using classical ML algorithms.

## Classes

The system classifies images into seven classes:

1. Glass
2. Paper
3. Cardboard
4. Plastic
5. Metal
6. Trash
7. Unknown

## Machine Learning Pipeline

The project follows this pipeline:

1. Data loading
2. Train/test split
3. Data augmentation on training images only
4. Unknown class generation
5. Feature extraction
6. Feature scaling
7. SVM training
8. k-NN training
9. Model evaluation
10. Real-time camera deployment

## Feature Extraction Methods

The following handcrafted image descriptors are used:

- HOG for shape and gradient information
- Color histograms in BGR, HSV, and LAB spaces
- HSV color moments
- Local Binary Pattern for texture
- GLCM texture features
- Hu Moments for shape
- Canny edge density and contour statistics
- Gabor texture filters

## Models

Two classical ML models are trained and compared:

- Support Vector Machine with RBF kernel
- k-Nearest Neighbors

Both models use GridSearchCV for hyperparameter tuning.

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
