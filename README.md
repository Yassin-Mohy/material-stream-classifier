# Material Stream Identification System

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-8C1D40?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

A feature-based machine learning system for classifying waste materials into seven categories using classical computer vision and traditional ML models. The project follows a full pipeline from raw image ingestion to feature extraction, model training, evaluation, and real-time webcam inference.

---

## Motivation & Project Overview

Waste sorting is a practical computer vision problem where the input is often small, noisy, and visually similar across classes. Rather than relying on end-to-end deep learning, this project uses handcrafted image descriptors and classical ML to build a compact, explainable classifier for material recognition.

The system classifies items into:

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash
7. Unknown

The `unknown` class is included to represent synthetic or hard-to-classify samples so the deployment stage can safely reject low-confidence predictions.

---

## Problem Description

The project is designed around three core requirements:

- Build a multi-class material classifier using classical ML.
- Preserve a clean train/test workflow with augmentation only applied to the training split.
- Provide a real-time camera application that predicts material class from a live frame.

The system is intentionally built as a reproducible pipeline, not a one-off notebook experiment. Each stage produces persisted artifacts that can be reused for evaluation and deployment.

---

## Architecture Summary

```text
Raw Images
	│
	▼
Train/Test Split
	│
	├──────────────► Data Augmentation (train only)
	│
	├──────────────► Unknown Class Generation
	│
	▼
Feature Extraction
	│
	▼
Feature Scaling + Label Encoding
	│
	├──────────────► SVM Training
	│
	├──────────────► k-NN Training
	│
	▼
Evaluation + Reports
	│
	▼
Real-Time Webcam Inference
```

The workflow is implemented using the following main scripts:

- `src/augment.py` for augmentation and split export
- `src/create_unknown.py` for synthetic unknown samples
- `src/features.py` for handcrafted feature extraction
- `src/train.py` for SVM and k-NN training
- `src/evaluate.py` for saved-model evaluation and plots
- `src/realtime_app.py` for webcam-based inference

---

## Dataset

The project uses a six-class waste image dataset plus a generated `unknown` class.

### Known Classes

| Class | Description |
|---|---|
| Cardboard | Packaging and box-like material |
| Glass | Bottles, jars, and similar transparent or reflective waste |
| Metal | Cans and metallic objects |
| Paper | Paper sheets, printed pages, and paper waste |
| Plastic | Bottles, wrappers, containers, and synthetic plastic waste |
| Trash | Mixed waste that does not fit the recyclable categories |

### Unknown Class

The `unknown` class is used as a safety fallback for samples that are too ambiguous or too low-confidence to classify into the six core categories. This is helpful during both evaluation and real-time use.

### Data Layout

```text
dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
├── trash/
└── unknown/

augmented/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── trash/
│   └── unknown/
└── test/
	├── cardboard/
	├── glass/
	├── metal/
	├── paper/
	├── plastic/
	├── trash/
	└── unknown/
```

The training split is augmented to balance classes and improve robustness. The test split is kept untouched except for the class export step.

---

## Feature Engineering

The project uses a handcrafted feature vector built from multiple image descriptors. This keeps the system lightweight, interpretable, and compatible with classical ML models.

### Extracted Features

- HOG for shape and gradient structure
- Color histograms in BGR, HSV, and LAB spaces
- HSV color moments
- Local Binary Pattern for texture
- GLCM texture statistics
- Hu moments for shape invariants
- Canny edge density and contour statistics
- Gabor texture responses

### Why these features?

The selected descriptors capture both appearance and texture cues. That matters because waste categories often differ by material texture, shape, and color distribution rather than fine-grained semantic content.

---

## Models

Two classical models are trained and compared:

| Model | Role | Tuning |
|---|---|---|
| SVM | Main classifier with strong margin-based separation | `GridSearchCV` over `C`, `gamma`, and `kernel` |
| k-NN | Simple baseline and comparison model | `GridSearchCV` over `n_neighbors`, distance metric, and weighting |

Both models use the same scaled feature vectors and the same encoded labels.

---

## Results

Current saved-model results from the repository artifacts:

| Model | All-class Accuracy | All-class Macro F1 | Known-class Accuracy |
|---|---:|---:|---:|
| SVM | 0.7903 | 0.7606 | 0.76 |
| k-NN | 0.7119 | 0.6662 | 0.67 |

---

## Real-Time Application

The real-time app is implemented in `src/realtime_app.py` and uses:

- OpenCV webcam capture
- A center-crop region of interest
- The saved scaler and best model
- A confidence threshold to fall back to `unknown`

At runtime, the app shows the predicted label, confidence, top predictions, and FPS.

---

## Repository Structure

```text
Project/
├── augmented/
├── dataset/
├── features/
├── models/
├── notebooks/
│   └── train_colab.ipynb
├── reports/
├── src/
│   ├── augment.py
│   ├── config.py
│   ├── create_unknown.py
│   ├── evaluate.py
│   ├── features.py
│   ├── realtime_app.py
│   ├── test_single_image.py
│   ├── train.py
│   └── utils.py
├── Final Project.pdf
├── README.md
└── requirements.txt
```

---

## Setup & Reproducibility

### Prerequisites

- Python 3.12+
- `pip`
- OpenCV
- scikit-learn
- scikit-image
- joblib
- matplotlib and seaborn for evaluation plots

### 1. Install dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 2. Generate the augmented dataset

```bash
python src/augment.py
python src/create_unknown.py
```

### 3. Extract features

```bash
python src/features.py
```

### 4. Train the models

```bash
python src/train.py
```

### 5. Evaluate the saved models

```bash
python src/evaluate.py
```

### 6. Run the real-time demo

```bash
python src/realtime_app.py
```

---

## Colab Option

If your local machine cannot handle full training, use the notebook in `notebooks/train_colab.ipynb`.

Recommended workflow:

1. Upload `project_for_colab.zip` to Colab.
2. Extract it under `/content/project`.
3. Set `PROJECT_ROOT = "/content/project"` in the notebook.
4. Run the install, verification, training, and evaluation cells.

The notebook includes both a full mode that calls `train.main()` and a reduced fast mode for lighter runs.

---

## Reproducibility Notes

- The file ordering used in feature extraction is deterministic.
- Random seeds are fixed in the data preparation and training pipeline.
- Model and report artifacts are saved under `models/` and `reports/`.
- The best model is selected by macro F1 on the test split.

For exact reruns, regenerate the augmented data, regenerate features, retrain, and then re-evaluate in that order.

---

## Output Artifacts

After training, the project writes:

- `features/X_train.npy`
- `features/X_test.npy`
- `features/y_train.npy`
- `features/y_test.npy`
- `features/label_encoder.pkl`
- `features/scaler.pkl`
- `models/svm_model.pkl`
- `models/knn_model.pkl`
- `models/best_model.pkl`
- `reports/classification_report_svm.txt`
- `reports/classification_report_knn.txt`
- confusion matrix PNG files in `reports/`

---

## Limitations

- The current saved models do not reach the PDF’s 0.85 primary-class accuracy threshold.
- The project uses handcrafted features rather than a deep learning backbone, so performance is limited by feature quality and model capacity.
- The real-time app depends on webcam availability and OpenCV GUI support.

---
## Team Members

- Hady El Fadaly - [Github Profile](https://github.com/hadyelfadaly)
- Hozayfa Ashraf - [Github Profile](https://github.com/HozayfaAshraf)
- Omar Waleed El Sobky - [Github Profile](https://github.com/Omarsobky)
- Yassin Mohy Eldin - [Github Profile](https://github.com/Yassin-Mohy)
- Ibrahim Wael El Noty - [Github Profile](https://github.com/ibrahimelnouty)

---

## Future Improvements

- Try stronger feature selection or dimensionality reduction.
- Evaluate a tuned linear or calibrated classifier on the handcrafted feature space.
- Compare with a lightweight CNN baseline.
- Add a proper hold-out validation split for model selection separate from final test reporting.

---

## Final Notes

This repository is meant to show a complete end-to-end material classification pipeline: data preparation, augmentation, feature engineering, training, evaluation, and deployment. The structure is in place, but the reported accuracy still needs improvement before it fully satisfies the assignment threshold.
