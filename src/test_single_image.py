import cv2
import joblib
import numpy as np
import sys

from config import MODELS_DIR, FEATURES_DIR
from features import extract_features


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/test_single_image.py path_to_image")
        return

    image_path = sys.argv[1]

    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(FEATURES_DIR / "scaler.pkl")
    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")

    img = cv2.imread(image_path)

    if img is None:
        print("Could not read image:", image_path)
        return

    feature_vector = extract_features(img).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)

    probabilities = model.predict_proba(feature_vector)[0]

    sorted_indices = np.argsort(probabilities)[::-1]

    print("\nPrediction probabilities:")
    for index in sorted_indices:
        label = label_encoder.inverse_transform([index])[0]
        confidence = probabilities[index]
        print(f"{label}: {confidence * 100:.2f}%")

    best_index = sorted_indices[0]
    best_label = label_encoder.inverse_transform([best_index])[0]

    print("\nFinal prediction:", best_label)


if __name__ == "__main__":
    main()