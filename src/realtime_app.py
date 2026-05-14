import cv2
import time
import joblib
import numpy as np

from config import MODELS_DIR, FEATURES_DIR
from features import extract_features


CONFIDENCE_THRESHOLD = 0.40


def predict_material(model, scaler, label_encoder, roi):
    feature_vector = extract_features(roi).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)

    probabilities = model.predict_proba(feature_vector)[0]
    sorted_indices = np.argsort(probabilities)[::-1]

    best_index = sorted_indices[0]
    best_confidence = float(probabilities[best_index])
    best_label = label_encoder.inverse_transform([best_index])[0]

    top_predictions = []

    for index in sorted_indices[:3]:
        label = label_encoder.inverse_transform([index])[0]
        confidence = float(probabilities[index])
        top_predictions.append((label, confidence))

    if best_confidence < CONFIDENCE_THRESHOLD:
        return "unknown", best_confidence, top_predictions

    return best_label, best_confidence, top_predictions


def main():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(FEATURES_DIR / "scaler.pkl")
    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1).")
        return

    label = "Loading..."
    confidence = 0.0
    top_predictions = []

    frame_count = 0
    start_time = time.time()

    print("Real-time app started.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        height, width = frame.shape[:2]

        box_size = 300
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = width // 2 + box_size // 2
        y2 = height // 2 + box_size // 2

        roi = frame[y1:y2, x1:x2]

        if frame_count % 10 == 0 and roi.size > 0:
            label, confidence, top_predictions = predict_material(
                model,
                scaler,
                label_encoder,
                roi
            )

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{label.upper()} ({confidence * 100:.1f}%)",
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            (0, 255, 0),
            2
        )

        y_text = 90

        for pred_label, pred_conf in top_predictions:
            cv2.putText(
                frame,
                f"{pred_label}: {pred_conf * 100:.1f}%",
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            y_text += 30

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Press Q to quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Material Stream Classifier", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()