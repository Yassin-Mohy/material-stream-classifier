import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

from config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR


def evaluate_saved_model(model_filename, model_name):
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")

    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")
    model = joblib.load(MODELS_DIR / model_filename)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print("\n" + "=" * 60)
    print(f"{model_name} Evaluation")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )

    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / f"classification_report_{model_name.lower()}.txt"

    with open(report_path, "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {f1:.4f}\n\n")
        file.write(report)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap="Blues"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{model_name.lower()}.png", dpi=300)
    plt.close()


def main():
    evaluate_saved_model("svm_model.pkl", "SVM")
    evaluate_saved_model("knn_model.pkl", "KNN")


if __name__ == "__main__":
    main()