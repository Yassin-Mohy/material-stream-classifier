import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, SEED


def load_features():
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")

    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")

    return X_train, X_test, y_train, y_test, label_encoder


def train_svm(X_train, y_train):
    print("\nTraining SVM with GridSearchCV...")

    parameter_grid = {
        "C": [1, 3, 10, 30, 100],
        "gamma": ["scale", 0.01, 0.003, 0.001],
        "kernel": ["rbf"]
    }

    svm = SVC(
        probability=True,
        class_weight="balanced",
        random_state=SEED
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"Best SVM parameters: {grid_search.best_params_}")
    print(f"Best SVM CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_knn(X_train, y_train):
    print("\nTraining k-NN with GridSearchCV...")

    parameter_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

    knn = KNeighborsClassifier()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print(f"Best k-NN parameters: {grid_search.best_params_}")
    print(f"Best k-NN CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    print("\n" + "=" * 60)
    print(f"{model_name} Results")
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

    with open(REPORTS_DIR / f"classification_report_{model_name.lower()}.txt", "w") as file:
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro F1-score: {f1:.4f}\n\n")
        file.write(report)

    return accuracy, f1


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, label_encoder = load_features()

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    svm_model = train_svm(X_train, y_train)
    knn_model = train_knn(X_train, y_train)

    svm_accuracy, svm_f1 = evaluate_model(
        svm_model,
        X_test,
        y_test,
        label_encoder,
        "SVM"
    )

    knn_accuracy, knn_f1 = evaluate_model(
        knn_model,
        X_test,
        y_test,
        label_encoder,
        "KNN"
    )

    joblib.dump(svm_model, MODELS_DIR / "svm_model.pkl")
    joblib.dump(knn_model, MODELS_DIR / "knn_model.pkl")

    if svm_f1 >= knn_f1:
        joblib.dump(svm_model, MODELS_DIR / "best_model.pkl")
        print("\nBest model: SVM")
    else:
        joblib.dump(knn_model, MODELS_DIR / "best_model.pkl")
        print("\nBest model: k-NN")

    print("\nFinal Model Comparison")
    print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print(f"k-NN Accuracy: {knn_accuracy:.4f}")


if __name__ == "__main__":
    main()