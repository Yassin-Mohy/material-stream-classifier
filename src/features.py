import cv2
import joblib
import numpy as np
from tqdm import tqdm

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import AUGMENTED_DIR, FEATURES_DIR, IMG_SIZE
from utils import collect_image_paths


def extract_hog(img_gray):
    return hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )


def extract_color_histograms(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    features = []

    for image in [img_bgr, img_hsv, img_lab]:
        for channel in range(3):
            hist = cv2.calcHist(
                [image],
                [channel],
                None,
                [32],
                [0, 256]
            ).flatten()

            hist = hist / (hist.sum() + 1e-7)
            features.extend(hist)

    return np.array(features)


def extract_color_moments(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    features = []

    for channel in range(3):
        pixels = img_hsv[:, :, channel].astype(np.float32)

        features.append(np.mean(pixels))
        features.append(np.std(pixels))
        features.append(np.median(pixels))

    return np.array(features)


def extract_lbp(img_gray):
    lbp = local_binary_pattern(
        img_gray,
        P=24,
        R=3,
        method="uniform"
    )

    hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
    hist = hist.astype(np.float32)

    return hist / (hist.sum() + 1e-7)


def extract_glcm(img_gray):
    glcm = graycomatrix(
        img_gray,
        distances=[1, 2, 4],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = []

    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]:
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)

    return np.array(features)


def extract_hu_moments(img_gray):
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()

    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)


def extract_edge_features(img_gray):
    edges = cv2.Canny(img_gray, 100, 200)

    edge_density = np.sum(edges > 0) / edges.size

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.array([edge_density, 0, 0, 0])

    areas = [cv2.contourArea(c) for c in contours]

    return np.array([
        edge_density,
        len(contours),
        np.mean(areas),
        np.std(areas)
    ])


def extract_gabor_features(img_gray):
    features = []

    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0
        )

        filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)

        features.append(filtered.mean())
        features.append(filtered.std())

    return np.array(features)


def extract_features(img_bgr):
    img = cv2.resize(img_bgr, IMG_SIZE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = np.concatenate([
        extract_hog(img_gray),
        extract_color_histograms(img),
        extract_color_moments(img),
        extract_lbp(img_gray),
        extract_glcm(img_gray),
        extract_hu_moments(img_gray),
        extract_edge_features(img_gray),
        extract_gabor_features(img_gray)
    ])

    return features.astype(np.float32)


def load_dataset(split_name):
    X = []
    y = []

    split_dir = AUGMENTED_DIR / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Folder not found: {split_dir}. Run augment.py and create_unknown.py first."
        )

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = collect_image_paths(class_dir)

        print(f"{split_name.upper()} - {class_name}: {len(image_paths)} images")

        for path in tqdm(image_paths):
            img = cv2.imread(str(path))

            if img is None:
                continue

            feature_vector = extract_features(img)

            X.append(feature_vector)
            y.append(class_name)

    return np.array(X), np.array(y)


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Feature Extraction")
    print("=" * 60)

    X_train, y_train_raw = load_dataset("train")
    X_test, y_test_raw = load_dataset("test")

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(FEATURES_DIR / "X_train.npy", X_train_scaled)
    np.save(FEATURES_DIR / "X_test.npy", X_test_scaled)
    np.save(FEATURES_DIR / "y_train.npy", y_train)
    np.save(FEATURES_DIR / "y_test.npy", y_test)

    joblib.dump(label_encoder, FEATURES_DIR / "label_encoder.pkl")
    joblib.dump(scaler, FEATURES_DIR / "scaler.pkl")

    print("\nFeature extraction complete.")
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Testing shape: {X_test_scaled.shape}")
    print(f"Classes: {label_encoder.classes_}")


if __name__ == "__main__":
    main()