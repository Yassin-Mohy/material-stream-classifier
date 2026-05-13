import cv2
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import DATASET_DIR, AUGMENTED_DIR, IMG_SIZE, TARGET_COUNT, TEST_SIZE, SEED
from utils import collect_image_paths, save_image

random.seed(SEED)
np.random.seed(SEED)


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_crop_resize(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.75, 0.95)

    crop_h = int(h * scale)
    crop_w = int(w * scale)

    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)

    crop = img[y:y + crop_h, x:x + crop_w]
    return cv2.resize(crop, (w, h))


def adjust_brightness_contrast(img):
    alpha = random.uniform(0.75, 1.35)
    beta = random.randint(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def blur_or_noise(img):
    choice = random.choice(["blur", "noise", "none"])

    if choice == "blur":
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)

    if choice == "noise":
        noise = np.random.normal(0, random.randint(5, 20), img.shape).astype(np.int16)
        noisy = img.astype(np.int16) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return img


def random_augment(img):
    operations = [
        lambda x: rotate_image(x, random.choice([-25, -15, 15, 25])),
        lambda x: cv2.flip(x, 1),
        random_crop_resize,
        adjust_brightness_contrast,
        blur_or_noise,
    ]

    selected_operations = random.sample(operations, random.randint(1, 3))

    for operation in selected_operations:
        img = operation(img)

    return img


def save_original_training_images(class_name, train_paths):
    output_dir = AUGMENTED_DIR / "train" / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for path in tqdm(train_paths, desc=f"Saving originals for {class_name}"):
        img = cv2.imread(str(path))

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        save_image(output_dir / f"orig_{saved:05d}.jpg", img)
        saved += 1

    return saved


def augment_training_class(class_name, train_paths):
    output_dir = AUGMENTED_DIR / "train" / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = save_original_training_images(class_name, train_paths)

    while saved < TARGET_COUNT:
        path = random.choice(train_paths)
        img = cv2.imread(str(path))

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        augmented_img = random_augment(img)

        save_image(output_dir / f"aug_{saved:05d}.jpg", augmented_img)
        saved += 1

    return saved


def save_test_class(class_name, test_paths):
    output_dir = AUGMENTED_DIR / "test" / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for path in tqdm(test_paths, desc=f"Saving test images for {class_name}"):
        img = cv2.imread(str(path))

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        save_image(output_dir / f"test_{saved:05d}.jpg", img)
        saved += 1

    return saved


def main():
    print("=" * 60)
    print("Material Stream Classifier - Data Preparation")
    print("=" * 60)

    class_dirs = sorted([d for d in DATASET_DIR.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"No class folders found in: {DATASET_DIR}")
        return

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = collect_image_paths(class_dir)

        if len(image_paths) < 2:
            print(f"Skipping {class_name}: not enough images.")
            continue

        train_paths, test_paths = train_test_split(
            image_paths,
            test_size=TEST_SIZE,
            random_state=SEED,
            shuffle=True
        )

        print(f"\nClass: {class_name}")
        print(f"Original images: {len(image_paths)}")
        print(f"Train images before augmentation: {len(train_paths)}")
        print(f"Test images: {len(test_paths)}")

        train_count = augment_training_class(class_name, train_paths)
        test_count = save_test_class(class_name, test_paths)

        print(f"Saved training images: {train_count}")
        print(f"Saved testing images: {test_count}")

    print("\nData preparation complete.")
    print("Next step: python src/create_unknown.py")


if __name__ == "__main__":
    main()