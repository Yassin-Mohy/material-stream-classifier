import cv2
import numpy as np
import random

from config import AUGMENTED_DIR, IMG_SIZE, TARGET_COUNT, SEED
from utils import save_image

random.seed(SEED)
np.random.seed(SEED)


def create_noise_image():
    return np.random.randint(
        0,
        256,
        (IMG_SIZE[1], IMG_SIZE[0], 3),
        dtype=np.uint8
    )


def create_blurred_noise_image():
    img = create_noise_image()
    return cv2.GaussianBlur(img, (31, 31), 0)


def create_plain_image():
    value = random.choice([0, 255, 128])
    img = np.full((IMG_SIZE[1], IMG_SIZE[0], 3), value, dtype=np.uint8)

    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img


def create_random_shapes_image():
    img = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8)

    for _ in range(random.randint(3, 10)):
        color = tuple(np.random.randint(0, 256, 3).tolist())

        x1 = random.randint(0, IMG_SIZE[0] - 1)
        y1 = random.randint(0, IMG_SIZE[1] - 1)
        x2 = random.randint(0, IMG_SIZE[0] - 1)
        y2 = random.randint(0, IMG_SIZE[1] - 1)

        if random.choice(["circle", "rectangle"]) == "circle":
            radius = random.randint(5, 30)
            cv2.circle(img, (x1, y1), radius, color, -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    return img


def generate_unknown_dataset():
    train_dir = AUGMENTED_DIR / "train" / "unknown"
    test_dir = AUGMENTED_DIR / "test" / "unknown"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        create_noise_image,
        create_blurred_noise_image,
        create_plain_image,
        create_random_shapes_image
    ]

    for i in range(TARGET_COUNT):
        generator = random.choice(generators)
        img = generator()
        save_image(train_dir / f"unknown_train_{i:05d}.jpg", img)

    for i in range(100):
        generator = random.choice(generators)
        img = generator()
        save_image(test_dir / f"unknown_test_{i:05d}.jpg", img)

    print("Unknown class generated successfully.")
    print(f"Training unknown images: {TARGET_COUNT}")
    print("Testing unknown images: 100")


def main():
    generate_unknown_dataset()


if __name__ == "__main__":
    main()