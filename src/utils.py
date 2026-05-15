import cv2
from pathlib import Path

from config import VALID_EXTENSIONS


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_image_file(path):
    return Path(path).suffix.lower() in VALID_EXTENSIONS


def collect_image_paths(folder):
    folder = Path(folder)
    # Deterministic ordering avoids split drift across different filesystems.
    return sorted([p for p in folder.iterdir() if p.is_file() and is_image_file(p)])


def read_image(path):
    return cv2.imread(str(path))


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
