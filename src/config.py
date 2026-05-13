from pathlib import Path

# Root folder of the project:
# material-stream-classifier/
BASE_DIR = Path(__file__).resolve().parents[1]

# Main folders
DATASET_DIR = BASE_DIR / "dataset"
DATA_DIR = BASE_DIR / "data"
AUGMENTED_DIR = DATA_DIR / "augmented"
UNKNOWN_DIR = DATA_DIR / "unknown"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Image settings
IMG_SIZE = (128, 128)

# Dataset settings
TARGET_COUNT = 500
TEST_SIZE = 0.20
SEED = 42

# Supported image formats
VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def create_project_folders():
    """
    Create required output folders if they do not already exist.
    """
    folders = [
        DATA_DIR,
        AUGMENTED_DIR,
        AUGMENTED_DIR / "train",
        AUGMENTED_DIR / "test",
        UNKNOWN_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_folders()
    print("Project folders created successfully.")