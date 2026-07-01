import kagglehub
from pathlib import Path
import os

COMPETITION_NAME = "rogii-wellbore-geology-prediction"

# Project root = parent folder of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/raw
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

if KAGGLE_USERNAME is not None:
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME

if KAGGLE_KEY is not None:
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY


def download_dataset():
    # If dataset already downloaded, do not download again
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        print("Dataset already exists at:", RAW_DATA_DIR)
        return RAW_DATA_DIR

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    path = kagglehub.competition_download(
        COMPETITION_NAME,
        output_dir=str(RAW_DATA_DIR)
    )

    print("Dataset downloaded to:", path)
    return Path(path)


if __name__ == "__main__":
    download_dataset()