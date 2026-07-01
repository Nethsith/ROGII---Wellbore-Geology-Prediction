import kagglehub
from pathlib import Path
import subprocess

COMPETITION_NAME = "rogii-wellbore-geology-prediction"

# Project root = parent folder of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/raw
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    # If dataset already downloaded, do not download again
    if RAW_DATA_DIR.exists() and any(RAW_DATA_DIR.iterdir()):
        print("Dataset already exists at:", RAW_DATA_DIR)
        return RAW_DATA_DIR

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "kaggle",
        "competitions",
        "download",
        "-c", COMPETITION_NAME,
        "-p", str(RAW_DATA_DIR),
        "--force"
    ], check=True)

    print("Download completed!")

    print("Dataset downloaded to:", RAW_DATA_DIR)
    return RAW_DATA_DIR


if __name__ == "__main__":
    download_dataset()