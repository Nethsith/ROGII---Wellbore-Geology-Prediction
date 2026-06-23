import kagglehub
from pathlib import Path

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

    path = kagglehub.competition_download(
        COMPETITION_NAME,
        output_dir=str(RAW_DATA_DIR)
    )

    print("Dataset downloaded to:", path)
    return Path(path)


if __name__ == "__main__":
    download_dataset()