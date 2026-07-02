from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

MANIFEST_PATH = INTERIM_DIR / "ps_manifest.csv"
REPORT_PATH = INTERIM_DIR / "ps_validation_report.csv"


def get_well_id(file_path: Path) -> str:
    """Extract well ID from: 00bbac68__horizontal_well.csv"""
    return file_path.name.split("__", 1)[0]


def find_prediction_start(tvt_input: pd.Series) -> int | None:
    """
    Return the first row where TVT_input becomes NaN.

    Example:
    known, known, known, NaN, NaN
                        ↑
                prediction_start_row = 3
    """
    missing_mask = tvt_input.isna()

    # No missing TVT_input values means there is no prediction zone.
    if not missing_mask.any():
        return None

    prediction_start_row = int(missing_mask.idxmax())

    # TVT_input must be known before PS.
    if tvt_input.iloc[:prediction_start_row].isna().any():
        raise ValueError(
            "TVT_input has missing values before the Prediction Start point."
        )

    # TVT_input must remain missing after PS.
    if tvt_input.iloc[prediction_start_row:].notna().any():
        raise ValueError(
            "TVT_input becomes available again after the Prediction Start point."
        )

    return prediction_start_row


def inspect_horizontal_well(file_path: Path, split: str) -> dict:
    """Inspect one horizontal-well CSV and identify its Prediction Start."""

    df = pd.read_csv(file_path)
    well_id = get_well_id(file_path)

    record = {
        "scenario_id": f"{split}__{well_id}__actual_ps",
        "split": split,
        "well_id": well_id,
        "source_file": str(file_path.relative_to(PROJECT_ROOT)),
        "rows": len(df),
        "status": "valid",
        "message": "",
        "ps_row": None,
        "prediction_start_row": None,
        "known_rows": None,
        "prediction_rows": None,
        "known_fraction": None,
        "prediction_fraction": None,
        "ps_MD": None,
        "last_known_TVT_input": None,
        "TVT_missing_after_PS": None,
    }

    if "TVT_input" not in df.columns:
        record["status"] = "invalid"
        record["message"] = "TVT_input column not found."
        return record

    if "MD" not in df.columns:
        record["status"] = "invalid"
        record["message"] = "MD column not found."
        return record

    try:
        prediction_start_row = find_prediction_start(df["TVT_input"])
    except ValueError as error:
        record["status"] = "invalid"
        record["message"] = str(error)
        return record

    if prediction_start_row is None:
        record["status"] = "invalid"
        record["message"] = "No prediction zone: TVT_input has no NaN values."
        return record

    if prediction_start_row == 0:
        record["status"] = "invalid"
        record["message"] = "No known TVT_input rows before Prediction Start."
        return record

    ps_row = prediction_start_row - 1
    known_rows = prediction_start_row
    prediction_rows = len(df) - prediction_start_row

    record["ps_row"] = ps_row
    record["prediction_start_row"] = prediction_start_row
    record["known_rows"] = known_rows
    record["prediction_rows"] = prediction_rows
    record["known_fraction"] = known_rows / len(df)
    record["prediction_fraction"] = prediction_rows / len(df)
    record["ps_MD"] = df.loc[ps_row, "MD"]
    record["last_known_TVT_input"] = df.loc[ps_row, "TVT_input"]

    # Training wells must have true TVT values after PS.
    if split == "train":
        if "TVT" not in df.columns:
            record["status"] = "invalid"
            record["message"] = "TVT target column not found in training file."
            return record

        tvt_missing_after_ps = int(
            df.loc[prediction_start_row:, "TVT"].isna().sum()
        )

        record["TVT_missing_after_PS"] = tvt_missing_after_ps

        if tvt_missing_after_ps > 0:
            record["status"] = "invalid"
            record["message"] = (
                "Training TVT contains missing values after Prediction Start."
            )

    return record


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw directory not found: {RAW_DIR}")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for split in ["train", "test"]:
        split_dir = RAW_DIR / split

        if not split_dir.exists():
            print(f"Skipped missing directory: {split_dir}")
            continue

        horizontal_files = sorted(
            split_dir.glob("*__horizontal_well.csv")
        )

        print(f"\nChecking {split}: {len(horizontal_files)} horizontal wells")

        for file_path in horizontal_files:
            record = inspect_horizontal_well(file_path, split)
            records.append(record)

    report_df = pd.DataFrame(records)

    # Full validation report: valid + invalid files.
    report_df.to_csv(REPORT_PATH, index=False)    
    
    if "status" not in report_df.columns:
        raise ValueError(
            f"Missing 'status' column. Available columns: {list(report_df.columns)}"
        )
    
    # Only valid wells become part of the PS manifest.
    manifest_df = report_df[
        report_df["status"] == "valid"
    ].copy()

    manifest_columns = [
        "scenario_id",
        "split",
        "well_id",
        "source_file",
        "rows",
        "ps_row",
        "prediction_start_row",
        "known_rows",
        "prediction_rows",
        "known_fraction",
        "prediction_fraction",
        "ps_MD",
        "last_known_TVT_input",
    ]

    manifest_df = manifest_df[manifest_columns]
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    print("\n" + "=" * 70)
    print("Prediction Start manifest created.")
    print(f"Valid wells:   {len(manifest_df)}")
    print(f"Invalid wells: {(report_df['status'] != 'valid').sum()}")
    print(f"Manifest:      {MANIFEST_PATH}")
    print(f"Full report:   {REPORT_PATH}")
    print("=" * 70)

    print("\nTest-well Prediction Start summary:")
    test_rows = manifest_df[manifest_df["split"] == "test"]

    if test_rows.empty:
        print("No valid test wells found.")
    else:
        print(
            test_rows[
                [
                    "well_id",
                    "ps_row",
                    "prediction_start_row",
                    "known_rows",
                    "prediction_rows",
                    "known_fraction",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()