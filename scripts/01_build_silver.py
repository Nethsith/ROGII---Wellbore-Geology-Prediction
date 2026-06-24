from pathlib import Path
from shutil import copy2

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
REPORT_DIR = SILVER_DIR / "_reports"

HORIZONTAL_REQUIRED = ["MD", "X", "Y", "Z", "GR"]
TYPEWELL_REQUIRED = ["TVT", "GR"]


def get_well_id(file_path: Path) -> str:
    """Get the well ID from files such as 00bbac68__horizontal_well.csv."""
    return file_path.name.split("__", 1)[0]


def get_file_kind(file_path: Path) -> str:
    name = file_path.name.lower()

    if "__horizontal_well" in name:
        return "horizontal_well"

    if "__typewell" in name:
        return "typewell"

    return "other"


def build_silver_file(input_path: Path, output_path: Path, split: str) -> dict:
    """Copy a raw CSV into Silver, preserving all original measurements."""
    df = pd.read_csv(input_path)

    file_kind = get_file_kind(input_path)
    well_id = get_well_id(input_path)

    # Keep all original columns and values unchanged.
    silver_df = df.copy()

    # Safe metadata columns.
    silver_df["well_id"] = well_id
    silver_df["row_index"] = range(len(silver_df))

    # GR remains NaN. We only record whether it is missing.
    if "GR" in silver_df.columns:
        silver_df["GR_missing"] = silver_df["GR"].isna().astype("int8")

    # This identifies rows Kaggle expects you to predict.
    if file_kind == "horizontal_well":
        if "TVT_input" in silver_df.columns:
            silver_df["is_prediction_zone"] = (
                silver_df["TVT_input"].isna().astype("int8")
            )
        else:
            silver_df["is_prediction_zone"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    silver_df.to_csv(output_path, index=False)

    required_columns = (
        HORIZONTAL_REQUIRED
        if file_kind == "horizontal_well"
        else TYPEWELL_REQUIRED
        if file_kind == "typewell"
        else []
    )

    missing_required = [
        column for column in required_columns
        if column not in df.columns
    ]

    report = {
        "split": split,
        "file": input_path.name,
        "file_kind": file_kind,
        "well_id": well_id,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_required_columns": ", ".join(missing_required) or "None",
        "MD_missing": int(df["MD"].isna().sum()) if "MD" in df.columns else None,
        "X_missing": int(df["X"].isna().sum()) if "X" in df.columns else None,
        "Y_missing": int(df["Y"].isna().sum()) if "Y" in df.columns else None,
        "Z_missing": int(df["Z"].isna().sum()) if "Z" in df.columns else None,
        "GR_missing": int(df["GR"].isna().sum()) if "GR" in df.columns else None,
        "TVT_missing": int(df["TVT"].isna().sum()) if "TVT" in df.columns else None,
        "TVT_input_missing": (
            int(df["TVT_input"].isna().sum())
            if "TVT_input" in df.columns
            else None
        ),
    }

    return report


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports = []

    # Copy sample submission without changing it.
    sample_submission = RAW_DIR / "sample_submission.csv"
    if sample_submission.exists():
        copy2(sample_submission, SILVER_DIR / sample_submission.name)

    for split in ["train", "test"]:
        input_dir = RAW_DIR / split

        if not input_dir.exists():
            print(f"Skipped missing directory: {input_dir}")
            continue

        csv_files = sorted(input_dir.rglob("*.csv"))
        print(f"\nProcessing {split}: {len(csv_files)} files")

        for input_path in csv_files:
            relative_path = input_path.relative_to(RAW_DIR)
            output_path = SILVER_DIR / relative_path

            report = build_silver_file(
                input_path=input_path,
                output_path=output_path,
                split=split,
            )

            reports.append(report)
            print(f"Saved: {relative_path}")

    report_df = pd.DataFrame(reports)
    report_path = REPORT_DIR / "silver_validation_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\nSilver dataset created.")
    print(f"Silver data: {SILVER_DIR}")
    print(f"Validation report: {report_path}")


if __name__ == "__main__":
    main()