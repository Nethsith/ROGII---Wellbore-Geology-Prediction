from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
REPORT_DIR = PROJECT_ROOT / "reports"

CATBOOST_PREDICTIONS_PATH = (
    REPORT_DIR / "catboost_validation_predictions.csv"
)

ENRICHED_PREDICTIONS_PATH = (
    REPORT_DIR / "catboost_enriched_validation_predictions.csv"
)

BY_WELL_PATH = REPORT_DIR / "catboost_error_by_well.csv"
BY_DISTANCE_PATH = REPORT_DIR / "catboost_error_by_distance_from_ps.csv"
BY_GR_PATH = REPORT_DIR / "catboost_error_by_gr_missing.csv"
BY_ZONE_LENGTH_PATH = (
    REPORT_DIR / "catboost_error_by_prediction_zone_length.csv"
)
REPORT_PATH = REPORT_DIR / "catboost_error_report.json"


# ============================================================
# Helpers
# ============================================================
def get_well_id_from_gold_file(file_path: Path) -> str:
    """
    Example:
    train__015fe0d2__actual_ps__gold_features.csv
    -> 015fe0d2
    """
    parts = file_path.name.split("__")

    if len(parts) < 2:
        raise ValueError(f"Unexpected Gold filename: {file_path.name}")

    return parts[1]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def safe_correlation(x: pd.Series, y: pd.Series) -> float | None:
    """Return correlation safely, or None when it cannot be calculated."""
    valid = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(valid) < 2:
        return None

    if valid["x"].std() == 0 or valid["y"].std() == 0:
        return None

    return float(valid["x"].corr(valid["y"]))


def error_metrics(df: pd.DataFrame) -> dict:
    """Calculate standard regression error metrics."""
    actual = df["actual_TVT"].to_numpy(dtype=float)
    predicted = df["predicted_TVT"].to_numpy(dtype=float)

    residual = predicted - actual
    absolute_error = np.abs(residual)

    return {
        "rows": int(len(df)),
        "RMSE": rmse(actual, predicted),
        "MAE": float(absolute_error.mean()),
        "median_absolute_error": float(np.median(absolute_error)),
        "mean_bias_predicted_minus_actual": float(residual.mean()),
    }


def aggregate_error_metrics(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Calculate error metrics for every category in one column."""
    records = []

    for group_name, group_df in df.groupby(
        group_column,
        dropna=False,
        observed=False,
    ):
        row = {
            group_column: str(group_name),
            **error_metrics(group_df),
            "GR_missing_rate": float(group_df["GR_missing"].mean()),
            "mean_MD_since_PS": float(group_df["MD_since_PS"].mean()),
            "mean_progress_after_PS": float(
                group_df["progress_after_PS"].mean()
            ),
        }

        records.append(row)

    return pd.DataFrame(records)


def load_catboost_predictions() -> pd.DataFrame:
    if not CATBOOST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"CatBoost predictions not found:\n{CATBOOST_PREDICTIONS_PATH}\n"
            "Run 05_train_catboost.py first."
        )

    required_columns = [
        "well_id",
        "source_row_index",
        "actual_TVT",
        "predicted_TVT",
    ]

    predictions = pd.read_csv(
        CATBOOST_PREDICTIONS_PATH,
        dtype={
            "well_id": "string",
            "source_row_index": "int32",
        },
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            "CatBoost prediction file is missing columns: "
            f"{missing_columns}"
        )

    predictions["well_id"] = predictions["well_id"].astype("string")

    return predictions


def load_gold_metadata(validation_wells: set[str]) -> pd.DataFrame:
    """
    Load only required Gold columns for the CatBoost validation wells.
    """
    train_dir = GOLD_DIR / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Gold train folder not found: {train_dir}")

    gold_files = sorted(train_dir.glob("*__gold_features.csv"))

    if not gold_files:
        raise FileNotFoundError(
            f"No Gold files found inside: {train_dir}"
        )

    metadata_columns = [
        "well_id",
        "source_row_index",
        "MD_since_PS",
        "GR_missing",
        "GR_coverage_51",
        "Z_since_PS",
        "spatial_distance_since_PS",
    ]

    metadata_blocks = []
    found_wells = set()

    print(f"Reading Gold context from {len(validation_wells)} validation wells...")

    for file_path in gold_files:
        well_id = get_well_id_from_gold_file(file_path)

        if well_id not in validation_wells:
            continue

        gold_df = pd.read_csv(
            file_path,
            usecols=metadata_columns,
            dtype={
                "well_id": "string",
                "source_row_index": "int32",
                "MD_since_PS": "float32",
                "GR_missing": "int8",
                "GR_coverage_51": "float32",
                "Z_since_PS": "float32",
                "spatial_distance_since_PS": "float32",
            },
        )

        gold_df["well_id"] = gold_df["well_id"].astype("string")

        metadata_blocks.append(gold_df)
        found_wells.add(well_id)

    missing_wells = validation_wells - found_wells

    if missing_wells:
        raise ValueError(
            "Gold files missing for validation wells:\n"
            f"{sorted(missing_wells)}"
        )

    if not metadata_blocks:
        raise ValueError("No Gold metadata was loaded.")

    metadata = pd.concat(metadata_blocks, ignore_index=True)

    duplicated = metadata.duplicated(
        subset=["well_id", "source_row_index"]
    )

    if duplicated.any():
        raise ValueError(
            "Duplicate well_id + source_row_index rows found in Gold data.\n"
            "Delete data/gold and rebuild script 03."
        )

    return metadata


def make_progress_band(progress: pd.Series) -> pd.Series:
    """Create relative-distance bands from Prediction Start to zone end."""
    bins = [-0.001, 0.10, 0.20, 0.40, 0.60, 0.80, 1.001]

    labels = [
        "0–10% from PS",
        "10–20% from PS",
        "20–40% from PS",
        "40–60% from PS",
        "60–80% from PS",
        "80–100% from PS",
    ]

    return pd.cut(
        progress.clip(lower=0, upper=1),
        bins=bins,
        labels=labels,
        include_lowest=True,
    )


def make_zone_length_bands(well_summary: pd.DataFrame) -> pd.Series:
    """Create five approximately equal well groups by prediction-zone size."""
    unique_lengths = well_summary["prediction_zone_rows"].nunique()

    if unique_lengths < 2:
        return pd.Series(
            ["All wells"] * len(well_summary),
            index=well_summary.index,
        )

    number_of_bins = min(5, unique_lengths)

    return pd.qcut(
        well_summary["prediction_zone_rows"],
        q=number_of_bins,
        duplicates="drop",
    ).astype("string")


def python_value(value):
    """Convert NumPy values to normal Python values for JSON."""
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    return value


# ============================================================
# Main
# ============================================================
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CatBoost Error Analysis")
    print("=" * 70)

    predictions = load_catboost_predictions()

    validation_wells = set(predictions["well_id"].dropna().unique())

    print(f"Validation wells: {len(validation_wells)}")
    print(f"Validation rows:  {len(predictions):,}")

    metadata = load_gold_metadata(validation_wells)

    # --------------------------------------------------------
    # Merge predictions with Gold context
    # --------------------------------------------------------
    key_columns = ["well_id", "source_row_index"]

    merged = predictions.merge(
        metadata,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged[merged["_merge"] != "both"].copy()

    if not unmatched.empty:
        unmatched_path = REPORT_DIR / "catboost_unmatched_rows.csv"
        unmatched.to_csv(unmatched_path, index=False)

        raise ValueError(
            "Prediction rows and Gold rows do not match.\n"
            f"Unmatched rows: {len(unmatched):,}\n"
            f"Details: {unmatched_path}"
        )

    merged = merged.drop(columns="_merge")

    # --------------------------------------------------------
    # Error calculations
    # --------------------------------------------------------
    merged["residual"] = (
        merged["predicted_TVT"] - merged["actual_TVT"]
    )

    merged["absolute_error"] = merged["residual"].abs()
    merged["squared_error"] = merged["residual"] ** 2

    # Distance progress inside each well's prediction zone.
    max_distance_per_well = merged.groupby("well_id")[
        "MD_since_PS"
    ].transform("max")

    merged["progress_after_PS"] = np.where(
        max_distance_per_well > 0,
        merged["MD_since_PS"] / max_distance_per_well,
        0.0,
    )

    merged["progress_after_PS"] = (
        merged["progress_after_PS"]
        .clip(lower=0, upper=1)
        .astype("float32")
    )

    merged["distance_from_PS_band"] = make_progress_band(
        merged["progress_after_PS"]
    )

    merged["GR_status"] = np.where(
        merged["GR_missing"] == 1,
        "GR missing at row",
        "GR available at row",
    )

    # Save all enriched row-level analysis data.
    merged.to_csv(ENRICHED_PREDICTIONS_PATH, index=False)

    # --------------------------------------------------------
    # Error by well
    # --------------------------------------------------------
    well_records = []

    for well_id, well_df in merged.groupby("well_id", observed=False):
        row = {
            "well_id": str(well_id),
            **error_metrics(well_df),
            "prediction_zone_rows": int(len(well_df)),
            "prediction_zone_max_MD_since_PS": float(
                well_df["MD_since_PS"].max()
            ),
            "mean_MD_since_PS": float(
                well_df["MD_since_PS"].mean()
            ),
            "GR_missing_rate": float(well_df["GR_missing"].mean()),
            "mean_GR_coverage_51": float(
                well_df["GR_coverage_51"].mean()
            ),
            "mean_spatial_distance_since_PS": float(
                well_df["spatial_distance_since_PS"].mean()
            ),
        }

        well_records.append(row)

    by_well = pd.DataFrame(well_records).sort_values(
        by="RMSE",
        ascending=False,
    )

    by_well["prediction_zone_length_band"] = make_zone_length_bands(
        by_well
    )

    by_well.to_csv(BY_WELL_PATH, index=False)

    # --------------------------------------------------------
    # Error by relative distance from PS
    # --------------------------------------------------------
    by_distance = aggregate_error_metrics(
        merged,
        group_column="distance_from_PS_band",
    )

    distance_order = [
        "0–10% from PS",
        "10–20% from PS",
        "20–40% from PS",
        "40–60% from PS",
        "60–80% from PS",
        "80–100% from PS",
    ]

    by_distance["distance_from_PS_band"] = pd.Categorical(
        by_distance["distance_from_PS_band"],
        categories=distance_order,
        ordered=True,
    )

    by_distance = by_distance.sort_values("distance_from_PS_band")
    by_distance.to_csv(BY_DISTANCE_PATH, index=False)

    # --------------------------------------------------------
    # Error by GR availability
    # --------------------------------------------------------
    by_gr = aggregate_error_metrics(
        merged,
        group_column="GR_status",
    ).sort_values(
        by="RMSE",
        ascending=False,
    )

    by_gr.to_csv(BY_GR_PATH, index=False)

    # --------------------------------------------------------
    # Error by prediction-zone length
    # Each well gets equal importance in this table.
    # --------------------------------------------------------
    zone_length_records = []

    for band, band_df in by_well.groupby(
        "prediction_zone_length_band",
        dropna=False,
        observed=False,
    ):
        zone_length_records.append(
            {
                "prediction_zone_length_band": str(band),
                "well_count": int(len(band_df)),
                "min_prediction_rows": int(
                    band_df["prediction_zone_rows"].min()
                ),
                "max_prediction_rows": int(
                    band_df["prediction_zone_rows"].max()
                ),
                "mean_well_RMSE": float(band_df["RMSE"].mean()),
                "median_well_RMSE": float(band_df["RMSE"].median()),
                "mean_well_MAE": float(band_df["MAE"].mean()),
                "mean_GR_missing_rate": float(
                    band_df["GR_missing_rate"].mean()
                ),
            }
        )

    by_zone_length = pd.DataFrame(zone_length_records).sort_values(
        by="min_prediction_rows"
    )

    by_zone_length.to_csv(BY_ZONE_LENGTH_PATH, index=False)

    # --------------------------------------------------------
    # Save summary report
    # --------------------------------------------------------
    overall_metrics = error_metrics(merged)

    gr_available = merged[merged["GR_missing"] == 0]
    gr_missing = merged[merged["GR_missing"] == 1]

    worst_well = by_well.iloc[0]
    best_well = by_well.iloc[-1]

    report = {
        "validation_rows": int(len(merged)),
        "validation_wells": int(merged["well_id"].nunique()),
        "overall_metrics": {
            key: python_value(value)
            for key, value in overall_metrics.items()
        },
        "correlations": {
            "absolute_error_vs_MD_since_PS": safe_correlation(
                merged["absolute_error"],
                merged["MD_since_PS"],
            ),
            "absolute_error_vs_progress_after_PS": safe_correlation(
                merged["absolute_error"],
                merged["progress_after_PS"],
            ),
            "absolute_error_vs_GR_missing": safe_correlation(
                merged["absolute_error"],
                merged["GR_missing"],
            ),
        },
        "RMSE_by_GR_status": {
            "GR_available": (
                rmse(
                    gr_available["actual_TVT"].to_numpy(dtype=float),
                    gr_available["predicted_TVT"].to_numpy(dtype=float),
                )
                if len(gr_available) > 0
                else None
            ),
            "GR_missing": (
                rmse(
                    gr_missing["actual_TVT"].to_numpy(dtype=float),
                    gr_missing["predicted_TVT"].to_numpy(dtype=float),
                )
                if len(gr_missing) > 0
                else None
            ),
        },
        "worst_RMSE_well": {
            "well_id": str(worst_well["well_id"]),
            "RMSE": python_value(worst_well["RMSE"]),
            "MAE": python_value(worst_well["MAE"]),
            "prediction_zone_rows": python_value(
                worst_well["prediction_zone_rows"]
            ),
            "GR_missing_rate": python_value(
                worst_well["GR_missing_rate"]
            ),
        },
        "best_RMSE_well": {
            "well_id": str(best_well["well_id"]),
            "RMSE": python_value(best_well["RMSE"]),
            "MAE": python_value(best_well["MAE"]),
            "prediction_zone_rows": python_value(
                best_well["prediction_zone_rows"]
            ),
            "GR_missing_rate": python_value(
                best_well["GR_missing_rate"]
            ),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Overall CatBoost Validation Error")
    print("=" * 70)
    print(f"RMSE: {overall_metrics['RMSE']:.6f}")
    print(f"MAE:  {overall_metrics['MAE']:.6f}")

    print("\nError by GR status:")
    print(
        by_gr[
            ["GR_status", "rows", "RMSE", "MAE", "GR_missing_rate"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nWorst 10 wells by RMSE:")
    print(
        by_well.head(10)[
            [
                "well_id",
                "RMSE",
                "MAE",
                "prediction_zone_rows",
                "GR_missing_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved files")
    print(f"Enriched predictions: {ENRICHED_PREDICTIONS_PATH}")
    print(f"Error by well:        {BY_WELL_PATH}")
    print(f"Error by distance:    {BY_DISTANCE_PATH}")
    print(f"Error by GR status:   {BY_GR_PATH}")
    print(f"Error by zone length: {BY_ZONE_LENGTH_PATH}")
    print(f"Summary report:       {REPORT_PATH}")


if __name__ == "__main__":
    main()