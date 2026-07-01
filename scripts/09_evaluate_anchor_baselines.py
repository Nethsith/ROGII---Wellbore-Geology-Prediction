from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_DIR = PROJECT_ROOT / "data" / "silver"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "reports"

PS_MANIFEST_PATH = INTERIM_DIR / "ps_manifest.csv"
XGB_REPORT_PATH = REPORT_DIR / "xgboost_training_report.json"

PREDICTIONS_PATH = REPORT_DIR / "anchor_baseline_validation_predictions.csv"
RESULTS_PATH = REPORT_DIR / "anchor_baseline_results.csv"
BY_WELL_PATH = REPORT_DIR / "anchor_baseline_by_well.csv"
REPORT_PATH = REPORT_DIR / "anchor_baseline_report.json"


# ============================================================
# Settings
# ============================================================
SLOPE_WINDOW = 100
RIDGE_ALPHA = 1.0

BASELINE_COLUMNS = [
    "MD_slope_baseline_TVT",
    "Z_slope_baseline_TVT",
    "MD_Z_ridge_baseline_TVT",
    "trajectory_ridge_baseline_TVT",
]


# ============================================================
# Helpers
# ============================================================
def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def get_well_id(file_path: Path) -> str:
    return file_path.name.split("__", 1)[0]


def find_horizontal_file(well_id: str) -> Path:
    matches = sorted(
        (SILVER_DIR / "train").glob(
            f"{well_id}__horizontal_well.csv"
        )
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one horizontal file for {well_id}. "
            f"Found: {matches}"
        )

    return matches[0]


def safe_slope(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Calculate y-change per x-change using valid values only."""
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return 0.0

    x_valid = x[valid]
    y_valid = y[valid]

    x_centered = x_valid - x_valid.mean()
    denominator = np.sum(x_centered ** 2)

    if denominator <= 1e-12:
        return 0.0

    return float(
        np.sum(x_centered * (y_valid - y_valid.mean()))
        / denominator
    )


def load_validation_wells() -> set[str]:
    if not XGB_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost report not found: {XGB_REPORT_PATH}\n"
            "Run 04_train_xgboost.py first."
        )

    with open(XGB_REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    validation_wells = report.get("validation_wells")

    if not validation_wells:
        raise ValueError(
            "validation_wells not found in "
            "xgboost_training_report.json"
        )

    return set(validation_wells)


def load_ps_manifest() -> pd.DataFrame:
    if not PS_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"PS manifest not found: {PS_MANIFEST_PATH}\n"
            "Run 02_create_ps_manifest.py first."
        )

    manifest = pd.read_csv(PS_MANIFEST_PATH)

    required_columns = [
        "split",
        "well_id",
        "ps_row",
        "prediction_start_row",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in manifest.columns
    ]

    if missing_columns:
        raise ValueError(
            f"PS manifest missing columns: {missing_columns}"
        )

    return manifest


def fit_ridge_model(
    known_features: np.ndarray,
    known_target: np.ndarray,
    alpha: float,
) -> dict | None:
    """
    Fit Ridge regression manually.

    Standardization is fitted only with pre-PS rows.
    """
    valid = (
        np.isfinite(known_target)
        & np.isfinite(known_features).all(axis=1)
    )

    x = known_features[valid]
    y = known_target[valid]

    if len(x) < max(10, x.shape[1] + 2):
        return None

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)

    feature_std[feature_std < 1e-12] = 1.0

    x_scaled = (x - feature_mean) / feature_std

    x_design = np.column_stack(
        [np.ones(len(x_scaled)), x_scaled]
    )

    penalty = np.eye(x_design.shape[1])
    penalty[0, 0] = 0.0  # Do not penalize intercept.

    try:
        coefficients = np.linalg.solve(
            x_design.T @ x_design + alpha * penalty,
            x_design.T @ y,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(
            x_design.T @ x_design + alpha * penalty
        ) @ (x_design.T @ y)

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "coefficients": coefficients,
    }


def predict_ridge_model(
    model: dict,
    features: np.ndarray,
) -> np.ndarray:
    """Predict using a fitted Ridge model."""
    valid = np.isfinite(features).all(axis=1)

    predictions = np.full(len(features), np.nan, dtype=float)

    if valid.sum() == 0:
        return predictions

    x_scaled = (
        features[valid] - model["feature_mean"]
    ) / model["feature_std"]

    x_design = np.column_stack(
        [np.ones(len(x_scaled)), x_scaled]
    )

    predictions[valid] = x_design @ model["coefficients"]

    return predictions


def anchored_ridge_prediction(
    known_features: np.ndarray,
    known_tvt: np.ndarray,
    ps_features: np.ndarray,
    prediction_features: np.ndarray,
    tvt_at_ps: float,
    fallback_prediction: np.ndarray,
) -> np.ndarray:
    """
    Fit only on known pre-PS rows.

    Then force continuity at PS:

    final prediction =
    TVT_at_PS + (ridge prediction at row - ridge prediction at PS)
    """
    model = fit_ridge_model(
        known_features=known_features,
        known_target=known_tvt,
        alpha=RIDGE_ALPHA,
    )

    if model is None:
        return fallback_prediction.copy()

    ps_prediction = predict_ridge_model(
        model=model,
        features=ps_features.reshape(1, -1),
    )[0]

    post_prediction = predict_ridge_model(
        model=model,
        features=prediction_features,
    )

    anchored_prediction = (
        tvt_at_ps + (post_prediction - ps_prediction)
    )

    invalid = ~np.isfinite(anchored_prediction)

    anchored_prediction[invalid] = fallback_prediction[invalid]

    return anchored_prediction


def calculate_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    residual = prediction - actual

    return {
        "RMSE": rmse(actual, prediction),
        "MAE": mae(actual, prediction),
        "median_absolute_error": float(
            np.median(np.abs(residual))
        ),
        "mean_bias_predicted_minus_actual": float(
            residual.mean()
        ),
    }


def evaluate_one_well(
    well_id: str,
    manifest_row: pd.Series,
) -> pd.DataFrame:
    horizontal_path = find_horizontal_file(well_id)
    df = pd.read_csv(horizontal_path)

    required_columns = [
        "MD",
        "X",
        "Y",
        "Z",
        "TVT_input",
        "TVT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{horizontal_path.name} is missing: {missing_columns}"
        )

    ps_row = int(manifest_row["ps_row"])
    prediction_start_row = int(
        manifest_row["prediction_start_row"]
    )

    md = pd.to_numeric(df["MD"], errors="coerce").to_numpy(
        dtype=float
    )

    x = pd.to_numeric(df["X"], errors="coerce").to_numpy(
        dtype=float
    )

    y = pd.to_numeric(df["Y"], errors="coerce").to_numpy(
        dtype=float
    )

    z = pd.to_numeric(df["Z"], errors="coerce").to_numpy(
        dtype=float
    )

    tvt_input = pd.to_numeric(
        df["TVT_input"],
        errors="coerce",
    ).to_numpy(dtype=float)

    actual_tvt = pd.to_numeric(
        df["TVT"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(tvt_input[ps_row]):
        raise ValueError(
            f"{well_id}: TVT_input at PS is missing."
        )

    tvt_at_ps = float(tvt_input[ps_row])

    # Known section: rows up to and including PS.
    known_slice = slice(0, ps_row + 1)

    # Prediction section: rows after PS.
    post_slice = slice(prediction_start_row, len(df))

    known_md = md[known_slice]
    known_x = x[known_slice]
    known_y = y[known_slice]
    known_z = z[known_slice]
    known_tvt = tvt_input[known_slice]

    post_md = md[post_slice]
    post_x = x[post_slice]
    post_y = y[post_slice]
    post_z = z[post_slice]
    post_actual_tvt = actual_tvt[post_slice]

    if not np.isfinite(post_actual_tvt).all():
        raise ValueError(
            f"{well_id}: training TVT is missing after PS."
        )

    # --------------------------------------------------------
    # 1. MD-slope baseline: same concept as current Gold model.
    # --------------------------------------------------------
    slope_start = max(0, ps_row - SLOPE_WINDOW + 1)

    md_slope = safe_slope(
        md[slope_start:ps_row + 1],
        tvt_input[slope_start:ps_row + 1],
    )

    md_slope_prediction = (
        tvt_at_ps + md_slope * (post_md - md[ps_row])
    )

    # --------------------------------------------------------
    # 2. Z-slope baseline.
    # --------------------------------------------------------
    z_slope = safe_slope(
        z[slope_start:ps_row + 1],
        tvt_input[slope_start:ps_row + 1],
    )

    z_slope_prediction = (
        tvt_at_ps + z_slope * (post_z - z[ps_row])
    )

    # --------------------------------------------------------
    # 3. Ridge baseline: MD + Z
    # --------------------------------------------------------
    known_md_z = np.column_stack([known_md, known_z])
    ps_md_z = np.array([md[ps_row], z[ps_row]])
    post_md_z = np.column_stack([post_md, post_z])

    md_z_ridge_prediction = anchored_ridge_prediction(
        known_features=known_md_z,
        known_tvt=known_tvt,
        ps_features=ps_md_z,
        prediction_features=post_md_z,
        tvt_at_ps=tvt_at_ps,
        fallback_prediction=md_slope_prediction,
    )

    # --------------------------------------------------------
    # 4. Ridge baseline: MD + X + Y + Z
    # --------------------------------------------------------
    known_trajectory = np.column_stack(
        [known_md, known_x, known_y, known_z]
    )

    ps_trajectory = np.array(
        [md[ps_row], x[ps_row], y[ps_row], z[ps_row]]
    )

    post_trajectory = np.column_stack(
        [post_md, post_x, post_y, post_z]
    )

    trajectory_ridge_prediction = anchored_ridge_prediction(
        known_features=known_trajectory,
        known_tvt=known_tvt,
        ps_features=ps_trajectory,
        prediction_features=post_trajectory,
        tvt_at_ps=tvt_at_ps,
        fallback_prediction=md_slope_prediction,
    )

    return pd.DataFrame(
        {
            "well_id": well_id,
            "source_row_index": np.arange(
                prediction_start_row,
                len(df),
            ),
            "actual_TVT": post_actual_tvt,
            "TVT_at_PS": tvt_at_ps,
            "MD_since_PS": post_md - md[ps_row],
            "Z_since_PS": post_z - z[ps_row],
            "MD_slope_baseline_TVT": md_slope_prediction,
            "Z_slope_baseline_TVT": z_slope_prediction,
            "MD_Z_ridge_baseline_TVT": md_z_ridge_prediction,
            "trajectory_ridge_baseline_TVT": (
                trajectory_ridge_prediction
            ),
        }
    )


# ============================================================
# Main
# ============================================================
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_ps_manifest()
    validation_wells = load_validation_wells()

    validation_manifest = manifest[
        (manifest["split"] == "train")
        & (manifest["well_id"].isin(validation_wells))
    ].copy()

    if validation_manifest.empty:
        raise ValueError(
            "No validation wells found in ps_manifest.csv."
        )

    print("=" * 70)
    print("Anchor Baseline Evaluation")
    print("=" * 70)
    print(f"Validation wells: {len(validation_manifest)}")
    print(f"Recent slope window: {SLOPE_WINDOW}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    prediction_blocks = []

    for position, (_, manifest_row) in enumerate(
        validation_manifest.iterrows(),
        start=1,
    ):
        well_id = manifest_row["well_id"]

        well_predictions = evaluate_one_well(
            well_id=well_id,
            manifest_row=manifest_row,
        )

        prediction_blocks.append(well_predictions)

        print(
            f"Completed {position:>3}/"
            f"{len(validation_manifest)} | "
            f"{well_id} | "
            f"{len(well_predictions):,} rows"
        )

    predictions = pd.concat(
        prediction_blocks,
        ignore_index=True,
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    actual = predictions["actual_TVT"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------
    result_rows = []

    for baseline_column in BASELINE_COLUMNS:
        prediction = predictions[
            baseline_column
        ].to_numpy(dtype=float)

        result_rows.append(
            {
                "baseline": baseline_column,
                **calculate_metrics(actual, prediction),
            }
        )

    results_df = pd.DataFrame(result_rows).sort_values(
        by="RMSE",
        ascending=True,
    )

    results_df.to_csv(RESULTS_PATH, index=False)

    # --------------------------------------------------------
    # Results by well
    # --------------------------------------------------------
    by_well_records = []

    for well_id, well_df in predictions.groupby(
        "well_id",
        observed=False,
    ):
        actual_well = well_df["actual_TVT"].to_numpy(
            dtype=float
        )

        row = {
            "well_id": well_id,
            "rows": len(well_df),
        }

        for baseline_column in BASELINE_COLUMNS:
            predicted_well = well_df[
                baseline_column
            ].to_numpy(dtype=float)

            row[
                baseline_column.replace("_TVT", "_RMSE")
            ] = rmse(actual_well, predicted_well)

        by_well_records.append(row)

    by_well_df = pd.DataFrame(by_well_records)

    by_well_df.to_csv(BY_WELL_PATH, index=False)

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------
    best_row = results_df.iloc[0]

    report = {
        "validation_wells": int(
            predictions["well_id"].nunique()
        ),
        "validation_rows": int(len(predictions)),
        "slope_window": SLOPE_WINDOW,
        "ridge_alpha": RIDGE_ALPHA,
        "best_baseline": str(best_row["baseline"]),
        "best_baseline_RMSE": float(best_row["RMSE"]),
        "results": results_df.to_dict(orient="records"),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Overall Anchor Baseline Results")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved files")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Results:     {RESULTS_PATH}")
    print(f"By well:     {BY_WELL_PATH}")
    print(f"Report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()