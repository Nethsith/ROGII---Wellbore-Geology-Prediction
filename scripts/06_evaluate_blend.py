from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# Paths and settings
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

XGB_PREDICTIONS_PATH = REPORT_DIR / "xgboost_validation_predictions.csv"
CATBOOST_PREDICTIONS_PATH = REPORT_DIR / "catboost_validation_predictions.csv"

WEIGHT_RESULTS_PATH = REPORT_DIR / "blend_weight_results.csv"
BEST_BLEND_PATH = REPORT_DIR / "best_blend_validation_predictions.csv"
BLEND_REPORT_PATH = REPORT_DIR / "blend_training_report.json"

# CatBoost weight tested from 0.00 to 1.00 in steps of 0.01.
CATBOOST_WEIGHTS = np.round(np.arange(0.00, 1.01, 0.01), 2)


# ============================================================
# Helpers
# ============================================================
def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def load_prediction_file(file_path: Path, model_name: str) -> pd.DataFrame:
    """Load and rename one model's validation predictions."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"{model_name} validation predictions not found:\n{file_path}"
        )

    required_columns = [
        "well_id",
        "source_row_index",
        "TVT_at_PS",
        "actual_TVT_change_from_PS",
        "predicted_TVT_change_from_PS",
        "actual_TVT",
        "predicted_TVT",
    ]

    df = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{model_name} prediction file is missing columns: "
            f"{missing_columns}"
        )

    key_columns = ["well_id", "source_row_index"]

    rename_map = {
        column: f"{model_name}_{column}"
        for column in df.columns
        if column not in key_columns
    }

    return df.rename(columns=rename_map)


def validate_shared_values(merged: pd.DataFrame) -> None:
    """
    Ensure both models were evaluated on exactly the same target rows
    and use the same TVT anchor values.
    """
    checks = [
        (
            "TVT_at_PS",
            "xgb_TVT_at_PS",
            "catboost_TVT_at_PS",
        ),
        (
            "actual_TVT_change_from_PS",
            "xgb_actual_TVT_change_from_PS",
            "catboost_actual_TVT_change_from_PS",
        ),
        (
            "actual_TVT",
            "xgb_actual_TVT",
            "catboost_actual_TVT",
        ),
    ]

    for label, xgb_column, catboost_column in checks:
        xgb_values = merged[xgb_column].to_numpy(dtype=float)
        catboost_values = merged[catboost_column].to_numpy(dtype=float)

        is_equal = np.allclose(
            xgb_values,
            catboost_values,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

        if not is_equal:
            maximum_difference = np.nanmax(
                np.abs(xgb_values - catboost_values)
            )

            raise ValueError(
                f"Mismatch found in {label} between XGBoost and CatBoost.\n"
                f"Maximum difference: {maximum_difference}"
            )


# ============================================================
# Main
# ============================================================
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("XGBoost + CatBoost Blend Evaluation")
    print("=" * 70)

    xgb_df = load_prediction_file(
        file_path=XGB_PREDICTIONS_PATH,
        model_name="xgb",
    )

    catboost_df = load_prediction_file(
        file_path=CATBOOST_PREDICTIONS_PATH,
        model_name="catboost",
    )

    key_columns = ["well_id", "source_row_index"]

    # Outer merge ensures no validation rows are silently lost.
    merged = xgb_df.merge(
        catboost_df,
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_rows = merged[merged["_merge"] != "both"].copy()

    if not unmatched_rows.empty:
        unmatched_path = REPORT_DIR / "blend_unmatched_validation_rows.csv"

        unmatched_rows.to_csv(unmatched_path, index=False)

        raise ValueError(
            "XGBoost and CatBoost validation rows do not match.\n"
            f"Unmatched rows: {len(unmatched_rows):,}\n"
            f"Details saved to: {unmatched_path}"
        )

    merged = merged.drop(columns="_merge")

    validate_shared_values(merged)

    actual_tvt = merged["xgb_actual_TVT"].to_numpy(dtype=float)
    tvt_at_ps = merged["xgb_TVT_at_PS"].to_numpy(dtype=float)

    xgb_prediction = merged["xgb_predicted_TVT"].to_numpy(dtype=float)
    catboost_prediction = merged[
        "catboost_predicted_TVT"
    ].to_numpy(dtype=float)

    xgb_rmse = rmse(actual_tvt, xgb_prediction)
    catboost_rmse = rmse(actual_tvt, catboost_prediction)

    print(f"Matched validation rows: {len(merged):,}")
    print(f"XGBoost RMSE:            {xgb_rmse:.6f}")
    print(f"CatBoost RMSE:           {catboost_rmse:.6f}")

    # --------------------------------------------------------
    # Test blend weights
    # --------------------------------------------------------
    results = []

    for catboost_weight in CATBOOST_WEIGHTS:
        xgb_weight = 1.0 - catboost_weight

        blended_prediction = (
            xgb_weight * xgb_prediction
            + catboost_weight * catboost_prediction
        )

        blend_rmse = rmse(actual_tvt, blended_prediction)

        results.append(
            {
                "xgb_weight": xgb_weight,
                "catboost_weight": catboost_weight,
                "validation_RMSE": blend_rmse,
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="validation_RMSE",
        ascending=True,
    )

    results_df.to_csv(WEIGHT_RESULTS_PATH, index=False)

    best_result = results_df.iloc[0]

    best_xgb_weight = float(best_result["xgb_weight"])
    best_catboost_weight = float(best_result["catboost_weight"])
    best_rmse = float(best_result["validation_RMSE"])

    best_prediction = (
        best_xgb_weight * xgb_prediction
        + best_catboost_weight * catboost_prediction
    )

    best_prediction_change = best_prediction - tvt_at_ps

    # --------------------------------------------------------
    # Save best blend predictions
    # --------------------------------------------------------
    best_blend_df = pd.DataFrame(
        {
            "well_id": merged["well_id"],
            "source_row_index": merged["source_row_index"],
            "TVT_at_PS": tvt_at_ps,
            "actual_TVT": actual_tvt,
            "xgb_predicted_TVT": xgb_prediction,
            "catboost_predicted_TVT": catboost_prediction,
            "blended_predicted_TVT": best_prediction,
            "blended_predicted_TVT_change_from_PS": (
                best_prediction_change
            ),
            "absolute_error": np.abs(actual_tvt - best_prediction),
        }
    )

    best_blend_df.to_csv(BEST_BLEND_PATH, index=False)

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------
    report = {
        "validation_rows": int(len(merged)),
        "xgboost_validation_RMSE": xgb_rmse,
        "catboost_validation_RMSE": catboost_rmse,
        "best_xgb_weight": best_xgb_weight,
        "best_catboost_weight": best_catboost_weight,
        "best_blend_validation_RMSE": best_rmse,
        "blend_improvement_vs_xgboost": xgb_rmse - best_rmse,
        "blend_improvement_vs_catboost": catboost_rmse - best_rmse,
    }

    with open(BLEND_REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Best Blend Results")
    print("=" * 70)
    print(f"XGBoost weight:          {best_xgb_weight:.2f}")
    print(f"CatBoost weight:         {best_catboost_weight:.2f}")
    print(f"Best blend RMSE:         {best_rmse:.6f}")
    print(
        f"Improvement vs XGBoost:  "
        f"{xgb_rmse - best_rmse:.6f}"
    )
    print(
        f"Improvement vs CatBoost: "
        f"{catboost_rmse - best_rmse:.6f}"
    )

    print("\nTop 10 blend weights:")
    print(
        results_df.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSaved files")
    print(f"Weight results:          {WEIGHT_RESULTS_PATH}")
    print(f"Best blend predictions:  {BEST_BLEND_PATH}")
    print(f"Blend report:            {BLEND_REPORT_PATH}")


if __name__ == "__main__":
    main()