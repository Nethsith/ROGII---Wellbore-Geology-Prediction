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

BASE_GOLD_DIR = PROJECT_ROOT / "data" / "gold"
GOLD_V2_DIR = PROJECT_ROOT / "data" / "gold_v2"
REPORT_DIR = GOLD_V2_DIR / "_reports"

PS_MANIFEST_PATH = INTERIM_DIR / "ps_manifest.csv"
BASE_FEATURE_CONFIG_PATH = (
    BASE_GOLD_DIR / "xgboost_feature_columns.json"
)

V2_FEATURE_CONFIG_PATH = (
    GOLD_V2_DIR / "model_feature_columns.json"
)


# ============================================================
# Settings
# ============================================================
RIDGE_ALPHA = 1.0
MIN_RIDGE_ROWS = 10

NEW_FEATURE_COLUMNS = [
    "MD_Z_ridge_baseline_TVT",
    "trajectory_ridge_baseline_TVT",
    "MD_Z_ridge_baseline_change_from_PS",
    "trajectory_ridge_baseline_change_from_PS",
    "MD_Z_minus_linear_baseline",
    "trajectory_minus_linear_baseline",
    "MD_Z_ridge_used_fallback",
    "trajectory_ridge_used_fallback",
    "MD_Z_ridge_pre_PS_fit_RMSE",
    "trajectory_ridge_pre_PS_fit_RMSE",
    "known_TVT_rows_for_baseline",
]

TRAIN_ONLY_TARGET_COLUMNS = [
    "target_TVT_residual_from_MD_Z_baseline",
    "target_TVT_residual_from_trajectory_baseline",
]


# ============================================================
# Helpers
# ============================================================
def find_horizontal_file(split: str, well_id: str) -> Path:
    matches = sorted(
        (SILVER_DIR / split).glob(
            f"{well_id}__horizontal_well.csv"
        )
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one horizontal-well file for "
            f"{split}/{well_id}. Found: {matches}"
        )

    return matches[0]


def load_base_feature_columns() -> list[str]:
    if not BASE_FEATURE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Base Gold feature config not found:\n"
            f"{BASE_FEATURE_CONFIG_PATH}\n"
            "Run 03_build_gold.py first."
        )

    with open(BASE_FEATURE_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    feature_columns = config.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "No feature_columns found in base Gold feature config."
        )

    return feature_columns


def fit_ridge_model(
    features: np.ndarray,
    target: np.ndarray,
) -> dict | None:
    """
    Fit Ridge regression using only known pre-PS rows.

    Features are standardized using only those pre-PS rows.
    """
    valid = (
        np.isfinite(target)
        & np.isfinite(features).all(axis=1)
    )

    x = features[valid]
    y = target[valid]

    minimum_rows = max(MIN_RIDGE_ROWS, x.shape[1] + 2)

    if len(x) < minimum_rows:
        return None

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)

    # Prevent division by zero for constant features.
    feature_std[feature_std < 1e-12] = 1.0

    x_scaled = (x - feature_mean) / feature_std

    # First column is intercept.
    x_design = np.column_stack(
        [np.ones(len(x_scaled)), x_scaled]
    )

    penalty = np.eye(x_design.shape[1])
    penalty[0, 0] = 0.0  # Do not penalize intercept.

    try:
        coefficients = np.linalg.solve(
            x_design.T @ x_design + RIDGE_ALPHA * penalty,
            x_design.T @ y,
        )
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(
            x_design.T @ x_design + RIDGE_ALPHA * penalty
        ) @ (x_design.T @ y)

    fitted = x_design @ coefficients
    fit_rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "coefficients": coefficients,
        "fit_rmse": fit_rmse,
        "known_rows_used": int(len(x)),
    }


def predict_ridge_model(
    model: dict,
    features: np.ndarray,
) -> np.ndarray:
    """Predict while preserving invalid rows as NaN."""
    predictions = np.full(len(features), np.nan, dtype=float)

    valid = np.isfinite(features).all(axis=1)

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
    post_features: np.ndarray,
    tvt_at_ps: float,
    fallback_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Creates a ridge baseline that is forced to equal TVT_at_PS at PS.

    Final prediction:
    TVT_at_PS + [ridge(row) - ridge(PS)]

    Returns:
    prediction,
    fallback_indicator,
    pre_PS_fit_RMSE,
    known_rows_used
    """
    model = fit_ridge_model(
        features=known_features,
        target=known_tvt,
    )

    if model is None:
        return (
            fallback_prediction.copy(),
            np.ones(len(post_features), dtype="int8"),
            np.nan,
            0,
        )

    ps_prediction = predict_ridge_model(
        model=model,
        features=ps_features.reshape(1, -1),
    )[0]

    post_prediction = predict_ridge_model(
        model=model,
        features=post_features,
    )

    anchored_prediction = (
        tvt_at_ps + (post_prediction - ps_prediction)
    )

    used_fallback = ~np.isfinite(anchored_prediction)

    anchored_prediction[used_fallback] = (
        fallback_prediction[used_fallback]
    )

    return (
        anchored_prediction,
        used_fallback.astype("int8"),
        model["fit_rmse"],
        model["known_rows_used"],
    )


def get_known_tvt(
    horizontal_df: pd.DataFrame,
    split: str,
    ps_row: int,
) -> np.ndarray:
    """
    TVT is allowed only until PS.

    TVT after PS is never used as a feature.
    """
    if "TVT_input" in horizontal_df.columns:
        known_tvt = pd.to_numeric(
            horizontal_df["TVT_input"],
            errors="coerce",
        ).to_numpy(dtype=float, copy=True)

    elif split == "train" and "TVT" in horizontal_df.columns:
        # Safe fallback for training scenarios only.
        known_tvt = pd.to_numeric(
            horizontal_df["TVT"],
            errors="coerce",
        ).to_numpy(dtype=float, copy=True)

    else:
        raise ValueError(
            "Cannot build baseline: TVT_input is unavailable."
        )

    # Critical leakage protection.
    known_tvt[ps_row + 1:] = np.nan

    return known_tvt


def validate_base_gold(
    gold_df: pd.DataFrame,
    scenario_id: str,
) -> None:
    required_columns = [
        "source_row_index",
        "TVT_at_PS",
        "TVT_linear_baseline_from_PS",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in gold_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{scenario_id} base Gold file is missing: "
            f"{missing_columns}"
        )


def clean_old_v2_files() -> None:
    """Avoid duplicate Gold V2 files from older runs."""
    for split in ["train", "test"]:
        output_dir = GOLD_V2_DIR / split

        if output_dir.exists():
            for old_file in output_dir.glob(
                "*__gold_v2_features.csv"
            ):
                old_file.unlink()


# ============================================================
# Build one Gold V2 file
# ============================================================
def build_gold_v2_for_scenario(
    manifest_row: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    scenario_id = manifest_row["scenario_id"]
    split = manifest_row["split"]
    well_id = manifest_row["well_id"]

    ps_row = int(manifest_row["ps_row"])
    prediction_start_row = int(
        manifest_row["prediction_start_row"]
    )

    base_gold_path = (
        BASE_GOLD_DIR
        / split
        / f"{scenario_id}__gold_features.csv"
    )

    if not base_gold_path.exists():
        raise FileNotFoundError(
            f"Base Gold file not found:\n{base_gold_path}"
        )

    horizontal_path = find_horizontal_file(
        split=split,
        well_id=well_id,
    )

    gold_df = pd.read_csv(base_gold_path)
    horizontal_df = pd.read_csv(horizontal_path)

    validate_base_gold(
        gold_df=gold_df,
        scenario_id=scenario_id,
    )

    required_horizontal = ["MD", "X", "Y", "Z"]

    missing_horizontal = [
        column
        for column in required_horizontal
        if column not in horizontal_df.columns
    ]

    if missing_horizontal:
        raise ValueError(
            f"{horizontal_path.name} is missing: "
            f"{missing_horizontal}"
        )

    md = pd.to_numeric(
        horizontal_df["MD"],
        errors="coerce",
    ).to_numpy(dtype=float)

    x = pd.to_numeric(
        horizontal_df["X"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        horizontal_df["Y"],
        errors="coerce",
    ).to_numpy(dtype=float)

    z = pd.to_numeric(
        horizontal_df["Z"],
        errors="coerce",
    ).to_numpy(dtype=float)

    known_tvt = get_known_tvt(
        horizontal_df=horizontal_df,
        split=split,
        ps_row=ps_row,
    )

    if not np.isfinite(known_tvt[ps_row]):
        raise ValueError(
            f"{scenario_id}: TVT value at PS is missing."
        )

    tvt_at_ps = float(known_tvt[ps_row])

    source_rows = gold_df[
        "source_row_index"
    ].to_numpy(dtype=int)

    if len(source_rows) == 0:
        raise ValueError(
            f"{scenario_id}: base Gold file has no prediction rows."
        )

    if source_rows.min() < prediction_start_row:
        raise ValueError(
            f"{scenario_id}: base Gold contains rows before PS."
        )

    if source_rows.max() >= len(horizontal_df):
        raise ValueError(
            f"{scenario_id}: source row index exceeds horizontal file."
        )

    # Safety check: Gold anchor must match Silver anchor.
    gold_tvt_anchor = gold_df[
        "TVT_at_PS"
    ].to_numpy(dtype=float)

    if not np.allclose(
        gold_tvt_anchor,
        tvt_at_ps,
        equal_nan=False,
    ):
        raise ValueError(
            f"{scenario_id}: TVT_at_PS mismatch between "
            "base Gold and Silver data."
        )

    linear_baseline = gold_df[
        "TVT_linear_baseline_from_PS"
    ].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Known pre-PS data for Ridge fitting
    # --------------------------------------------------------
    known_slice = slice(0, ps_row + 1)

    known_md_z = np.column_stack(
        [
            md[known_slice],
            z[known_slice],
        ]
    )

    known_trajectory = np.column_stack(
        [
            md[known_slice],
            x[known_slice],
            y[known_slice],
            z[known_slice],
        ]
    )

    ps_md_z = np.array(
        [md[ps_row], z[ps_row]],
        dtype=float,
    )

    ps_trajectory = np.array(
        [
            md[ps_row],
            x[ps_row],
            y[ps_row],
            z[ps_row],
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Post-PS data, matching base Gold row order exactly
    # --------------------------------------------------------
    post_md_z = np.column_stack(
        [
            md[source_rows],
            z[source_rows],
        ]
    )

    post_trajectory = np.column_stack(
        [
            md[source_rows],
            x[source_rows],
            y[source_rows],
            z[source_rows],
        ]
    )

    (
        md_z_baseline,
        md_z_fallback,
        md_z_fit_rmse,
        md_z_known_rows,
    ) = anchored_ridge_prediction(
        known_features=known_md_z,
        known_tvt=known_tvt[known_slice],
        ps_features=ps_md_z,
        post_features=post_md_z,
        tvt_at_ps=tvt_at_ps,
        fallback_prediction=linear_baseline,
    )

    (
        trajectory_baseline,
        trajectory_fallback,
        trajectory_fit_rmse,
        trajectory_known_rows,
    ) = anchored_ridge_prediction(
        known_features=known_trajectory,
        known_tvt=known_tvt[known_slice],
        ps_features=ps_trajectory,
        post_features=post_trajectory,
        tvt_at_ps=tvt_at_ps,
        fallback_prediction=linear_baseline,
    )

    # --------------------------------------------------------
    # Add V2 features
    # --------------------------------------------------------
    gold_v2_df = gold_df.copy()

    gold_v2_df["MD_Z_ridge_baseline_TVT"] = md_z_baseline
    gold_v2_df["trajectory_ridge_baseline_TVT"] = (
        trajectory_baseline
    )

    gold_v2_df["MD_Z_ridge_baseline_change_from_PS"] = (
        md_z_baseline - tvt_at_ps
    )

    gold_v2_df[
        "trajectory_ridge_baseline_change_from_PS"
    ] = trajectory_baseline - tvt_at_ps

    gold_v2_df["MD_Z_minus_linear_baseline"] = (
        md_z_baseline - linear_baseline
    )

    gold_v2_df["trajectory_minus_linear_baseline"] = (
        trajectory_baseline - linear_baseline
    )

    gold_v2_df["MD_Z_ridge_used_fallback"] = md_z_fallback

    gold_v2_df["trajectory_ridge_used_fallback"] = (
        trajectory_fallback
    )

    gold_v2_df["MD_Z_ridge_pre_PS_fit_RMSE"] = md_z_fit_rmse

    gold_v2_df["trajectory_ridge_pre_PS_fit_RMSE"] = (
        trajectory_fit_rmse
    )

    gold_v2_df["known_TVT_rows_for_baseline"] = max(
        md_z_known_rows,
        trajectory_known_rows,
    )

    # Train-only residual targets.
    if split == "train":
        if "TVT" not in gold_v2_df.columns:
            raise ValueError(
                f"{scenario_id}: TVT target missing from training Gold."
            )

        true_tvt = gold_v2_df["TVT"].to_numpy(dtype=float)

        gold_v2_df[
            "target_TVT_residual_from_MD_Z_baseline"
        ] = true_tvt - md_z_baseline

        gold_v2_df[
            "target_TVT_residual_from_trajectory_baseline"
        ] = true_tvt - trajectory_baseline

    report = {
        "scenario_id": scenario_id,
        "split": split,
        "well_id": well_id,
        "rows": len(gold_v2_df),
        "known_rows_before_PS": ps_row + 1,
        "MD_Z_fallback_rows": int(md_z_fallback.sum()),
        "trajectory_fallback_rows": int(
            trajectory_fallback.sum()
        ),
        "MD_Z_pre_PS_fit_RMSE": md_z_fit_rmse,
        "trajectory_pre_PS_fit_RMSE": trajectory_fit_rmse,
    }

    return gold_v2_df, report


# ============================================================
# Main
# ============================================================
def main():
    if not PS_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"PS manifest not found:\n{PS_MANIFEST_PATH}\n"
            "Run 02_create_ps_manifest.py first."
        )

    if not BASE_GOLD_DIR.exists():
        raise FileNotFoundError(
            f"Base Gold folder not found:\n{BASE_GOLD_DIR}\n"
            "Run 03_build_gold.py first."
        )

    GOLD_V2_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    clean_old_v2_files()

    ps_manifest = pd.read_csv(PS_MANIFEST_PATH)

    required_manifest_columns = [
        "scenario_id",
        "split",
        "well_id",
        "ps_row",
        "prediction_start_row",
    ]

    missing_manifest = [
        column
        for column in required_manifest_columns
        if column not in ps_manifest.columns
    ]

    if missing_manifest:
        raise ValueError(
            f"PS manifest is missing columns: {missing_manifest}"
        )

    base_feature_columns = load_base_feature_columns()
    v2_feature_columns = (
        base_feature_columns + NEW_FEATURE_COLUMNS
    )

    reports = []

    print("=" * 70)
    print("Building Gold V2 Dataset")
    print("=" * 70)

    for position, (_, manifest_row) in enumerate(
        ps_manifest.iterrows(),
        start=1,
    ):
        scenario_id = manifest_row["scenario_id"]
        split = manifest_row["split"]

        gold_v2_df, report = build_gold_v2_for_scenario(
            manifest_row=manifest_row
        )

        output_dir = GOLD_V2_DIR / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir
            / f"{scenario_id}__gold_v2_features.csv"
        )

        gold_v2_df.to_csv(output_path, index=False)

        report["gold_v2_file"] = str(
            output_path.relative_to(PROJECT_ROOT)
        )

        reports.append(report)

        print(
            f"Saved {position:>3}/{len(ps_manifest)} | "
            f"{split} | {scenario_id} | "
            f"{len(gold_v2_df):,} rows"
        )

    report_df = pd.DataFrame(reports)

    report_path = REPORT_DIR / "gold_v2_build_report.csv"
    report_df.to_csv(report_path, index=False)

    feature_config = {
        "direct_target_column": "target_TVT_change_from_PS",
        "residual_target_column": (
            "target_TVT_residual_from_trajectory_baseline"
        ),
        "alternative_residual_target_column": (
            "target_TVT_residual_from_MD_Z_baseline"
        ),
        "feature_columns": v2_feature_columns,
        "base_feature_count": len(base_feature_columns),
        "new_feature_count": len(NEW_FEATURE_COLUMNS),
        "total_feature_count": len(v2_feature_columns),
        "excluded_columns": [
            "scenario_id",
            "well_id",
            "source_split",
            "typewell_file",
            "source_row_index",
            "is_prediction_zone",
            "TVT",
            "target_TVT_change_from_PS",
            "target_TVT_residual_from_MD_Z_baseline",
            "target_TVT_residual_from_trajectory_baseline",
        ],
    }

    with open(
        V2_FEATURE_CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(feature_config, file, indent=2)

    train_rows = int(
        report_df.loc[
            report_df["split"] == "train",
            "rows",
        ].sum()
    )

    test_rows = int(
        report_df.loc[
            report_df["split"] == "test",
            "rows",
        ].sum()
    )

    print("\n" + "=" * 70)
    print("Gold V2 Dataset Created Successfully")
    print("=" * 70)
    print(f"Train rows:      {train_rows:,}")
    print(f"Test rows:       {test_rows:,}")
    print(f"Feature count:   {len(v2_feature_columns)}")
    print(f"Build report:    {report_path}")
    print(f"Feature config:  {V2_FEATURE_CONFIG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()