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
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
REPORT_DIR = GOLD_DIR / "_reports"

PS_MANIFEST_PATH = INTERIM_DIR / "ps_manifest.csv"

ROLLING_WINDOWS = [5, 21, 51]
SLOPE_WINDOW = 100

REQUIRED_HORIZONTAL_COLUMNS = ["MD", "X", "Y", "Z", "GR"]


# ============================================================
# Helpers
# ============================================================
def get_well_id(file_path: Path) -> str:
    """Example: 00bbac68__horizontal_well.csv -> 00bbac68"""
    return file_path.name.split("__", 1)[0]


def find_horizontal_file(split: str, well_id: str) -> Path:
    split_dir = SILVER_DIR / split
    matches = sorted(split_dir.glob(f"{well_id}__horizontal_well.csv"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one horizontal well file for {split}/{well_id}. "
            f"Found: {matches}"
        )

    return matches[0]


def find_typewell_file(split: str, well_id: str) -> Path:
    split_dir = SILVER_DIR / split
    matches = sorted(split_dir.glob(f"{well_id}__typewell*.csv"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one typewell file for {split}/{well_id}. "
            f"Found: {matches}"
        )

    return matches[0]


def safe_slope(md: pd.Series, values: pd.Series) -> float:
    """
    Calculates a simple TVT change-per-MD slope.
    Uses only known rows before PS.
    """
    valid = pd.DataFrame({"MD": md, "value": values}).dropna()

    if len(valid) < 2:
        return 0.0

    x = valid["MD"].to_numpy(dtype=float)
    y = valid["value"].to_numpy(dtype=float)

    x_centered = x - x.mean()
    denominator = np.sum(x_centered ** 2)

    if denominator == 0:
        return 0.0

    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def add_gr_features(features: pd.DataFrame, gr: pd.Series) -> None:
    """
    Add GR features without filling missing GR values.
    XGBoost can handle NaN values directly.
    """
    features["GR"] = gr
    features["GR_missing"] = gr.isna().astype("int8")

    features["GR_diff_1"] = gr.diff(1)
    features["GR_diff_5"] = gr.diff(5)

    observed_gr = gr.dropna()

    if len(observed_gr) > 1 and observed_gr.std(ddof=0) > 0:
        features["GR_zscore_well"] = (
            gr - observed_gr.mean()
        ) / observed_gr.std(ddof=0)
    else:
        features["GR_zscore_well"] = np.nan

    gr_iqr = observed_gr.quantile(0.75) - observed_gr.quantile(0.25)

    if len(observed_gr) > 1 and gr_iqr > 0:
        features["GR_robust_well"] = (
            gr - observed_gr.median()
        ) / gr_iqr
    else:
        features["GR_robust_well"] = np.nan

    for window in ROLLING_WINDOWS:
        rolling_gr = gr.rolling(
            window=window,
            center=True,
            min_periods=1,
        )

        features[f"GR_roll_mean_{window}"] = rolling_gr.mean()
        features[f"GR_roll_std_{window}"] = rolling_gr.std()
        features[f"GR_roll_min_{window}"] = rolling_gr.min()
        features[f"GR_roll_max_{window}"] = rolling_gr.max()

        features[f"GR_coverage_{window}"] = (
            gr.notna()
            .astype("float32")
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )


def get_typewell_summary(typewell_path: Path) -> dict:
    """
    Uses only TVT and GR because Geology is unavailable in test typewells.
    """
    typewell_df = pd.read_csv(typewell_path)

    for column in ["TVT", "GR"]:
        if column not in typewell_df.columns:
            raise ValueError(
                f"{typewell_path.name} does not contain required column: {column}"
            )

    typewell_tvt = pd.to_numeric(typewell_df["TVT"], errors="raise")
    typewell_gr = pd.to_numeric(typewell_df["GR"], errors="raise")

    return {
        "typewell_rows": len(typewell_df),
        "typewell_GR_missing_fraction": float(typewell_gr.isna().mean()),
        "typewell_TVT_min": typewell_tvt.min(),
        "typewell_TVT_max": typewell_tvt.max(),
        "typewell_TVT_mean": typewell_tvt.mean(),
        "typewell_TVT_std": typewell_tvt.std(),
        "typewell_GR_min": typewell_gr.min(),
        "typewell_GR_max": typewell_gr.max(),
        "typewell_GR_mean": typewell_gr.mean(),
        "typewell_GR_std": typewell_gr.std(),
        "typewell_GR_median": typewell_gr.median(),
    }


def get_known_tvt_context(
    horizontal_df: pd.DataFrame,
    split: str,
    ps_row: int,
) -> pd.Series:
    """
    Gets TVT values allowed before PS.

    Preferred source: TVT_input
    Fallback for artificial future PS scenarios in train: TVT up to PS only.
    """
    if "TVT_input" in horizontal_df.columns:
        known_tvt = pd.to_numeric(
            horizontal_df["TVT_input"],
            errors="raise",
        ).copy()

        # Safety: make every value after PS unavailable.
        known_tvt.iloc[ps_row + 1:] = np.nan
        return known_tvt

    if split == "train" and "TVT" in horizontal_df.columns:
        known_tvt = pd.to_numeric(
            horizontal_df["TVT"],
            errors="raise",
        ).copy()

        # This fallback is safe because values after PS are hidden.
        known_tvt.iloc[ps_row + 1:] = np.nan
        return known_tvt

    raise ValueError(
        "Cannot create TVT anchor features. "
        "TVT_input is missing and no safe training TVT fallback is available."
    )


def build_gold_for_scenario(
    manifest_row: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Builds Gold features for one well and one Prediction Start scenario.

    Output contains only rows after PS:
    - Train: rows with target TVT
    - Test: rows requiring predictions
    """
    split = manifest_row["split"]
    well_id = manifest_row["well_id"]
    scenario_id = manifest_row["scenario_id"]

    ps_row = int(manifest_row["ps_row"])
    prediction_start_row = int(manifest_row["prediction_start_row"])

    horizontal_path = find_horizontal_file(split, well_id)
    typewell_path = find_typewell_file(split, well_id)

    horizontal_df = pd.read_csv(horizontal_path)

    missing_columns = [
        column
        for column in REQUIRED_HORIZONTAL_COLUMNS
        if column not in horizontal_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{horizontal_path.name} is missing: {missing_columns}"
        )

    if prediction_start_row <= 0:
        raise ValueError(
            f"{scenario_id}: Prediction Start must be after at least one known row."
        )

    if prediction_start_row >= len(horizontal_df):
        raise ValueError(
            f"{scenario_id}: No rows exist after Prediction Start."
        )

    # --------------------------------------------------------
    # Original input values
    # --------------------------------------------------------
    md = pd.to_numeric(horizontal_df["MD"], errors="raise")
    x = pd.to_numeric(horizontal_df["X"], errors="raise")
    y = pd.to_numeric(horizontal_df["Y"], errors="raise")
    z = pd.to_numeric(horizontal_df["Z"], errors="raise")
    gr = pd.to_numeric(horizontal_df["GR"], errors="raise")

    known_tvt = get_known_tvt_context(
        horizontal_df=horizontal_df,
        split=split,
        ps_row=ps_row,
    )

    if pd.isna(known_tvt.iloc[ps_row]):
        raise ValueError(
            f"{scenario_id}: TVT at PS is missing, so no anchor can be created."
        )

    # --------------------------------------------------------
    # PS anchor values: safe information known before prediction
    # --------------------------------------------------------
    tvt_at_ps = float(known_tvt.iloc[ps_row])
    md_at_ps = float(md.iloc[ps_row])
    x_at_ps = float(x.iloc[ps_row])
    y_at_ps = float(y.iloc[ps_row])
    z_at_ps = float(z.iloc[ps_row])
    gr_at_ps = gr.iloc[ps_row]

    context_start = max(0, ps_row - SLOPE_WINDOW + 1)

    recent_md = md.iloc[context_start:ps_row + 1]
    recent_tvt = known_tvt.iloc[context_start:ps_row + 1]

    tvt_slope_recent = safe_slope(recent_md, recent_tvt)
    tvt_slope_all_known = safe_slope(
        md.iloc[:ps_row + 1],
        known_tvt.iloc[:ps_row + 1],
    )

    recent_tvt_valid = recent_tvt.dropna()

    tvt_recent_std = (
        float(recent_tvt_valid.std())
        if len(recent_tvt_valid) > 1
        else 0.0
    )

    tvt_recent_range = (
        float(recent_tvt_valid.max() - recent_tvt_valid.min())
        if len(recent_tvt_valid) > 0
        else 0.0
    )

    typewell_summary = get_typewell_summary(typewell_path)

    # --------------------------------------------------------
    # Base Gold features for every original row
    # --------------------------------------------------------
    features = pd.DataFrame(index=horizontal_df.index)

    features["scenario_id"] = scenario_id
    features["well_id"] = well_id
    features["source_split"] = split
    features["typewell_file"] = typewell_path.name
    features["source_row_index"] = np.arange(len(horizontal_df))

    # Original values
    features["MD"] = md
    features["X"] = x
    features["Y"] = y
    features["Z"] = z

    # Position inside the well
    md_start = float(md.iloc[0])
    md_range = float(md.iloc[-1] - md.iloc[0])

    features["row_fraction"] = (
        np.arange(len(horizontal_df)) / max(len(horizontal_df) - 1, 1)
    )

    features["MD_from_start"] = md - md_start

    if md_range != 0:
        features["MD_fraction"] = (md - md_start) / md_range
    else:
        features["MD_fraction"] = 0.0

    features["X_from_start"] = x - x.iloc[0]
    features["Y_from_start"] = y - y.iloc[0]
    features["Z_from_start"] = z - z.iloc[0]

    features["horizontal_offset_from_start"] = np.hypot(
        features["X_from_start"],
        features["Y_from_start"],
    )

    # One-foot / row-to-row movement
    delta_md = md.diff().replace(0, np.nan)
    delta_x = x.diff()
    delta_y = y.diff()
    delta_z = z.diff()

    horizontal_step = np.hypot(delta_x, delta_y)
    spatial_step = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    features["delta_MD"] = delta_md
    features["delta_X"] = delta_x
    features["delta_Y"] = delta_y
    features["delta_Z"] = delta_z
    features["horizontal_step"] = horizontal_step
    features["spatial_step"] = spatial_step
    features["vertical_change_per_MD"] = delta_z / delta_md
    features["horizontal_change_per_MD"] = horizontal_step / delta_md

    azimuth_rad = np.arctan2(delta_x, delta_y)

    features["azimuth_sin"] = np.sin(azimuth_rad)
    features["azimuth_cos"] = np.cos(azimuth_rad)

    # GR features
    add_gr_features(features, gr)

    # --------------------------------------------------------
    # Prediction Start features
    # --------------------------------------------------------
    features["MD_since_PS"] = md - md_at_ps
    features["X_since_PS"] = x - x_at_ps
    features["Y_since_PS"] = y - y_at_ps
    features["Z_since_PS"] = z - z_at_ps

    features["horizontal_distance_since_PS"] = np.hypot(
        features["X_since_PS"],
        features["Y_since_PS"],
    )

    features["spatial_distance_since_PS"] = np.sqrt(
        features["X_since_PS"] ** 2
        + features["Y_since_PS"] ** 2
        + features["Z_since_PS"] ** 2
    )

    features["GR_at_PS"] = gr_at_ps
    features["GR_change_since_PS"] = gr - gr_at_ps

    features["TVT_at_PS"] = tvt_at_ps
    features["TVT_slope_recent_before_PS"] = tvt_slope_recent
    features["TVT_slope_all_before_PS"] = tvt_slope_all_known
    features["TVT_recent_std_before_PS"] = tvt_recent_std
    features["TVT_recent_range_before_PS"] = tvt_recent_range

    # A simple safe baseline based on the TVT trend before PS.
    features["TVT_linear_baseline_from_PS"] = (
        tvt_at_ps
        + tvt_slope_recent * features["MD_since_PS"]
    )

    features["PS_row_fraction"] = ps_row / max(len(horizontal_df) - 1, 1)

    # Typewell numerical summaries
    for column, value in typewell_summary.items():
        features[column] = value

    # --------------------------------------------------------
    # Keep only rows after PS for model training/prediction
    # --------------------------------------------------------
    gold_df = features.iloc[prediction_start_row:].copy()

    gold_df["is_prediction_zone"] = 1

    # Train-only target columns
    if split == "train":
        if "TVT" not in horizontal_df.columns:
            raise ValueError(
                f"{horizontal_path.name} has no TVT target column."
            )

        target_tvt = pd.to_numeric(
            horizontal_df["TVT"],
            errors="raise",
        ).iloc[prediction_start_row:].reset_index(drop=True)

        gold_df = gold_df.reset_index(drop=True)

        if target_tvt.isna().any():
            raise ValueError(
                f"{scenario_id}: TVT target has missing values after PS."
            )

        gold_df["TVT"] = target_tvt
        gold_df["target_TVT_change_from_PS"] = (
            gold_df["TVT"] - tvt_at_ps
        )

    else:
        gold_df = gold_df.reset_index(drop=True)

    report = {
        "scenario_id": scenario_id,
        "split": split,
        "well_id": well_id,
        "source_rows": len(horizontal_df),
        "ps_row": ps_row,
        "prediction_start_row": prediction_start_row,
        "gold_rows": len(gold_df),
        "typewell_file": typewell_path.name,
        "gold_file": "",
    }

    return gold_df, report


# ============================================================
# Main
# ============================================================
def main():
    if not SILVER_DIR.exists():
        raise FileNotFoundError(
            f"Silver folder not found: {SILVER_DIR}\n"
            "Run 01_build_silver.py first."
        )

    if not PS_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"PS manifest not found: {PS_MANIFEST_PATH}\n"
            "Run 02_create_ps_manifest.py first."
        )

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ps_manifest = pd.read_csv(PS_MANIFEST_PATH)

    required_manifest_columns = [
        "scenario_id",
        "split",
        "well_id",
        "ps_row",
        "prediction_start_row",
    ]

    missing_manifest_columns = [
        column
        for column in required_manifest_columns
        if column not in ps_manifest.columns
    ]

    if missing_manifest_columns:
        raise ValueError(
            "PS manifest is missing columns: "
            f"{missing_manifest_columns}"
        )

    if ps_manifest.empty:
        raise ValueError("PS manifest is empty.")

    reports = []
    train_feature_columns = None
    test_feature_columns = None

    for _, manifest_row in ps_manifest.iterrows():
        scenario_id = manifest_row["scenario_id"]
        split = manifest_row["split"]

        if split not in {"train", "test"}:
            raise ValueError(
                f"{scenario_id}: split must be train or test, got {split}"
            )

        gold_df, report = build_gold_for_scenario(manifest_row)

        output_dir = GOLD_DIR / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{scenario_id}__gold_features.csv"
        gold_df.to_csv(output_path, index=False)

        report["gold_file"] = str(output_path.relative_to(PROJECT_ROOT))
        reports.append(report)

        non_feature_columns = {
            "scenario_id",
            "well_id",
            "source_split",
            "typewell_file",
            "source_row_index",
            "is_prediction_zone",
            "TVT",
            "target_TVT_change_from_PS",
        }

        feature_columns = [
            column
            for column in gold_df.columns
            if column not in non_feature_columns
        ]

        if split == "train":
            if train_feature_columns is None:
                train_feature_columns = feature_columns
            elif train_feature_columns != feature_columns:
                raise ValueError(
                    f"Feature mismatch in training scenario: {scenario_id}"
                )

        if split == "test":
            if test_feature_columns is None:
                test_feature_columns = feature_columns
            elif test_feature_columns != feature_columns:
                raise ValueError(
                    f"Feature mismatch in test scenario: {scenario_id}"
                )

        print(
            f"Saved {split}: {output_path.name} "
            f"({len(gold_df):,} prediction rows)"
        )

    if train_feature_columns is None:
        raise ValueError(
            "No training scenarios found in ps_manifest.csv."
        )

    if test_feature_columns is None:
        raise ValueError(
            "No test scenarios found in ps_manifest.csv."
        )

    if train_feature_columns != test_feature_columns:
        train_only = sorted(
            set(train_feature_columns) - set(test_feature_columns)
        )
        test_only = sorted(
            set(test_feature_columns) - set(train_feature_columns)
        )

        raise ValueError(
            "Train and test feature columns do not match.\n"
            f"Train-only: {train_only}\n"
            f"Test-only: {test_only}"
        )

    report_df = pd.DataFrame(reports)

    report_path = REPORT_DIR / "gold_build_report.csv"
    report_df.to_csv(report_path, index=False)

    feature_config = {
        "target_column": "TVT",
        "alternative_target_column": "target_TVT_change_from_PS",
        "feature_columns": train_feature_columns,
        "excluded_columns": [
            "scenario_id",
            "well_id",
            "source_split",
            "typewell_file",
            "source_row_index",
            "is_prediction_zone",
            "TVT",
            "target_TVT_change_from_PS",
        ],
    }

    feature_path = GOLD_DIR / "xgboost_feature_columns.json"

    with open(feature_path, "w", encoding="utf-8") as file:
        json.dump(feature_config, file, indent=2)

    train_rows = int(
        report_df.loc[report_df["split"] == "train", "gold_rows"].sum()
    )

    test_rows = int(
        report_df.loc[report_df["split"] == "test", "gold_rows"].sum()
    )

    print("\n" + "=" * 70)
    print("Gold dataset created successfully.")
    print(f"Train prediction rows: {train_rows:,}")
    print(f"Test prediction rows:  {test_rows:,}")
    print(f"Feature count:          {len(train_feature_columns)}")
    print(f"Gold build report:      {report_path}")
    print(f"Feature list:           {feature_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()