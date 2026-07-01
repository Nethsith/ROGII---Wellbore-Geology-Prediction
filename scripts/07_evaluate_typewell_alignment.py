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

PREDICTIONS_PATH = REPORT_DIR / "typewell_alignment_sample_predictions.csv"
SUMMARY_PATH = REPORT_DIR / "typewell_alignment_well_summary.csv"
REPORT_PATH = REPORT_DIR / "typewell_alignment_report.json"


# ============================================================
# Alignment settings
# ============================================================
# Horizontal GR signature: 20 rows before + current + 20 rows after.
WINDOW_RADIUS_ROWS = 20
WINDOW_SIZE = WINDOW_RADIUS_ROWS * 2 + 1

# Need enough real GR readings inside the signature.
MIN_COMMON_SAMPLES = 12

# Search around the TVT extrapolation baseline.
CANDIDATE_HALF_WIDTH_TVT = 500.0
CANDIDATE_TVT_STEP = 2.0

# Small penalty prevents choosing a very distant candidate
# when two GR signatures look similarly good.
PRIOR_PENALTY = 0.10

# Evaluate sampled post-PS rows for all XGBoost validation wells.
ROWS_PER_WELL = 30

# TVT slope uses the last known rows before PS.
SLOPE_WINDOW = 100


# ============================================================
# Helpers
# ============================================================
def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def get_well_id(file_path: Path) -> str:
    return file_path.name.split("__", 1)[0]


def find_file(split: str, well_id: str, file_type: str) -> Path:
    folder = SILVER_DIR / split

    if file_type == "horizontal":
        matches = sorted(folder.glob(f"{well_id}__horizontal_well.csv"))
    elif file_type == "typewell":
        matches = sorted(folder.glob(f"{well_id}__typewell*.csv"))
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {file_type} file for {split}/{well_id}. "
            f"Found: {matches}"
        )

    return matches[0]


def safe_slope(md: np.ndarray, tvt: np.ndarray) -> float:
    valid = np.isfinite(md) & np.isfinite(tvt)

    if valid.sum() < 2:
        return 0.0

    x = md[valid]
    y = tvt[valid]

    x_centered = x - x.mean()
    denominator = np.sum(x_centered ** 2)

    if denominator == 0:
        return 0.0

    return float(
        np.sum(x_centered * (y - y.mean())) / denominator
    )


def load_xgb_validation_wells() -> list[str]:
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
            "validation_wells not found in xgboost_training_report.json"
        )

    return sorted(validation_wells)


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

    missing = [
        column for column in required_columns
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            f"PS manifest missing required columns: {missing}"
        )

    return manifest


def create_typewell_windows(
    typewell_gr: np.ndarray,
) -> np.ndarray:
    """
    Returns every rolling typewell GR window.

    Shape:
    [number_of_possible_centers, WINDOW_SIZE]
    """
    if len(typewell_gr) < WINDOW_SIZE:
        raise ValueError(
            "Typewell is shorter than the requested GR window size."
        )

    return np.lib.stride_tricks.sliding_window_view(
        typewell_gr,
        WINDOW_SIZE,
    )


def calculate_correlation_scores(
    horizontal_window: np.ndarray,
    typewell_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Pearson correlation for every candidate typewell window.

    Missing GR values remain missing. Only shared observed points are used.
    """
    horizontal_valid = np.isfinite(horizontal_window)

    typewell_valid = np.isfinite(typewell_windows)

    mask = typewell_valid & horizontal_valid[None, :]
    counts = mask.sum(axis=1)

    safe_horizontal = np.where(horizontal_valid, horizontal_window, 0.0)
    safe_typewell = np.where(typewell_valid, typewell_windows, 0.0)

    sum_h = (mask * safe_horizontal[None, :]).sum(axis=1)
    sum_t = (mask * safe_typewell).sum(axis=1)

    mean_h = np.divide(
        sum_h,
        counts,
        out=np.full_like(sum_h, np.nan, dtype=float),
        where=counts > 0,
    )

    mean_t = np.divide(
        sum_t,
        counts,
        out=np.full_like(sum_t, np.nan, dtype=float),
        where=counts > 0,
    )

    centered_h = safe_horizontal[None, :] - mean_h[:, None]
    centered_t = safe_typewell - mean_t[:, None]

    covariance = (mask * centered_h * centered_t).sum(axis=1)

    variance_h = (mask * centered_h**2).sum(axis=1)
    variance_t = (mask * centered_t**2).sum(axis=1)

    denominator = np.sqrt(variance_h * variance_t)

    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan, dtype=float),
        where=denominator > 1e-12,
    )

    correlation[counts < MIN_COMMON_SAMPLES] = np.nan

    return correlation, counts


def match_horizontal_row_to_typewell(
    horizontal_gr: np.ndarray,
    row_index: int,
    baseline_tvt: float,
    typewell_tvt: np.ndarray,
    typewell_windows: np.ndarray,
) -> dict:
    """
    Match one horizontal GR signature to candidate typewell signatures.

    No post-PS TVT is used here.
    """
    start = row_index - WINDOW_RADIUS_ROWS
    end = row_index + WINDOW_RADIUS_ROWS + 1

    horizontal_window = horizontal_gr[start:end]

    if len(horizontal_window) != WINDOW_SIZE:
        return {
            "matched_TVT": np.nan,
            "match_score": np.nan,
            "match_correlation": np.nan,
            "match_confidence": np.nan,
            "match_coverage": 0.0,
            "match_direction": "none",
        }

    if np.isfinite(horizontal_window).sum() < MIN_COMMON_SAMPLES:
        return {
            "matched_TVT": np.nan,
            "match_score": np.nan,
            "match_correlation": np.nan,
            "match_confidence": np.nan,
            "match_coverage": (
                np.isfinite(horizontal_window).sum() / WINDOW_SIZE
            ),
            "match_direction": "insufficient_GR",
        }

    candidate_tvt_values = np.arange(
        baseline_tvt - CANDIDATE_HALF_WIDTH_TVT,
        baseline_tvt + CANDIDATE_HALF_WIDTH_TVT + CANDIDATE_TVT_STEP,
        CANDIDATE_TVT_STEP,
    )

    candidate_centers = np.searchsorted(
        typewell_tvt,
        candidate_tvt_values,
    )

    candidate_centers = np.clip(
        candidate_centers,
        WINDOW_RADIUS_ROWS,
        len(typewell_tvt) - WINDOW_RADIUS_ROWS - 1,
    )

    candidate_centers = np.unique(candidate_centers)

    # In sliding windows, row 0 represents a window centered at WINDOW_RADIUS_ROWS.
    candidate_window_starts = candidate_centers - WINDOW_RADIUS_ROWS
    candidate_windows = typewell_windows[candidate_window_starts]

    forward_corr, forward_counts = calculate_correlation_scores(
        horizontal_window=horizontal_window,
        typewell_windows=candidate_windows,
    )

    reverse_corr, reverse_counts = calculate_correlation_scores(
        horizontal_window=horizontal_window,
        typewell_windows=candidate_windows[:, ::-1],
    )

    use_reverse = reverse_corr > forward_corr

    best_corr = np.where(use_reverse, reverse_corr, forward_corr)
    best_counts = np.where(use_reverse, reverse_counts, forward_counts)

    prior_distance = (
        (typewell_tvt[candidate_centers] - baseline_tvt)
        / CANDIDATE_HALF_WIDTH_TVT
    )

    score = 1.0 - best_corr + PRIOR_PENALTY * prior_distance**2
    score[~np.isfinite(best_corr)] = np.nan

    if np.all(~np.isfinite(score)):
        return {
            "matched_TVT": np.nan,
            "match_score": np.nan,
            "match_correlation": np.nan,
            "match_confidence": np.nan,
            "match_coverage": (
                np.isfinite(horizontal_window).sum() / WINDOW_SIZE
            ),
            "match_direction": "no_valid_match",
        }

    best_position = int(np.nanargmin(score))

    sorted_scores = np.sort(score[np.isfinite(score)])

    if len(sorted_scores) >= 2:
        confidence = float(sorted_scores[1] - sorted_scores[0])
    else:
        confidence = np.nan

    return {
        "matched_TVT": float(
            typewell_tvt[candidate_centers[best_position]]
        ),
        "match_score": float(score[best_position]),
        "match_correlation": float(best_corr[best_position]),
        "match_confidence": confidence,
        "match_coverage": float(
            best_counts[best_position] / WINDOW_SIZE
        ),
        "match_direction": (
            "reverse"
            if use_reverse[best_position]
            else "forward"
        ),
    }


def get_sample_rows(
    prediction_start_row: int,
    total_rows: int,
) -> np.ndarray:
    first_row = max(
        prediction_start_row,
        WINDOW_RADIUS_ROWS,
    )

    last_row = min(
        total_rows - WINDOW_RADIUS_ROWS - 1,
        total_rows - 1,
    )

    if first_row > last_row:
        return np.array([], dtype=int)

    count = min(
        ROWS_PER_WELL,
        last_row - first_row + 1,
    )

    sampled_rows = np.linspace(
        first_row,
        last_row,
        num=count,
        dtype=int,
    )

    return np.unique(sampled_rows)


def evaluate_one_well(
    well_id: str,
    manifest_row: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    horizontal_path = find_file(
        split="train",
        well_id=well_id,
        file_type="horizontal",
    )

    typewell_path = find_file(
        split="train",
        well_id=well_id,
        file_type="typewell",
    )

    horizontal_df = pd.read_csv(horizontal_path)
    typewell_df = pd.read_csv(typewell_path)

    required_horizontal = ["MD", "GR", "TVT_input", "TVT"]
    required_typewell = ["TVT", "GR"]

    for column in required_horizontal:
        if column not in horizontal_df.columns:
            raise ValueError(
                f"{horizontal_path.name} is missing column: {column}"
            )

    for column in required_typewell:
        if column not in typewell_df.columns:
            raise ValueError(
                f"{typewell_path.name} is missing column: {column}"
            )

    md = pd.to_numeric(
        horizontal_df["MD"],
        errors="coerce",
    ).to_numpy(dtype=float)

    horizontal_gr = pd.to_numeric(
        horizontal_df["GR"],
        errors="coerce",
    ).to_numpy(dtype=float)

    tvt_input = pd.to_numeric(
        horizontal_df["TVT_input"],
        errors="coerce",
    ).to_numpy(dtype=float)

    actual_tvt = pd.to_numeric(
        horizontal_df["TVT"],
        errors="coerce",
    ).to_numpy(dtype=float)

    typewell_tvt = pd.to_numeric(
        typewell_df["TVT"],
        errors="coerce",
    ).to_numpy(dtype=float)

    typewell_gr = pd.to_numeric(
        typewell_df["GR"],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid_typewell = (
        np.isfinite(typewell_tvt)
        & np.isfinite(typewell_gr)
    )

    typewell_tvt = typewell_tvt[valid_typewell]
    typewell_gr = typewell_gr[valid_typewell]

    sorting_order = np.argsort(typewell_tvt)

    typewell_tvt = typewell_tvt[sorting_order]
    typewell_gr = typewell_gr[sorting_order]

    if len(typewell_tvt) < WINDOW_SIZE:
        raise ValueError(
            f"{well_id}: typewell is too short for alignment."
        )

    ps_row = int(manifest_row["ps_row"])
    prediction_start_row = int(
        manifest_row["prediction_start_row"]
    )

    tvt_at_ps = tvt_input[ps_row]

    if not np.isfinite(tvt_at_ps):
        raise ValueError(f"{well_id}: TVT_input at PS is missing.")

    slope_start = max(0, ps_row - SLOPE_WINDOW + 1)

    recent_slope = safe_slope(
        md[slope_start:ps_row + 1],
        tvt_input[slope_start:ps_row + 1],
    )

    md_at_ps = md[ps_row]

    typewell_windows = create_typewell_windows(typewell_gr)

    sampled_rows = get_sample_rows(
        prediction_start_row=prediction_start_row,
        total_rows=len(horizontal_df),
    )

    records = []

    for row_index in sampled_rows:
        linear_baseline = (
            tvt_at_ps
            + recent_slope * (md[row_index] - md_at_ps)
        )

        match = match_horizontal_row_to_typewell(
            horizontal_gr=horizontal_gr,
            row_index=int(row_index),
            baseline_tvt=float(linear_baseline),
            typewell_tvt=typewell_tvt,
            typewell_windows=typewell_windows,
        )

        matched_tvt = match["matched_TVT"]

        # Fall back to the allowed pre-PS slope baseline
        # when no usable GR signature exists.
        alignment_prediction = (
            matched_tvt
            if np.isfinite(matched_tvt)
            else linear_baseline
        )

        records.append(
            {
                "well_id": well_id,
                "source_row_index": int(row_index),
                "MD": md[row_index],
                "actual_TVT": actual_tvt[row_index],
                "TVT_at_PS": tvt_at_ps,
                "linear_baseline_TVT": linear_baseline,
                "matched_typewell_TVT": matched_tvt,
                "alignment_prediction_TVT": alignment_prediction,
                "match_score": match["match_score"],
                "match_correlation": match["match_correlation"],
                "match_confidence": match["match_confidence"],
                "match_coverage": match["match_coverage"],
                "match_direction": match["match_direction"],
                "used_baseline_fallback": int(
                    not np.isfinite(matched_tvt)
                ),
            }
        )

    predictions_df = pd.DataFrame(records)

    valid_matches = predictions_df[
        predictions_df["matched_typewell_TVT"].notna()
    ]

    summary = {
        "well_id": well_id,
        "sampled_rows": len(predictions_df),
        "matched_rows": len(valid_matches),
        "match_rate": (
            len(valid_matches) / len(predictions_df)
            if len(predictions_df) > 0
            else np.nan
        ),
        "baseline_RMSE": (
            rmse(
                predictions_df["actual_TVT"].to_numpy(),
                predictions_df["linear_baseline_TVT"].to_numpy(),
            )
            if len(predictions_df) > 0
            else np.nan
        ),
        "alignment_RMSE_matched_only": (
            rmse(
                valid_matches["actual_TVT"].to_numpy(),
                valid_matches["matched_typewell_TVT"].to_numpy(),
            )
            if len(valid_matches) > 0
            else np.nan
        ),
        "alignment_with_fallback_RMSE": (
            rmse(
                predictions_df["actual_TVT"].to_numpy(),
                predictions_df[
                    "alignment_prediction_TVT"
                ].to_numpy(),
            )
            if len(predictions_df) > 0
            else np.nan
        ),
        "mean_match_correlation": (
            valid_matches["match_correlation"].mean()
            if len(valid_matches) > 0
            else np.nan
        ),
    }

    return predictions_df, summary


# ============================================================
# Main
# ============================================================
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_ps_manifest()
    validation_wells = load_xgb_validation_wells()

    train_manifest = manifest[
        (manifest["split"] == "train")
        & (manifest["well_id"].isin(validation_wells))
    ].copy()

    if train_manifest.empty:
        raise ValueError(
            "No XGBoost validation wells found in ps_manifest.csv"
        )

    print("=" * 70)
    print("Typewell GR Alignment Evaluation")
    print("=" * 70)
    print(f"Validation wells: {len(train_manifest)}")
    print(f"Rows sampled per well: {ROWS_PER_WELL}")
    print(f"GR window size: {WINDOW_SIZE}")
    print(
        "This is a diagnostic baseline, not a direct "
        "replacement for CatBoost."
    )

    all_predictions = []
    all_summaries = []

    total_wells = len(train_manifest)

    for position, (_, manifest_row) in enumerate(
        train_manifest.iterrows(),
        start=1,
    ):
        well_id = manifest_row["well_id"]

        predictions_df, summary = evaluate_one_well(
            well_id=well_id,
            manifest_row=manifest_row,
        )

        all_predictions.append(predictions_df)
        all_summaries.append(summary)

        print(
            f"Completed {position:>3}/{total_wells} | "
            f"{well_id} | "
            f"match rate: {summary['match_rate']:.1%}"
        )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(all_summaries)

    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    actual = predictions_df["actual_TVT"].to_numpy(dtype=float)
    baseline = predictions_df[
        "linear_baseline_TVT"
    ].to_numpy(dtype=float)

    aligned = predictions_df[
        "alignment_prediction_TVT"
    ].to_numpy(dtype=float)

    matched_rows = predictions_df[
        predictions_df["matched_typewell_TVT"].notna()
    ]

    overall_baseline_rmse = rmse(actual, baseline)
    overall_alignment_fallback_rmse = rmse(actual, aligned)

    if len(matched_rows) > 0:
        matched_only_rmse = rmse(
            matched_rows["actual_TVT"].to_numpy(dtype=float),
            matched_rows[
                "matched_typewell_TVT"
            ].to_numpy(dtype=float),
        )
    else:
        matched_only_rmse = np.nan

    report = {
        "validation_wells": int(len(train_manifest)),
        "sampled_rows": int(len(predictions_df)),
        "matched_rows": int(len(matched_rows)),
        "overall_match_rate": float(
            len(matched_rows) / len(predictions_df)
        ),
        "linear_baseline_RMSE_same_sample": overall_baseline_rmse,
        "alignment_RMSE_matched_only": matched_only_rmse,
        "alignment_with_fallback_RMSE": overall_alignment_fallback_rmse,
        "window_size": WINDOW_SIZE,
        "candidate_half_width_TVT": CANDIDATE_HALF_WIDTH_TVT,
        "candidate_TVT_step": CANDIDATE_TVT_STEP,
        "rows_per_well": ROWS_PER_WELL,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 70)
    print("Typewell Alignment Results")
    print("=" * 70)
    print(f"Sampled rows:                    {len(predictions_df):,}")
    print(
        f"Rows with valid GR alignment:    "
        f"{len(matched_rows):,}"
    )
    print(
        f"GR alignment match rate:         "
        f"{len(matched_rows) / len(predictions_df):.2%}"
    )
    print(
        f"Linear baseline RMSE:            "
        f"{overall_baseline_rmse:.6f}"
    )
    print(
        f"Alignment RMSE (matched rows):   "
        f"{matched_only_rmse:.6f}"
    )
    print(
        f"Alignment + fallback RMSE:       "
        f"{overall_alignment_fallback_rmse:.6f}"
    )
    print("=" * 70)

    print("\nSaved files")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Well summary: {SUMMARY_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()