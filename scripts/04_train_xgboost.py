from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Settings
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_CONFIG_PATH = GOLD_DIR / "xgboost_feature_columns.json"

TARGET_COLUMN = "target_TVT_change_from_PS"

VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42

NUM_BOOST_ROUND = 2500
EARLY_STOPPING_ROUNDS = 100

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "learning_rate": 0.04,
    "max_depth": 8,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "max_bin": 256,
    "seed": RANDOM_STATE,
    "nthread": -1,
}


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
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def load_feature_config() -> list[str]:
    """Read the approved Gold feature list."""
    if not FEATURE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Feature configuration not found: {FEATURE_CONFIG_PATH}\n"
            "Run 03_build_gold.py first."
        )

    with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    feature_columns = config.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "No feature columns found in xgboost_feature_columns.json"
        )

    return feature_columns


def load_training_data(
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Loads all post-PS Gold training rows.

    Returns:
        X            Feature matrix
        y            Target: TVT change from PS
        groups       Numeric group ID for whole-well validation
        source_rows  Original row index in the horizontal CSV
        well_lookup  Maps numeric group IDs back to well IDs
    """
    train_dir = GOLD_DIR / "train"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Gold train directory not found: {train_dir}"
        )

    gold_files = sorted(train_dir.glob("*__gold_features.csv"))

    if not gold_files:
        raise FileNotFoundError(
            f"No Gold training CSV files found in: {train_dir}"
        )

    use_columns = feature_columns + [
        TARGET_COLUMN,
        "source_row_index",
    ]

    dtype_map = {
        column: "float32"
        for column in feature_columns + [TARGET_COLUMN]
    }
    dtype_map["source_row_index"] = "int32"

    feature_blocks = []
    target_blocks = []
    group_blocks = []
    source_row_blocks = []

    well_lookup = []

    print(f"Loading {len(gold_files)} Gold training files...")

    for group_id, file_path in enumerate(gold_files):
        well_id = get_well_id_from_gold_file(file_path)

        df = pd.read_csv(
            file_path,
            usecols=use_columns,
            dtype=dtype_map,
        )

        missing_columns = [
            column
            for column in use_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: {missing_columns}"
            )

        X_block = df[feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        y_block = df[TARGET_COLUMN].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if np.isnan(y_block).any():
            raise ValueError(
                f"{file_path.name} contains missing target values."
            )

        feature_blocks.append(X_block)
        target_blocks.append(y_block)

        group_blocks.append(
            np.full(
                shape=len(df),
                fill_value=group_id,
                dtype=np.int32,
            )
        )

        source_row_blocks.append(
            df["source_row_index"].to_numpy(
                dtype=np.int32,
                copy=True,
            )
        )

        well_lookup.append(well_id)

        print(
            f"Loaded {group_id + 1:>3}/{len(gold_files)} "
            f"| {well_id} | {len(df):,} rows"
        )

    print("\nCombining training data into memory...")

    X = np.concatenate(feature_blocks, axis=0)
    y = np.concatenate(target_blocks, axis=0)
    groups = np.concatenate(group_blocks, axis=0)
    source_rows = np.concatenate(source_row_blocks, axis=0)

    return X, y, groups, source_rows, well_lookup


def predict_best_iteration(
    model: xgb.Booster,
    dmatrix: xgb.DMatrix,
) -> np.ndarray:
    """
    Predict using only the best early-stopping iteration.
    Handles different XGBoost versions safely.
    """
    best_iteration = getattr(model, "best_iteration", None)

    if best_iteration is None:
        return model.predict(dmatrix)

    try:
        return model.predict(
            dmatrix,
            iteration_range=(0, best_iteration + 1),
        )
    except TypeError:
        return model.predict(
            dmatrix,
            ntree_limit=best_iteration + 1,
        )


# ============================================================
# Main
# ============================================================
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_config()

    print("=" * 70)
    print("XGBoost Training")
    print("=" * 70)
    print(f"Feature count: {len(feature_columns)}")
    print(f"Target:        {TARGET_COLUMN}")
    print(f"XGBoost:       {xgb.__version__}")
    print(f"Python:        {platform.python_version()}")

    X, y, groups, source_rows, well_lookup = load_training_data(
        feature_columns
    )

    print("\n" + "=" * 70)
    print("Training data loaded")
    print(f"Rows:          {len(X):,}")
    print(f"Features:      {X.shape[1]}")
    print(f"Unique wells:  {len(well_lookup)}")
    print("=" * 70)

    # --------------------------------------------------------
    # Whole-well validation split
    # --------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_index, validation_index = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X[train_index]
    y_train = y[train_index]

    X_valid = X[validation_index]
    y_valid = y[validation_index]

    valid_groups = groups[validation_index]
    valid_source_rows = source_rows[validation_index]

    train_well_ids = sorted(
        {
            well_lookup[group_id]
            for group_id in np.unique(groups[train_index])
        }
    )

    valid_well_ids = sorted(
        {
            well_lookup[group_id]
            for group_id in np.unique(valid_groups)
        }
    )

    print("\nWhole-well split")
    print(f"Training wells:   {len(train_well_ids)}")
    print(f"Validation wells: {len(valid_well_ids)}")
    print(f"Training rows:    {len(X_train):,}")
    print(f"Validation rows:  {len(X_valid):,}")

    # --------------------------------------------------------
    # Extract useful validation references before DMatrix setup
    # --------------------------------------------------------
    tvt_at_ps_index = feature_columns.index("TVT_at_PS")
    baseline_index = feature_columns.index(
        "TVT_linear_baseline_from_PS"
    )

    valid_tvt_at_ps = X_valid[:, tvt_at_ps_index].copy()
    valid_linear_baseline = X_valid[:, baseline_index].copy()

    # --------------------------------------------------------
    # XGBoost training
    # --------------------------------------------------------
    print("\nCreating XGBoost matrices...")

    dtrain = xgb.DMatrix(
        data=X_train,
        label=y_train,
        feature_names=feature_columns,
        missing=np.nan,
    )

    dvalid = xgb.DMatrix(
        data=X_valid,
        label=y_valid,
        feature_names=feature_columns,
        missing=np.nan,
    )

    print("\nTraining model...")

    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[
            (dtrain, "train"),
            (dvalid, "validation"),
        ],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------
    predicted_change = predict_best_iteration(model, dvalid)

    actual_tvt = valid_tvt_at_ps + y_valid
    predicted_tvt = valid_tvt_at_ps + predicted_change

    model_rmse = rmse(actual_tvt, predicted_tvt)
    linear_baseline_rmse = rmse(
        actual_tvt,
        valid_linear_baseline,
    )

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)

    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    print(f"Best iteration:          {best_iteration}")
    print(f"XGBoost validation RMSE: {model_rmse:.6f}")
    print(f"Linear baseline RMSE:    {linear_baseline_rmse:.6f}")
    print("=" * 70)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    model_path = MODEL_DIR / "xgboost_tvt_change_model.json"
    model.save_model(model_path)

    feature_path = MODEL_DIR / "xgboost_feature_columns.json"

    with open(feature_path, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------
    validation_well_ids = [
        well_lookup[group_id]
        for group_id in valid_groups
    ]

    validation_predictions = pd.DataFrame(
        {
            "well_id": validation_well_ids,
            "source_row_index": valid_source_rows,
            "TVT_at_PS": valid_tvt_at_ps,
            "actual_TVT_change_from_PS": y_valid,
            "predicted_TVT_change_from_PS": predicted_change,
            "actual_TVT": actual_tvt,
            "predicted_TVT": predicted_tvt,
            "linear_baseline_TVT": valid_linear_baseline,
            "absolute_error": np.abs(actual_tvt - predicted_tvt),
        }
    )

    validation_prediction_path = (
        REPORT_DIR / "xgboost_validation_predictions.csv"
    )

    validation_predictions.to_csv(
        validation_prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------
    importance_dict = model.get_score(importance_type="gain")

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": [
                importance_dict.get(feature, 0.0)
                for feature in feature_columns
            ],
        }
    ).sort_values(
        by="gain_importance",
        ascending=False,
    )

    importance_path = REPORT_DIR / "xgboost_feature_importance.csv"

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save training report
    # --------------------------------------------------------
    training_report = {
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_columns),
        "total_rows": int(len(X)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "training_well_count": len(train_well_ids),
        "validation_well_count": len(valid_well_ids),
        "validation_wells": valid_well_ids,
        "random_state": RANDOM_STATE,
        "validation_fraction": VALIDATION_FRACTION,
        "best_iteration": (
            int(best_iteration)
            if best_iteration is not None
            else None
        ),
        "best_xgboost_score": (
            float(best_score)
            if best_score is not None
            else None
        ),
        "xgboost_validation_RMSE": model_rmse,
        "linear_baseline_RMSE": linear_baseline_rmse,
        "xgboost_parameters": XGB_PARAMS,
    }

    training_report_path = REPORT_DIR / "xgboost_training_report.json"

    with open(training_report_path, "w", encoding="utf-8") as file:
        json.dump(training_report, file, indent=2)

    print("\nSaved files")
    print(f"Model:               {model_path}")
    print(f"Feature list:        {feature_path}")
    print(f"Training report:     {training_report_path}")
    print(f"Validation output:   {validation_prediction_path}")
    print(f"Feature importance:  {importance_path}")


if __name__ == "__main__":
    main()