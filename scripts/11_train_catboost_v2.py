from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


# ============================================================
# Paths and settings
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLD_V2_DIR = PROJECT_ROOT / "data" / "gold_v2"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_CONFIG_PATH = GOLD_V2_DIR / "model_feature_columns.json"
XGB_REPORT_PATH = REPORT_DIR / "xgboost_training_report.json"

TARGET_COLUMN = "target_TVT_residual_from_trajectory_baseline"
BASELINE_COLUMN = "trajectory_ridge_baseline_TVT"

ITERATIONS = 3500
EARLY_STOPPING_ROUNDS = 150
RANDOM_STATE = 42

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": ITERATIONS,
    "learning_rate": 0.045,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "border_count": 128,
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "task_type": "CPU",
    "allow_writing_files": False,
}


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def get_well_id_from_gold_file(file_path: Path) -> str:
    parts = file_path.name.split("__")

    if len(parts) < 2:
        raise ValueError(f"Unexpected Gold V2 filename: {file_path.name}")

    return parts[1]


def load_feature_columns() -> list[str]:
    if not FEATURE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Gold V2 feature config not found: {FEATURE_CONFIG_PATH}"
        )

    with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    feature_columns = config.get("feature_columns")

    if not feature_columns:
        raise ValueError("No feature_columns found in Gold V2 config.")

    if BASELINE_COLUMN not in feature_columns:
        raise ValueError(
            f"{BASELINE_COLUMN} is not in the feature list."
        )

    return feature_columns


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
            "validation_wells not found in xgboost_training_report.json"
        )

    return set(validation_wells)


def clean_feature_array(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    array[~np.isfinite(array)] = np.nan
    return array


def load_split_data(
    feature_columns: list[str],
    validation_wells: set[str],
):
    train_dir = GOLD_V2_DIR / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Gold V2 train folder not found: {train_dir}")

    gold_files = sorted(train_dir.glob("*__gold_v2_features.csv"))

    if not gold_files:
        raise FileNotFoundError(
            f"No Gold V2 training files found inside: {train_dir}"
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

    train_x_blocks = []
    train_y_blocks = []

    valid_x_blocks = []
    valid_y_blocks = []
    valid_well_blocks = []
    valid_source_row_blocks = []

    print(f"Loading {len(gold_files)} Gold V2 training files...")

    for file_number, file_path in enumerate(gold_files, start=1):
        well_id = get_well_id_from_gold_file(file_path)

        df = pd.read_csv(
            file_path,
            usecols=use_columns,
            dtype=dtype_map,
        )

        x_block = clean_feature_array(
            df[feature_columns].to_numpy(dtype=np.float32, copy=True)
        )

        y_block = df[TARGET_COLUMN].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if np.isnan(y_block).any():
            raise ValueError(
                f"{file_path.name} contains missing residual target values."
            )

        if well_id in validation_wells:
            valid_x_blocks.append(x_block)
            valid_y_blocks.append(y_block)

            valid_well_blocks.append(
                np.full(len(df), well_id, dtype=object)
            )

            valid_source_row_blocks.append(
                df["source_row_index"].to_numpy(dtype=np.int32, copy=True)
            )
        else:
            train_x_blocks.append(x_block)
            train_y_blocks.append(y_block)

        print(
            f"Loaded {file_number:>3}/{len(gold_files)} "
            f"| {well_id} | {len(df):,} rows"
        )

    x_train = np.concatenate(train_x_blocks, axis=0)
    y_train = np.concatenate(train_y_blocks, axis=0)

    x_valid = np.concatenate(valid_x_blocks, axis=0)
    y_valid = np.concatenate(valid_y_blocks, axis=0)

    valid_well_ids = np.concatenate(valid_well_blocks, axis=0)
    valid_source_rows = np.concatenate(valid_source_row_blocks, axis=0)

    return (
        x_train,
        y_train,
        x_valid,
        y_valid,
        valid_well_ids,
        valid_source_rows,
        sorted(validation_wells),
    )


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns()
    validation_wells = load_validation_wells()

    print("=" * 70)
    print("CatBoost V2 Training")
    print("=" * 70)
    print(f"Feature count:      {len(feature_columns)}")
    print(f"Target:             {TARGET_COLUMN}")
    print(f"Baseline:           {BASELINE_COLUMN}")
    print(f"Validation wells:   {len(validation_wells)}")
    print(f"Python:             {platform.python_version()}")

    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        valid_well_ids,
        valid_source_rows,
        validation_well_list,
    ) = load_split_data(
        feature_columns=feature_columns,
        validation_wells=validation_wells,
    )

    print("\n" + "=" * 70)
    print("Data loaded")
    print(f"Training rows:      {len(x_train):,}")
    print(f"Validation rows:    {len(x_valid):,}")
    print(f"Feature count:      {x_train.shape[1]}")
    print("=" * 70)

    baseline_index = feature_columns.index(BASELINE_COLUMN)

    valid_baseline_tvt = x_valid[:, baseline_index].copy()

    train_pool = Pool(
        data=x_train,
        label=y_train,
        feature_names=feature_columns,
    )

    valid_pool = Pool(
        data=x_valid,
        label=y_valid,
        feature_names=feature_columns,
    )

    model = CatBoostRegressor(**CATBOOST_PARAMS)

    print("\nTraining CatBoost V2...")

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        verbose=100,
    )

    predicted_residual = model.predict(valid_pool)

    actual_tvt = valid_baseline_tvt + y_valid
    predicted_tvt = valid_baseline_tvt + predicted_residual

    catboost_v2_rmse = rmse(actual_tvt, predicted_tvt)
    baseline_rmse = rmse(actual_tvt, valid_baseline_tvt)

    print("\n" + "=" * 70)
    print("Validation Results")
    print("=" * 70)
    print(f"Best iteration:             {model.get_best_iteration()}")
    print(f"CatBoost V2 validation RMSE:{catboost_v2_rmse:.6f}")
    print(f"Trajectory baseline RMSE:   {baseline_rmse:.6f}")
    print("=" * 70)

    model_path = MODEL_DIR / "catboost_v2_residual_model.cbm"
    model.save_model(model_path)

    feature_path = MODEL_DIR / "catboost_v2_feature_columns.json"

    with open(feature_path, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    validation_predictions = pd.DataFrame(
        {
            "well_id": valid_well_ids,
            "source_row_index": valid_source_rows,
            "trajectory_ridge_baseline_TVT": valid_baseline_tvt,
            "actual_TVT_residual": y_valid,
            "predicted_TVT_residual": predicted_residual,
            "actual_TVT": actual_tvt,
            "predicted_TVT": predicted_tvt,
            "absolute_error": np.abs(actual_tvt - predicted_tvt),
        }
    )

    prediction_path = REPORT_DIR / "catboost_v2_validation_predictions.csv"
    validation_predictions.to_csv(prediction_path, index=False)

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(
                type="PredictionValuesChange"
            ),
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    importance_path = REPORT_DIR / "catboost_v2_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    report = {
        "target_column": TARGET_COLUMN,
        "baseline_column": BASELINE_COLUMN,
        "feature_count": len(feature_columns),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "validation_well_count": len(validation_well_list),
        "validation_wells": validation_well_list,
        "best_iteration": int(model.get_best_iteration()),
        "catboost_v2_validation_RMSE": catboost_v2_rmse,
        "trajectory_baseline_RMSE": baseline_rmse,
        "parameters": CATBOOST_PARAMS,
    }

    report_path = REPORT_DIR / "catboost_v2_training_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nSaved files")
    print(f"Model:              {model_path}")
    print(f"Feature list:       {feature_path}")
    print(f"Training report:    {report_path}")
    print(f"Predictions:        {prediction_path}")
    print(f"Feature importance: {importance_path}")


if __name__ == "__main__":
    main()