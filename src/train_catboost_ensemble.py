"""Train a CatBoost ensemble for the Christmas tree survival task.

The script reads train/test CSV files, builds additional tabular features,
validates several CatBoost models with stratified cross-validation and creates
`submission.csv` with predicted probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET_COL = "survived_to_18jan"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-inspired features for the tree survival task."""
    df = df.copy()

    columns_with_missing_values = [
        "humidity_pct",
        "radiator_distance_m",
        "garland_hours_per_day",
        "waterings_per_week",
        "ornaments_weight_kg",
        "ceiling_height_m",
    ]

    for column in columns_with_missing_values:
        df[f"{column}_isna"] = df[column].isna().astype(int)

    df["missing_count"] = df[columns_with_missing_values].isna().sum(axis=1)

    # Tree age and size.
    df["cut_age_to_jan18"] = df["cut_days_before_jan1"] + 17
    df["tree_height_m"] = df["tree_height_cm"] / 100.0
    df["cut_age_effective"] = df["cut_age_to_jan18"] * (1 - df["potted_tree"])

    # Heat, humidity and dryness.
    df["dryness"] = df["room_temp_c"] - (df["humidity_pct"] / 10.0)
    df["humidity_deficit"] = 100.0 - df["humidity_pct"]
    df["temp_humidity_ratio"] = df["room_temp_c"] / (df["humidity_pct"] + 1.0)
    df["temp_squared"] = df["room_temp_c"] ** 2
    df["humidity_squared"] = df["humidity_pct"] ** 2
    df["temp_x_humidity_deficit"] = df["room_temp_c"] * df["humidity_deficit"]

    # Heat source and garland-related risk.
    garland_hours = df["garland_hours_per_day"].fillna(0)
    electric_heater = (df["heating_type"] == "electric_heater").astype(int)
    df["heat_source_score"] = garland_hours + 2.0 * electric_heater
    df["heat_risk"] = df["heat_source_score"] / (df["radiator_distance_m"] + 0.5)
    df["distance_heat_inverse"] = 1.0 / (df["radiator_distance_m"] + 0.3)
    df["garland_heat_load"] = garland_hours * (1 + df["led_garland"].fillna(0))
    df["garland_led_interaction"] = garland_hours * (1 + df["led_garland"].fillna(0))
    df["garland_x_temp"] = garland_hours * df["room_temp_c"]
    df["garland_per_cut_age"] = garland_hours / (df["cut_age_to_jan18"] + 1.0)

    # Watering and care intensity.
    df["watering_intensity"] = df["waterings_per_week"] + df["mist_spray"].fillna(0)
    df["watering_per_cut_age"] = df["watering_intensity"] / (df["cut_age_to_jan18"] + 1.0)
    df["watering_per_effective_age"] = df["watering_intensity"] / (df["cut_age_effective"] + 1.0)
    df["water_per_height"] = df["waterings_per_week"] / (df["tree_height_m"] + 0.5)
    df["potted_x_watering"] = df["potted_tree"] * df["watering_intensity"]

    # Apartment and placement context.
    df["area_height"] = df["apartment_area_m2"] * df["ceiling_height_m"]
    df["area_per_tree_height"] = df["apartment_area_m2"] / (df["tree_height_m"] + 0.5)
    df["area_per_person"] = df["apartment_area_m2"] / (df["children_count"] + 1.0)
    df["garland_per_area"] = garland_hours / (df["apartment_area_m2"] + 5.0)
    df["heat_per_area"] = df["heat_source_score"] / (df["apartment_area_m2"] + 10.0)
    df["heat_risk_per_area"] = df["heat_risk"] / (df["apartment_area_m2"] + 10.0)
    df["vent_temp"] = df["window_ventilation_per_day"] * df["room_temp_c"]

    # Categorical interactions encoded as numerical helpers.
    df["cat_x_dense"] = df["cat_present"] * (df["tree_form"] == "dense").astype(int)
    df["height_x_form_dense"] = df["tree_height_m"] * (df["tree_form"] == "dense").astype(int)
    df["children_cat_interaction"] = df["children_count"] * df["cat_present"]

    wing_map = {"north": -1.0, "south": 1.0, "east": 0.4, "west": 0.4}
    df["sun_score"] = df["wing"].map(wing_map).astype(float)
    df["sun_x_temp"] = df["sun_score"] * df["room_temp_c"]

    # Smooth transformations for skewed numerical features.
    df["log_area"] = np.log1p(df["apartment_area_m2"])
    df["log_ornaments"] = np.log1p(df["ornaments_weight_kg"])
    df["log_tree_height"] = np.log1p(df["tree_height_cm"])

    return df


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that should be handled by CatBoost as categorical."""
    return [
        column
        for column in df.columns
        if df[column].dtype == "object" or str(df[column].dtype).startswith("category")
    ]


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Convert predictions to rank-based values in the 0..1 range."""
    return pd.Series(values).rank(method="average").values / (len(values) + 1.0)


def logit(values: np.ndarray) -> np.ndarray:
    """Apply a stable logit transformation."""
    clipped_values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped_values / (1 - clipped_values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply the sigmoid transformation."""
    return 1.0 / (1.0 + np.exp(-values))


def get_model_params() -> list[dict]:
    """Return CatBoost configurations used in the ensemble."""
    common_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "allow_writing_files": False,
        "early_stopping_rounds": 100,
        "auto_class_weights": "Balanced",
        "verbose": 200,
    }

    return [
        {
            **common_params,
            "iterations": 700,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 8.0,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 1.0,
            "random_strength": 1.0,
            "random_seed": 42,
            "boosting_type": "Ordered",
        },
        {
            **common_params,
            "iterations": 700,
            "learning_rate": 0.04,
            "depth": 7,
            "l2_leaf_reg": 6.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.9,
            "random_seed": 77,
            "boosting_type": "Plain",
            "one_hot_max_size": 20,
        },
        {
            **common_params,
            "iterations": 800,
            "learning_rate": 0.035,
            "depth": 9,
            "l2_leaf_reg": 10.0,
            "bootstrap_type": "MVS",
            "subsample": 0.8,
            "min_data_in_leaf": 40,
            "rsm": 0.9,
            "random_strength": 1.5,
            "random_seed": 101,
            "boosting_type": "Ordered",
            "border_count": 128,
        },
    ]


def choose_best_blend(
    y_true: pd.Series,
    oof_predictions: list[pd.Series],
    test_predictions: list[np.ndarray],
    model_scores: list[float],
    model_indices: list[int],
) -> tuple[str, float, np.ndarray]:
    """Try several blending strategies and return the best one by OOF ROC-AUC."""
    weights = np.array([model_scores[index] for index in model_indices], dtype=float)
    weights = weights / weights.sum()

    weighted_oof = np.zeros(len(y_true))
    weighted_test = np.zeros(len(test_predictions[0]))
    rank_oof = np.zeros(len(y_true))
    rank_test = np.zeros(len(test_predictions[0]))
    logit_oof = np.zeros(len(y_true))
    logit_test = np.zeros(len(test_predictions[0]))

    for weight, index in zip(weights, model_indices):
        model_oof = oof_predictions[index].values
        model_test = test_predictions[index]

        weighted_oof += model_oof * weight
        weighted_test += model_test * weight
        rank_oof += rank_normalize(model_oof) * weight
        rank_test += rank_normalize(model_test) * weight
        logit_oof += logit(model_oof) * weight
        logit_test += logit(model_test) * weight

    blend_candidates = {
        "weighted": (roc_auc_score(y_true, weighted_oof), weighted_test),
        "rank": (roc_auc_score(y_true, rank_oof), rank_test),
        "logit": (roc_auc_score(y_true, sigmoid(logit_oof)), sigmoid(logit_test)),
    }

    best_name, (best_score, best_predictions) = max(
        blend_candidates.items(),
        key=lambda item: item[1][0],
    )

    scores_text = ", ".join(
        f"{name}={score:.6f}" for name, (score, _) in blend_candidates.items()
    )
    print(f"Blend {model_indices}: {scores_text} -> best={best_name} ({best_score:.6f})")

    return best_name, best_score, best_predictions


def train_and_predict(data_dir: Path, output_path: Path) -> None:
    """Run the full training, validation and submission generation pipeline."""
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    train_features = add_features(train_df)
    test_features = add_features(test_df)

    y = train_features[TARGET_COL]
    X = train_features.drop(columns=[TARGET_COL])
    categorical_columns = get_categorical_columns(X)

    print(f"Features after engineering: {X.shape[1]}")
    print(f"Categorical columns: {len(categorical_columns)}")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_params = get_model_params()

    model_scores: list[float] = []
    oof_predictions: list[pd.Series] = []
    test_predictions: list[np.ndarray] = []

    for model_number, params in enumerate(model_params, start=1):
        print(f"\n=== Model {model_number}/{len(model_params)} ===")
        model_oof = pd.Series(0.0, index=X.index)
        model_test = np.zeros(len(test_features))

        for fold_number, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            train_pool = Pool(X_train, y_train, cat_features=categorical_columns)
            valid_pool = Pool(X_valid, y_valid, cat_features=categorical_columns)
            test_pool = Pool(test_features, cat_features=categorical_columns)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            valid_predictions = model.predict_proba(valid_pool)[:, 1]
            fold_score = roc_auc_score(y_valid, valid_predictions)
            print(f"Model {model_number} | fold {fold_number}: ROC-AUC = {fold_score:.6f}")

            model_oof.iloc[valid_index] = valid_predictions
            model_test += model.predict_proba(test_pool)[:, 1] / splitter.n_splits

        model_score = roc_auc_score(y, model_oof)
        print(f"Model {model_number}: OOF ROC-AUC = {model_score:.6f}")

        model_scores.append(model_score)
        oof_predictions.append(model_oof)
        test_predictions.append(model_test)

    all_indices = list(range(len(model_params)))
    _, full_blend_score, full_blend_predictions = choose_best_blend(
        y,
        oof_predictions,
        test_predictions,
        model_scores,
        all_indices,
    )

    top_model_count = 2
    top_indices = np.argsort(model_scores)[-top_model_count:][::-1].tolist()
    _, top_blend_score, top_blend_predictions = choose_best_blend(
        y,
        oof_predictions,
        test_predictions,
        model_scores,
        top_indices,
    )

    if top_blend_score >= full_blend_score:
        final_predictions = top_blend_predictions
        print(f"Final choice: top-{top_model_count} blend, OOF ROC-AUC = {top_blend_score:.6f}")
    else:
        final_predictions = full_blend_predictions
        print(f"Final choice: full blend, OOF ROC-AUC = {full_blend_score:.6f}")

    submission = sample_submission.copy()
    submission[TARGET_COL] = final_predictions
    submission.to_csv(output_path, index=False)

    print(f"Saved submission to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CatBoost ensemble and create a competition submission.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory with train.csv, test.csv and sample_submission.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path for the generated submission file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_predict(data_dir=args.data_dir, output_path=args.output)


if __name__ == "__main__":
    main()
