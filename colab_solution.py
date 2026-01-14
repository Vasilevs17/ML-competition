# -*- coding: utf-8 -*-
"""Colab-ready training script for the tree survival ROC-AUC task.

Usage (inside Google Colab):
1) Upload train.csv, test.csv, sample_submission.csv to /content (or mount Drive).
2) If needed, install CatBoost: !pip -q install catboost
3) Run this file (or paste into a notebook cell).
4) It will create submission.csv with predicted probabilities.

Примечание: скрипт рассчитан на CPU и не требует GPU.
"""

import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_DIR = "."  # change to your folder in Colab if needed
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "submission.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    miss_cols = [
        "humidity_pct",
        "radiator_distance_m",
        "garland_hours_per_day",
        "waterings_per_week",
        "ornaments_weight_kg",
        "ceiling_height_m",
    ]
    for col in miss_cols:
        df[f"{col}_isna"] = df[col].isna().astype(int)
    df["missing_count"] = df[miss_cols].isna().sum(axis=1)

    df["cut_age_to_jan18"] = df["cut_days_before_jan1"] + 17
    df["tree_height_m"] = df["tree_height_cm"] / 100.0
    df["dryness"] = df["room_temp_c"] - (df["humidity_pct"] / 10.0)

    garland = df["garland_hours_per_day"].fillna(0)
    electric = (df["heating_type"] == "electric_heater").astype(int)
    df["heat_source_score"] = garland + 2.0 * electric
    df["heat_risk"] = df["heat_source_score"] / (df["radiator_distance_m"] + 0.5)

    df["watering_intensity"] = df["waterings_per_week"] + df["mist_spray"].fillna(0)
    df["watering_per_cut_age"] = df["watering_intensity"] / (df["cut_age_to_jan18"] + 1.0)

    df["ornament_density"] = df["ornaments_weight_kg"] / (df["tree_height_m"] + 0.5)
    df["vent_temp"] = df["window_ventilation_per_day"] * df["room_temp_c"]
    df["cat_x_dense"] = df["cat_present"] * (df["tree_form"] == "dense").astype(int)

    wing_map = {"north": -1.0, "south": 1.0, "east": 0.4, "west": 0.4}
    df["sun_score"] = df["wing"].map(wing_map).astype("float")
    df["sun_x_temp"] = df["sun_score"] * df["room_temp_c"]

    df["cut_age_effective"] = df["cut_age_to_jan18"] * (1 - df["potted_tree"])
    df["watering_per_effective_age"] = df["watering_intensity"] / (df["cut_age_effective"] + 1.0)

    df["temp_humidity_ratio"] = df["room_temp_c"] / (df["humidity_pct"] + 1.0)
    df["area_height"] = df["apartment_area_m2"] * df["ceiling_height_m"]
    df["garland_heat_load"] = df["garland_hours_per_day"].fillna(0) * (1 + df["led_garland"].fillna(0))
    df["children_cat_interaction"] = df["children_count"] * df["cat_present"]
    df["garland_per_area"] = df["garland_hours_per_day"].fillna(0) / (df["apartment_area_m2"] + 5.0)
    df["water_per_height"] = df["waterings_per_week"] / (df["tree_height_m"] + 0.5)
    df["heat_per_area"] = df["heat_source_score"] / (df["apartment_area_m2"] + 10.0)
    df["temp_squared"] = df["room_temp_c"] ** 2
    df["humidity_squared"] = df["humidity_pct"] ** 2
    df["garland_x_temp"] = df["garland_hours_per_day"].fillna(0) * df["room_temp_c"]
    df["height_x_form_dense"] = df["tree_height_m"] * (df["tree_form"] == "dense").astype(int)
    df["potted_x_watering"] = df["potted_tree"] * df["watering_intensity"]
    df["area_per_tree_height"] = df["apartment_area_m2"] / (df["tree_height_m"] + 0.5)
    df["area_per_person"] = df["apartment_area_m2"] / (df["children_count"] + 1.0)
    df["distance_heat_inverse"] = 1.0 / (df["radiator_distance_m"] + 0.3)
    df["garland_per_cut_age"] = df["garland_hours_per_day"].fillna(0) / (df["cut_age_to_jan18"] + 1.0)
    df["log_area"] = np.log1p(df["apartment_area_m2"])
    df["log_ornaments"] = np.log1p(df["ornaments_weight_kg"])
    df["log_tree_height"] = np.log1p(df["tree_height_cm"])
    df["garland_led_interaction"] = df["garland_hours_per_day"].fillna(0) * (1 + df["led_garland"].fillna(0))

    return df


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_col = "survived_to_18jan"
    train_fe = add_features(train_df)
    test_fe = add_features(test_df)

    y = train_fe[target_col]
    X = train_fe.drop(columns=[target_col])
    print(f"Готово: признаков после генерации = {X.shape[1]}")

    # Identify categorical columns automatically
    cat_cols = [
        col for col in X.columns
        if X[col].dtype == "object"
        or str(X[col].dtype).startswith("category")
    ]

    model_params = [
        dict(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=700,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=8.0,
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            random_strength=1.0,
            random_seed=42,
            verbose=200,
            allow_writing_files=False,
            early_stopping_rounds=100,
            boosting_type="Ordered",
            auto_class_weights="Balanced",
        ),
        dict(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=700,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=6.0,
            bootstrap_type="Bernoulli",
            subsample=0.9,
            random_seed=77,
            verbose=200,
            allow_writing_files=False,
            early_stopping_rounds=100,
            boosting_type="Plain",
            auto_class_weights="Balanced",
        ),
        dict(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=800,
            learning_rate=0.035,
            depth=9,
            l2_leaf_reg=10.0,
            bootstrap_type="MVS",
            subsample=0.8,
            min_data_in_leaf=40,
            rsm=0.9,
            random_strength=1.5,
            random_seed=101,
            verbose=200,
            allow_writing_files=False,
            early_stopping_rounds=100,
            boosting_type="Ordered",
            auto_class_weights="Balanced",
        ),
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = pd.Series(0.0, index=X.index)
    test_pred = 0.0
    model_weights = []
    oof_list = []
    test_list = []

    for model_idx, params in enumerate(model_params, start=1):
        print(f"\n=== Модель {model_idx}/{len(model_params)} ===")
        model_oof = pd.Series(0.0, index=X.index)
        model_test = 0.0

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
            y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

            train_pool = Pool(X_train, y_train, cat_features=cat_cols)
            valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)
            test_pool = Pool(test_fe, cat_features=cat_cols)

            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            valid_preds = model.predict_proba(valid_pool)[:, 1]
            fold_auc = roc_auc_score(y_valid, valid_preds)
            print(f"Модель {model_idx} | Фолд {fold}: ROC-AUC = {fold_auc:.6f}")

            model_oof.iloc[va_idx] = valid_preds
            model_test += model.predict_proba(test_pool)[:, 1] / skf.n_splits

        model_auc = roc_auc_score(y, model_oof)
        print(f"Модель {model_idx}: OOF ROC-AUC = {model_auc:.6f}")
        model_weights.append(model_auc)
        oof_list.append(model_oof)
        test_list.append(model_test)

    weights = np.array(model_weights)
    weights = weights / weights.sum()
    for weight, model_oof, model_test in zip(weights, oof_list, test_list):
        oof += model_oof * weight
        test_pred += model_test * weight
    oof_auc = roc_auc_score(y, oof)
    print(f"Итоговый OOF ROC-AUC (взвешенное среднее): {oof_auc:.6f}")
    preds = test_pred

    submission = pd.read_csv(SAMPLE_SUB_PATH)
    submission[target_col] = preds
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved submission to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
