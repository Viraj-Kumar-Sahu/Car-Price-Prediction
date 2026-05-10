from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_validate, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .config import REQUIRED_COLUMNS, TARGET_COLUMN
from .features import (
    FEATURE_ENGINEERING_DESCRIPTION,
    clean_for_native_cat_models,
    engineer_features,
    required_prediction_columns,
)
from .metrics import add_price_bucket, regression_metrics, segment_error_analysis


@dataclass
class TrainResult:
    best_model_name: str
    holdout_metrics: dict[str, float]
    artifact_path: str
    output_dir: str


def _load_and_validate(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def _pipeline(model: Any, preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def _optional_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=1,
        )
    except Exception:
        pass
    return models


def _base_models() -> dict[str, Any]:
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42, n_estimators=300),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "svr_rbf": SVR(kernel="rbf", C=1000.0, epsilon=0.1),
        "knn": KNeighborsRegressor(n_neighbors=9),
    }
    models.update(_optional_models())
    return models


def _cross_validate_models(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    preprocessor: ColumnTransformer,
    cv_repeats: int = 3,
) -> pd.DataFrame:
    cv = RepeatedKFold(n_splits=5, n_repeats=cv_repeats, random_state=42)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "mape": "neg_mean_absolute_percentage_error",
        "medae": "neg_median_absolute_error",
    }

    rows = []
    for name, model in _base_models().items():
        pipe = _pipeline(model, preprocessor)
        res = cross_validate(pipe, X_dev, y_dev, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "model": name,
                "cv_r2_mean": float(np.mean(res["test_r2"])),
                "cv_r2_std": float(np.std(res["test_r2"])),
                "cv_mae_mean": float(-np.mean(res["test_mae"])),
                "cv_rmse_mean": float(-np.mean(res["test_rmse"])),
                "cv_mape_mean": float(-np.mean(res["test_mape"])),
                "cv_medae_mean": float(-np.mean(res["test_medae"])),
            }
        )

    return pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)


def _tune_models(X_dev: pd.DataFrame, y_dev: pd.Series, preprocessor: ColumnTransformer, random_state: int) -> dict[str, Any]:
    tuned: dict[str, Any] = {}

    rf = _pipeline(RandomForestRegressor(random_state=random_state), preprocessor)
    rf_space = {
        "model__n_estimators": [200, 300, 500, 700],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.8],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_space,
        n_iter=20,
        cv=5,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
    )
    rf_search.fit(X_dev, y_dev)
    tuned["random_forest_tuned"] = rf_search.best_estimator_

    gb = _pipeline(GradientBoostingRegressor(random_state=random_state), preprocessor)
    gb_space = {
        "model__n_estimators": [150, 250, 400],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5],
        "model__subsample": [0.7, 0.85, 1.0],
    }
    gb_search = RandomizedSearchCV(
        gb,
        gb_space,
        n_iter=16,
        cv=5,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
    )
    gb_search.fit(X_dev, y_dev)
    tuned["gradient_boosting_tuned"] = gb_search.best_estimator_

    if "xgboost" in _optional_models():
        from xgboost import XGBRegressor

        xgb = _pipeline(
            XGBRegressor(random_state=random_state, objective="reg:squarederror", n_jobs=1),
            preprocessor,
        )
        xgb_space = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
        xgb_search = RandomizedSearchCV(
            xgb,
            xgb_space,
            n_iter=18,
            cv=5,
            scoring="r2",
            random_state=random_state,
            n_jobs=-1,
        )
        xgb_search.fit(X_dev, y_dev)
        tuned["xgboost_tuned"] = xgb_search.best_estimator_

    return tuned


def _native_categorical_benchmarks(train_df: pd.DataFrame, holdout_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    x_train_raw = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col].copy()
    x_hold_raw = holdout_df.drop(columns=[target_col]).copy()
    y_hold = holdout_df[target_col].copy()

    x_train, x_hold = clean_for_native_cat_models(x_train_raw, x_hold_raw)
    cat_cols = x_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    rows = []

    try:
        from catboost import CatBoostRegressor

        # CatBoost uses `iterations`/`depth` instead of sklearn-style `n_estimators`/`max_depth`
        cat_model = CatBoostRegressor(iterations=800, learning_rate=0.05, depth=8, random_state=42, verbose=0)
        cat_model.fit(x_train, y_train, cat_features=cat_cols)
        cat_preds = cat_model.predict(x_hold)
        row = {"model": "catboost_native"}
        row.update(regression_metrics(y_hold.values, cat_preds))
        rows.append(row)
    except Exception:
        pass

    try:
        import lightgbm as lgb

        x_train_lgb = x_train.copy()
        x_hold_lgb = x_hold.copy()
        for c in cat_cols:
            x_train_lgb[c] = x_train_lgb[c].astype("category")
            x_hold_lgb[c] = x_hold_lgb[c].astype("category")

        lgb_model = lgb.LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            random_state=42,
            objective="regression",
        )
        lgb_model.fit(x_train_lgb, y_train)
        lgb_preds = lgb_model.predict(x_hold_lgb)
        row = {"model": "lightgbm_native"}
        row.update(regression_metrics(y_hold.values, lgb_preds))
        rows.append(row)
    except Exception:
        pass

    return pd.DataFrame(rows)


def _time_aware_eval(best_model: Pipeline, df: pd.DataFrame, target_col: str) -> dict[str, float | int]:
    cutoff = int(df["year"].quantile(0.8))
    train_df = df[df["year"] <= cutoff]
    test_df = df[df["year"] > cutoff]

    if train_df.empty or test_df.empty:
        cutoff = int(df["year"].median())
        train_df = df[df["year"] <= cutoff]
        test_df = df[df["year"] > cutoff]

    if train_df.empty or test_df.empty:
        return {"cutoff_year": cutoff, "r2": np.nan, "mae": np.nan, "rmse": np.nan}

    x_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    x_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    model = clone(best_model)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = regression_metrics(y_test.values, preds)
    return {
        "cutoff_year": cutoff,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        **metrics,
    }


def _overfitting_diagnostics(models: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series, x_hold: pd.DataFrame, y_hold: pd.Series) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        fitted = clone(model).fit(x_train, y_train)
        train_pred = fitted.predict(x_train)
        hold_pred = fitted.predict(x_hold)
        train_r2 = regression_metrics(y_train.values, train_pred)["r2"]
        hold_r2 = regression_metrics(y_hold.values, hold_pred)["r2"]
        rows.append({"model": name, "train_r2": train_r2, "holdout_r2": hold_r2, "r2_gap": train_r2 - hold_r2})
    return pd.DataFrame(rows).sort_values("r2_gap", ascending=False)


def _learning_curve_summary(best_model: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    sizes, train_scores, val_scores = learning_curve(
        estimator=best_model,
        X=X,
        y=y,
        cv=5,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "train_size": sizes,
            "train_r2_mean": train_scores.mean(axis=1),
            "train_r2_std": train_scores.std(axis=1),
            "val_r2_mean": val_scores.mean(axis=1),
            "val_r2_std": val_scores.std(axis=1),
        }
    )


def _permutation_importance_report(
    best_model: Pipeline,
    x_hold: pd.DataFrame,
    y_hold: pd.Series,
    n_jobs: int = 1,
) -> pd.DataFrame:
    report = permutation_importance(best_model, x_hold, y_hold, n_repeats=15, random_state=42, scoring="r2", n_jobs=n_jobs)
    pre = best_model.named_steps["preprocess"]
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(report.importances_mean))]
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": report.importances_mean,
        "importance_std": report.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def _optional_shap_report(best_model: Pipeline, x_hold: pd.DataFrame, max_samples: int = 400) -> pd.DataFrame:
    try:
        import shap
    except Exception:
        return pd.DataFrame()

    try:
        model = best_model.named_steps["model"]
        pre = best_model.named_steps["preprocess"]
        x_transformed = pre.transform(x_hold)
        if hasattr(x_transformed, "toarray"):
            x_transformed = x_transformed.toarray()

        if x_transformed.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(x_transformed.shape[0], max_samples, replace=False)
            x_transformed = x_transformed[idx]

        explainer = shap.Explainer(model, x_transformed)
        shap_values = explainer(x_transformed)
        mean_abs = np.abs(shap_values.values).mean(axis=0)

        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def run_training_pipeline(data_path: Path, output_dir: Path, artifact_path: Path, random_state: int = 42) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    raw_df = _load_and_validate(Path(data_path))
    df = engineer_features(raw_df)

    dev_df, holdout_df = train_test_split(df, test_size=0.15, random_state=random_state)

    x_dev = dev_df.drop(columns=[TARGET_COLUMN])
    y_dev = dev_df[TARGET_COLUMN]
    x_hold = holdout_df.drop(columns=[TARGET_COLUMN])
    y_hold = holdout_df[TARGET_COLUMN]

    preprocessor = _build_preprocessor(x_dev)

    cv_repeats = max(1, int(os.getenv("CPP_CV_REPEATS", "3")))
    permutation_n_jobs = int(os.getenv("CPP_PERM_N_JOBS", "1"))

    cv_summary = _cross_validate_models(x_dev, y_dev, preprocessor, cv_repeats=cv_repeats)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)

    tuned_models = _tune_models(x_dev, y_dev, preprocessor, random_state=random_state)

    candidates: dict[str, Pipeline] = {
        row["model"]: _pipeline(_base_models()[row["model"]], preprocessor)
        for _, row in cv_summary.iterrows()
    }
    candidates.update(tuned_models)

    holdout_rows = []
    fitted_candidates: dict[str, Pipeline] = {}
    for name, model in candidates.items():
        fitted = clone(model).fit(x_dev, y_dev)
        pred = fitted.predict(x_hold)
        m = regression_metrics(y_hold.values, pred)
        holdout_rows.append({"model": name, **m})
        fitted_candidates[name] = fitted

    holdout_summary = pd.DataFrame(holdout_rows).sort_values("r2", ascending=False).reset_index(drop=True)

    native_cat_summary = _native_categorical_benchmarks(dev_df, holdout_df, TARGET_COLUMN)
    if not native_cat_summary.empty:
        holdout_summary = pd.concat([holdout_summary, native_cat_summary], ignore_index=True)
        holdout_summary = holdout_summary.sort_values("r2", ascending=False).reset_index(drop=True)

    holdout_summary.to_csv(output_dir / "holdout_summary.csv", index=False)

    best_model_name = str(holdout_summary.iloc[0]["model"])
    if best_model_name in fitted_candidates:
        best_model = fitted_candidates[best_model_name]
    else:
        best_model = max(
            fitted_candidates.items(),
            key=lambda item: regression_metrics(y_hold.values, item[1].predict(x_hold))["r2"],
        )[1]

    best_preds = best_model.predict(x_hold)
    best_holdout_metrics = regression_metrics(y_hold.values, best_preds)

    segment_df = x_hold[["fuel", "transmission", "owner"]].copy()
    segment_df["actual"] = y_hold.values
    segment_df["predicted"] = best_preds
    segment_df = add_price_bucket(segment_df, target_col="actual")
    segment_report = segment_error_analysis(segment_df)
    segment_report.to_csv(output_dir / "segment_error_analysis.csv", index=False)

    time_eval = _time_aware_eval(best_model, df, TARGET_COLUMN)
    with open(output_dir / "time_aware_evaluation.json", "w", encoding="utf-8") as fp:
        json.dump(time_eval, fp, indent=2)

    diag_input = {
        name: model
        for name, model in fitted_candidates.items()
        if any(k in name for k in ["random_forest", "gradient_boosting", "xgboost"])
    }
    if diag_input:
        overfit_df = _overfitting_diagnostics(diag_input, x_dev, y_dev, x_hold, y_hold)
        overfit_df.to_csv(output_dir / "overfitting_diagnostics.csv", index=False)

    lc_df = _learning_curve_summary(best_model, x_dev, y_dev)
    lc_df.to_csv(output_dir / "learning_curve_summary.csv", index=False)

    perm_df = _permutation_importance_report(best_model, x_hold, y_hold, n_jobs=permutation_n_jobs)
    perm_df.head(30).to_csv(output_dir / "permutation_importance_top30.csv", index=False)

    shap_df = _optional_shap_report(best_model, x_hold)
    if not shap_df.empty:
        shap_df.head(30).to_csv(output_dir / "shap_importance_top30.csv", index=False)

    bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "required_columns": required_prediction_columns(x_dev.columns),
        "feature_engineering": FEATURE_ENGINEERING_DESCRIPTION,
    }
    joblib.dump(bundle, artifact_path)

    summary = {
        "best_model": best_model_name,
        "holdout_metrics": best_holdout_metrics,
        "time_aware_metrics": time_eval,
        "artifact_path": str(artifact_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    return TrainResult(
        best_model_name=best_model_name,
        holdout_metrics=best_holdout_metrics,
        artifact_path=str(artifact_path),
        output_dir=str(output_dir),
    )
