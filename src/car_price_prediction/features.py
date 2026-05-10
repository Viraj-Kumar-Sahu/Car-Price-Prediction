from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


TEXT_MISSING = "__missing__"
DEFAULT_REFERENCE_YEAR = int(os.getenv("CPP_REFERENCE_YEAR", "2025"))
FEATURE_ENGINEERING_DESCRIPTION = "brand/model extraction + car_age + power_per_cc + km_per_year"


def engineer_features(df: pd.DataFrame, reference_year: int | None = None) -> pd.DataFrame:
    data = df.copy()

    data["brand"] = data["name"].astype(str).str.split().str[0].fillna(TEXT_MISSING)
    data["model"] = data["name"].astype(str).str.split().str[:2].str.join(" ").fillna(TEXT_MISSING)

    data["max_power"] = (
        data["max_power"].astype(str).str.split().str[0].replace({"nan": np.nan, "": np.nan})
    )
    data["max_power"] = pd.to_numeric(data["max_power"], errors="coerce")

    year_ref = DEFAULT_REFERENCE_YEAR if reference_year is None else int(reference_year)
    data["car_age"] = (year_ref - data["year"]).clip(lower=0)

    engine = pd.to_numeric(data["engine"], errors="coerce")
    max_power = pd.to_numeric(data["max_power"], errors="coerce")
    km_driven = pd.to_numeric(data["km_driven"], errors="coerce")

    data["power_per_cc"] = max_power / (engine + 1e-6)
    data["km_per_year"] = km_driven / (data["car_age"].replace(0, 1))

    return data


def clean_for_native_cat_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    numeric_cols = train.select_dtypes(include=["number", "bool"]).columns
    cat_cols = [c for c in train.columns if c not in numeric_cols]

    for col in numeric_cols:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    for col in cat_cols:
        train[col] = train[col].astype(str).fillna(TEXT_MISSING)
        test[col] = test[col].astype(str).fillna(TEXT_MISSING)

    return train, test


def get_price_buckets(series: pd.Series, q: int = 5) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=q, labels=[f"Q{i}" for i in range(1, q + 1)])


def required_prediction_columns(cols: Iterable[str]) -> list[str]:
    required = [
        "name",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "mileage(km/ltr/kg)",
        "engine",
        "max_power",
        "seats",
    ]
    existing = set(cols)
    return [c for c in required if c in existing]
