from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from .features import get_price_buckets


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denominator))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "medae": float(median_absolute_error(y_true, y_pred)),
    }


def segment_error_analysis(df: pd.DataFrame) -> pd.DataFrame:
    segment_keys = ["fuel", "transmission", "owner", "price_bucket"]
    out_frames = []

    for key in segment_keys:
        grouped = (
            df.groupby(key)
            .apply(lambda x: pd.Series(regression_metrics(x["actual"].values, x["predicted"].values)))
            .reset_index()
        )
        grouped.insert(0, "segment", key)
        grouped.rename(columns={key: "segment_value"}, inplace=True)
        out_frames.append(grouped)

    return pd.concat(out_frames, ignore_index=True)


def add_price_bucket(df: pd.DataFrame, target_col: str = "actual") -> pd.DataFrame:
    out = df.copy()
    out["price_bucket"] = get_price_buckets(out[target_col], q=5)
    return out
