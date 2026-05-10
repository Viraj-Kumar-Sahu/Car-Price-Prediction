from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .features import engineer_features


class ModelPredictor:
    def __init__(self, artifact_path: Path):
        bundle = joblib.load(artifact_path)
        self.model = bundle["model"]
        self.required_columns = bundle.get("required_columns", [])

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for inference: {missing}")

        transformed = engineer_features(df.copy())
        preds = self.model.predict(transformed)
        out = df.copy()
        out["predicted_selling_price"] = preds
        return out
