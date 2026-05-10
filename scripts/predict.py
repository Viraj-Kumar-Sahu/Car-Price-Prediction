from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from car_price_prediction.config import DEFAULT_ARTIFACT_PATH
from car_price_prediction.predictor import ModelPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on new car records")
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    predictor = ModelPredictor(args.artifact_path)
    out = predictor.predict_dataframe(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Predictions written to: {args.output_csv}")


if __name__ == "__main__":
    main()
