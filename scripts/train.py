from __future__ import annotations

import argparse
from pathlib import Path

from car_price_prediction.config import DEFAULT_ARTIFACT_PATH, DEFAULT_DATA_PATH, DEFAULT_OUTPUT_DIR
from car_price_prediction.training import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train car price prediction pipeline")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        artifact_path=args.artifact_path,
        random_state=args.random_state,
    )
    print(f"Best model: {result.best_model_name}")
    print(f"Holdout metrics: {result.holdout_metrics}")
    print(f"Artifact: {result.artifact_path}")
    print(f"Outputs: {result.output_dir}")


if __name__ == "__main__":
    main()
