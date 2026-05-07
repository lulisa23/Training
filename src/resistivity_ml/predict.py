"""Predict resistivity for candidate compounds and rank them."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import build_prediction_frame, read_table
from .schema import METADATA_COLUMNS


def tree_prediction_std_log10(model, x: pd.DataFrame) -> np.ndarray | None:
    """Estimate model spread across trees when the selected model supports it."""
    regressor = model.named_steps.get("regressor")
    if not hasattr(regressor, "estimators_"):
        return None

    transformer = model[:-1]
    transformed = transformer.transform(x)
    estimators = np.asarray(regressor.estimators_).ravel()
    if estimators.size == 0:
        return None

    predictions = np.vstack([estimator.predict(transformed) for estimator in estimators])
    return predictions.std(axis=0)


def prediction_output_columns(frame: pd.DataFrame) -> list[str]:
    """Keep useful identity columns at the front of the prediction output."""
    return [column for column in METADATA_COLUMNS if column in frame.columns]


def predict(args: argparse.Namespace) -> pd.DataFrame:
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    candidates = read_table(args.candidates)
    x = build_prediction_frame(candidates, feature_columns)

    predicted_log10 = model.predict(x)
    predicted_rho = np.power(10.0, predicted_log10)
    output = candidates.loc[:, prediction_output_columns(candidates)].copy()
    output["predicted_log10_rho"] = predicted_log10
    output["predicted_rho"] = predicted_rho

    spread = tree_prediction_std_log10(model, x)
    if spread is not None:
        output["prediction_std_log10"] = spread
        output["rho_factor_1std"] = np.power(10.0, spread)

    ascending = args.direction == "min"
    output = output.sort_values("predicted_rho", ascending=ascending).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(out_path, index=False)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained .joblib model.")
    parser.add_argument("--candidates", required=True, help="CSV/XLSX candidate table.")
    parser.add_argument("--out", help="Optional output CSV for ranked predictions.")
    parser.add_argument(
        "--direction",
        choices=["min", "max"],
        default="min",
        help="Whether the best compound has minimum or maximum resistivity.",
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to print to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked = predict(args)
    print(ranked.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
