"""Train resistivity prediction models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_training_arrays, filter_temperature, read_table
from .schema import TARGET_DEFAULT


def make_models(random_state: int) -> dict[str, Pipeline]:
    """Create conservative baseline models for small tabular datasets."""
    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        random_state=random_state,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }


def cross_validator(n_samples: int, random_state: int):
    """Pick a validation strategy that still works for limited lab data."""
    if n_samples < 3:
        return None
    if n_samples < 10:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=random_state)


def evaluate_model(model: Pipeline, x: pd.DataFrame, y_log10: np.ndarray, cv) -> dict[str, Any]:
    """Evaluate a model with log10-resistivity metrics."""
    if cv is None:
        model.fit(x, y_log10)
        predictions = model.predict(x)
        validation = "training_set_only"
    else:
        predictions = cross_val_predict(model, x, y_log10, cv=cv)
        validation = type(cv).__name__

    errors = predictions - y_log10
    metrics: dict[str, Any] = {
        "validation": validation,
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "mae_log10": float(mean_absolute_error(y_log10, predictions)),
    }
    if len(y_log10) >= 3 and np.unique(y_log10).size > 1:
        metrics["r2_log10"] = float(r2_score(y_log10, predictions))
    return metrics


def feature_summary(model: Pipeline, feature_columns: list[str]) -> list[dict[str, float | str]]:
    """Extract feature importance or coefficient magnitudes from a fitted pipeline."""
    regressor = model.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        values = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        values = np.abs(np.asarray(regressor.coef_)).ravel()
    else:
        return []

    ranked = sorted(
        zip(feature_columns, values, strict=True), key=lambda item: item[1], reverse=True
    )
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in ranked
        if float(importance) > 0
    ]


def train(args: argparse.Namespace) -> dict[str, Any]:
    frame = read_table(args.data)
    frame = filter_temperature(frame, args.temperature, args.temperature_tolerance)
    x, y_log10, y_raw, feature_columns, training_frame = build_training_arrays(
        frame=frame,
        target=args.target,
        include_temperature=args.include_temperature,
    )

    cv = cross_validator(len(x), args.random_state)
    model_results: dict[str, dict[str, Any]] = {}
    best_name: str | None = None
    best_score = float("inf")
    for name, model in make_models(args.random_state).items():
        metrics = evaluate_model(model, x, y_log10, cv)
        model_results[name] = metrics
        if metrics["rmse_log10"] < best_score:
            best_name = name
            best_score = metrics["rmse_log10"]

    if best_name is None:
        raise RuntimeError("No model could be evaluated.")

    best_model = make_models(args.random_state)[best_name]
    best_model.fit(x, y_log10)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_columns,
        "target": args.target,
        "target_transform": "log10",
        "temperature_c": args.temperature,
        "include_temperature": args.include_temperature,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(x)),
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    report = {
        "data": str(Path(args.data)),
        "model_out": str(model_path),
        "target": args.target,
        "temperature_c": args.temperature,
        "include_temperature": args.include_temperature,
        "n_samples": int(len(x)),
        "target_rho_min": float(y_raw.min()),
        "target_rho_median": float(y_raw.median()),
        "target_rho_max": float(y_raw.max()),
        "best_model": best_name,
        "models": model_results,
        "top_features": feature_summary(best_model, feature_columns)[:15],
    }

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.training_rows_out:
        rows_path = Path(args.training_rows_out)
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        training_frame.to_csv(rows_path, index=False)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV/XLSX training table.")
    parser.add_argument("--target", default=TARGET_DEFAULT, help="Positive resistivity column.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature filter, for example 500 or 600.",
    )
    parser.add_argument(
        "--temperature-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance used with --temperature.",
    )
    parser.add_argument(
        "--include-temperature",
        action="store_true",
        help="Use temperature_c as a feature for a combined-temperature model.",
    )
    parser.add_argument("--model-out", required=True, help="Output .joblib model path.")
    parser.add_argument("--report-out", help="Optional JSON training report path.")
    parser.add_argument(
        "--training-rows-out",
        help="Optional CSV containing rows used after filtering and target cleanup.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
