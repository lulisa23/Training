from __future__ import annotations

from argparse import Namespace

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from resistivity_ml import predict as predict_module
from resistivity_ml import train as train_module
from resistivity_ml.schema import FEATURE_COLUMNS


def _row(index: int, temperature_c: int = 500) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "compound": f"sample_{index}",
        "formula": f"A{index}B{index}O3",
        "A_site": "A",
        "B_site": "B",
        "temperature_c": temperature_c,
        "rho": 10 ** (-6 + index * 0.03 + (temperature_c - 500) * 0.001),
    }
    for feature_index, feature in enumerate(FEATURE_COLUMNS, start=1):
        row[feature] = 0.1 * feature_index + 0.01 * index
    return row


def test_train_and_predict_smoke(tmp_path, monkeypatch):
    def tiny_models(random_state: int):
        return {
            "random_forest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=10,
                            random_state=random_state,
                            min_samples_leaf=1,
                        ),
                    ),
                ]
            )
        }

    monkeypatch.setattr(train_module, "make_models", tiny_models)

    training_path = tmp_path / "training.csv"
    pd.DataFrame([_row(index) for index in range(12)]).to_csv(training_path, index=False)

    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    report = train_module.train(
        Namespace(
            data=training_path,
            target="rho",
            temperature=500,
            temperature_tolerance=1e-6,
            include_temperature=False,
            model_out=model_path,
            report_out=report_path,
            training_rows_out=None,
            random_state=7,
        )
    )

    assert model_path.exists()
    assert report["best_model"] == "random_forest"
    assert report["n_samples"] == 12

    candidates = pd.DataFrame([_row(0), _row(20)]).drop(columns=["rho"])
    candidate_path = tmp_path / "candidates.csv"
    out_path = tmp_path / "ranked.csv"
    candidates.to_csv(candidate_path, index=False)

    ranked = predict_module.predict(
        Namespace(
            model=model_path,
            candidates=candidate_path,
            out=out_path,
            direction="min",
            top=2,
        )
    )

    assert out_path.exists()
    assert list(ranked["rank"]) == [1, 2]
    assert ranked["predicted_rho"].is_monotonic_increasing
