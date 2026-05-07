"""Complete resistivity ML workflow for 500 C and 600 C data.

This script reads Excel/CSV feature tables, analyzes correlations, trains several
regression models on log10(rho), saves reports/figures/models, and optionally
predicts ranked candidate compounds.

Example:
    python3 scripts/complete_resistivity_workflow.py \
      --data-dir "D:/Machine learning" \
      --file-500 features_500.xlsx \
      --file-600 features_600.xlsx \
      --target rho_clean
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LEAKAGE_KEYWORDS = ("rho", "resistivity", "电阻率")


@dataclass(frozen=True)
class DatasetConfig:
    label: str
    path: Path
    candidate_path: Path | None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}，请使用 csv/xlsx/xls")

    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(axis=1, how="all")
    unnamed = [column for column in frame.columns if column.lower().startswith("unnamed")]
    return frame.drop(columns=unnamed, errors="ignore")


def valid_numeric_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric_columns if frame[column].nunique(dropna=True) > 1]


def correlation_analysis(
    frame: pd.DataFrame,
    target: str,
    output_dir: Path,
    label: str,
    corr_threshold: float,
    make_plots: bool,
) -> list[str]:
    columns = valid_numeric_columns(frame)
    if target not in columns:
        raise ValueError(f"{label}: 目标列 {target!r} 不是有效数值列，或没有变化。")

    corr = frame[columns].corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
    corr.to_csv(output_dir / f"correlation_matrix_{label}.csv", encoding="utf-8-sig")

    target_corr = corr[target].drop(labels=[target], errors="ignore")
    target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
    target_corr.to_csv(
        output_dir / f"correlation_with_{target}_{label}.csv",
        header=["correlation"],
        encoding="utf-8-sig",
    )

    selected = target_corr[target_corr.abs() > corr_threshold].index.tolist()
    print(f"\n{label} 与 {target} 相关性绝对值 > {corr_threshold} 的特征:")
    print(selected)
    print(f"\n{label} 与 {target} 的相关系数排序:")
    print(target_corr)

    if make_plots:
        plt.figure(figsize=(13, 10))
        sns.heatmap(
            corr,
            annot=len(corr.columns) <= 30,
            cmap="coolwarm",
            fmt=".2f",
            annot_kws={"size": 8},
            xticklabels=True,
            yticklabels=True,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75},
            mask=corr.isnull(),
        )
        plt.title(f"{label} 相关系数热力图", fontsize=16)
        plt.xticks(fontsize=9, rotation=45, ha="right")
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"correlation_heatmap_{label}.png", dpi=300)
        plt.close()

    return selected


def feature_columns(
    frame: pd.DataFrame,
    target: str,
    selected_by_corr: list[str],
    use_selected_features: bool,
    exclude_columns: set[str],
) -> list[str]:
    if use_selected_features:
        columns = selected_by_corr
    else:
        columns = valid_numeric_columns(frame)

    features: list[str] = []
    target_lower = target.lower()
    for column in columns:
        column_lower = column.lower()
        if column == target or column in exclude_columns:
            continue
        if any(keyword in column_lower for keyword in LEAKAGE_KEYWORDS):
            if column_lower != target_lower:
                continue
        features.append(column)

    if not features:
        raise ValueError("没有可用于建模的特征列，请检查数据列名和筛选阈值。")
    return features


def make_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 25))),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=random_state,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }


def cross_validator(n_samples: int, random_state: int):
    if n_samples < 3:
        return None
    if n_samples < 10:
        return LeaveOneOut()
    return KFold(n_splits=min(5, n_samples), shuffle=True, random_state=random_state)


def evaluate_models(
    x: pd.DataFrame,
    y_log10: np.ndarray,
    random_state: int,
) -> tuple[str, dict[str, dict[str, float | str]]]:
    cv = cross_validator(len(x), random_state)
    results: dict[str, dict[str, float | str]] = {}
    best_model_name = ""
    best_rmse = float("inf")

    for model_name, model in make_models(random_state).items():
        if cv is None:
            model.fit(x, y_log10)
            y_pred = model.predict(x)
            validation = "training_set_only"
        else:
            y_pred = cross_val_predict(model, x, y_log10, cv=cv)
            validation = type(cv).__name__

        error = y_pred - y_log10
        metrics: dict[str, float | str] = {
            "validation": validation,
            "rmse_log10": float(np.sqrt(np.mean(error**2))),
            "mae_log10": float(mean_absolute_error(y_log10, y_pred)),
        }
        if len(y_log10) >= 3 and np.unique(y_log10).size > 1:
            metrics["r2_log10"] = float(r2_score(y_log10, y_pred))
        results[model_name] = metrics

        if metrics["rmse_log10"] < best_rmse:
            best_rmse = float(metrics["rmse_log10"])
            best_model_name = model_name

    return best_model_name, results


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        score_name = "importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).ravel()
        score_name = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "score_type", "score"])

    rows = pd.DataFrame(
        {
            "feature": feature_names,
            "score_type": score_name,
            "score": values,
        }
    )
    return rows.sort_values("score", ascending=False).reset_index(drop=True)


def plot_feature_importance(importance: pd.DataFrame, output_path: Path, title: str) -> None:
    if importance.empty:
        return
    top = importance.head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["score"])
    plt.title(title)
    plt.xlabel(top["score_type"].iloc[0])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def prepare_training_data(
    frame: pd.DataFrame,
    target: str,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    working = frame.copy()
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[target])
    working = working.loc[working[target] > 0].copy()
    if working.empty:
        raise ValueError(f"{target} 没有正的数值，无法进行 log10 建模。")

    x = working[features].copy()
    for column in features:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    y_log10 = np.log10(working[target].to_numpy(dtype=float))
    return x, y_log10, working


def train_one_dataset(
    config: DatasetConfig,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    print(f"\n{'=' * 80}")
    print(f"开始处理 {config.label}: {config.path}")
    frame = read_table(config.path)
    print(f"{config.label} 数据形状: {frame.shape}")
    print(frame.head())

    selected = correlation_analysis(
        frame=frame,
        target=args.target,
        output_dir=output_dir,
        label=config.label,
        corr_threshold=args.corr_threshold,
        make_plots=not args.no_plots,
    )
    features = feature_columns(
        frame=frame,
        target=args.target,
        selected_by_corr=selected,
        use_selected_features=args.use_selected_features,
        exclude_columns=set(args.exclude_columns),
    )
    print(f"\n{config.label} 用于建模的特征数量: {len(features)}")
    print(features)

    x, y_log10, cleaned = prepare_training_data(frame, args.target, features)
    cleaned.to_csv(output_dir / f"cleaned_training_rows_{config.label}.csv", index=False)

    best_name, results = evaluate_models(x, y_log10, args.random_state)
    print(f"\n{config.label} 模型交叉验证结果:")
    print(pd.DataFrame(results).T.sort_values("rmse_log10"))
    print(f"\n{config.label} 最佳模型: {best_name}")

    best_model = make_models(args.random_state)[best_name]
    best_model.fit(x, y_log10)

    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": features,
        "target": args.target,
        "target_transform": "log10",
        "label": config.label,
    }
    model_path = output_dir / f"model_{config.label}.joblib"
    joblib.dump(model_bundle, model_path)

    importance = feature_importance(best_model, features)
    importance.to_csv(output_dir / f"feature_importance_{config.label}.csv", index=False)
    if not args.no_plots:
        plot_feature_importance(
            importance,
            output_dir / f"feature_importance_{config.label}.png",
            f"{config.label} 特征重要性",
        )

    prediction_path = None
    if config.candidate_path is not None:
        prediction_path = predict_candidates(
            candidate_path=config.candidate_path,
            model_bundle=model_bundle,
            output_dir=output_dir,
            label=config.label,
            direction=args.direction,
        )

    report = {
        "label": config.label,
        "data_path": str(config.path),
        "rows_raw": int(len(frame)),
        "rows_used": int(len(x)),
        "target": args.target,
        "feature_count": len(features),
        "features": features,
        "selected_by_correlation": selected,
        "use_selected_features": bool(args.use_selected_features),
        "best_model": best_name,
        "metrics": results,
        "model_path": str(model_path),
        "prediction_path": str(prediction_path) if prediction_path else None,
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    (output_dir / f"report_{config.label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def predict_candidates(
    candidate_path: Path,
    model_bundle: dict[str, Any],
    output_dir: Path,
    label: str,
    direction: str,
) -> Path:
    candidates = read_table(candidate_path)
    features = model_bundle["feature_columns"]
    missing = [feature for feature in features if feature not in candidates.columns]
    if missing:
        raise ValueError(f"{label} 候选表缺少特征列: {missing}")

    x_new = candidates[features].copy()
    for column in features:
        x_new[column] = pd.to_numeric(x_new[column], errors="coerce")

    log10_rho_pred = model_bundle["model"].predict(x_new)
    output = candidates.copy()
    output["log10_rho_pred"] = log10_rho_pred
    output["rho_pred"] = np.power(10.0, log10_rho_pred)
    output = output.sort_values("rho_pred", ascending=direction == "min").reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1)

    prediction_path = output_dir / f"candidate_predictions_{label}.csv"
    output.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    print(f"\n{label} 候选化合物预测排序前 20:")
    print(output.head(20))
    return prediction_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="完整电阻率机器学习流程")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="数据所在目录，例如 Windows 下可写 D:/Machine learning",
    )
    parser.add_argument("--file-500", default="features_500.xlsx")
    parser.add_argument("--file-600", default="features_600.xlsx")
    parser.add_argument("--candidate-500", default=None, help="可选：500 C 候选化合物表")
    parser.add_argument("--candidate-600", default=None, help="可选：600 C 候选化合物表")
    parser.add_argument("--target", default="rho_clean", help="目标电阻率列名")
    parser.add_argument(
        "--direction",
        choices=["min", "max"],
        default="min",
        help="最佳材料按低电阻率(min)还是高电阻率(max)排序",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.2,
        help="输出强相关特征的阈值",
    )
    parser.add_argument(
        "--use-selected-features",
        action="store_true",
        help="只用相关性绝对值超过阈值的特征建模；默认使用全部有效数值特征",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=[],
        help="额外排除的列名，例如 sample_id batch_id",
    )
    parser.add_argument("--output-dir", default="reports/complete_workflow")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true", help="不保存图片")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        DatasetConfig(
            label="500C",
            path=data_dir / args.file_500,
            candidate_path=data_dir / args.candidate_500 if args.candidate_500 else None,
        ),
        DatasetConfig(
            label="600C",
            path=data_dir / args.file_600,
            candidate_path=data_dir / args.candidate_600 if args.candidate_600 else None,
        ),
    ]

    reports = [train_one_dataset(config, args, output_dir) for config in configs]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部完成。结果已保存到: {output_dir}")
    print(f"汇总报告: {summary_path}")


if __name__ == "__main__":
    main()
