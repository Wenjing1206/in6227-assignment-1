"""IN6227 Assignment 1：可复现的二分类实验，训练与测试严格分离。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

SEED = 42
TARGET = "label"
THRESHOLD = 0.5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def clean_labels(frame):
    """删除无法监督训练或评估的空 Label；未知 Label 直接报错，避免静默误编码。"""
    frame = frame.copy()
    label = frame[TARGET].astype("string").str.strip()
    missing = label.isna() | label.eq("").fillna(False)
    invalid = set(label[~missing].unique()) - {"yes", "no"}
    if invalid:
        raise ValueError(f"Unexpected labels: {sorted(invalid)}")
    frame[TARGET] = label
    clean = frame.loc[~missing].copy()
    return clean, clean[TARGET].map({"no": 0, "yes": 1}).astype(int), int(missing.sum())


def inspect_training(frame, clean, numeric, categorical):
    """EDA 仅使用训练集；IQR 只标记异常范围，不自动删除合法但极端的样本。"""
    outliers = {}
    for col in numeric:
        values = clean[col].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        outliers[col] = {
            "lower_fence": float(low), "upper_fence": float(high),
            "outside_fences": int(((values < low) | (values > high)).sum()),
            "minimum": float(values.min()), "maximum": float(values.max()),
        }
    x = clean.drop(columns=TARGET)
    return {
        "raw_rows": len(frame), "labelled_rows": len(clean),
        "missing_label_rows": len(frame) - len(clean),
        "numeric_features": numeric, "categorical_features": categorical,
        "raw_missing_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "labelled_feature_missing_cells": int(x.isna().sum().sum()),
        "labelled_feature_missing_rows": int(x.isna().any(axis=1).sum()),
        "labels": {k: int(v) for k, v in clean[TARGET].value_counts().items()},
        "positive_prevalence": float(clean[TARGET].eq("yes").mean()),
        "duplicate_labelled_rows": int(clean.duplicated().sum()),
        "duplicate_feature_rows": int(x.duplicated().sum()),
        "category_counts": {c: int(clean[c].nunique()) for c in categorical},
        "explicit_unknown_counts": {c: int(clean[c].eq("Unknown").sum()) for c in categorical},
        "iqr_outliers": outliers,
        "decision": "Keep all 15 features and all valid-label rows; retain extreme values and literal Unknown categories.",
    }


def make_pipeline(numeric, categorical, model_name):
    """所有插补、编码和缩放均在 Pipeline 内拟合，防止 Cross-validation 数据泄漏。"""
    numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model_name == "Logistic Regression":
        numeric_steps.append(("scale", RobustScaler()))
        model = LogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-4, random_state=SEED)
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_features="sqrt", bootstrap=True,
            random_state=SEED, n_jobs=1,
        )
    preprocessing = ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def metrics(y_true, probability):
    """yes 固定为正类；阈值预先设定为 0.5，不根据测试结果调整。"""
    pred = np.asarray(probability) >= THRESHOLD
    return {
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def paired_bootstrap(y, probs, repetitions=1000):
    """配对分层 Bootstrap：两个模型使用相同测试重采样，仅度量固定模型的测试抽样不确定性。"""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    groups = [np.flatnonzero(y == k) for k in [0, 1]]
    delta = []
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        delta.append(average_precision_score(y[idx], probs["Random Forest"][idx]) -
                     average_precision_score(y[idx], probs["Logistic Regression"][idx]))
    return {
        "metric": "test Average Precision difference: Random Forest minus Logistic Regression",
        "repetitions": repetitions, "seed": SEED,
        "estimate": float(average_precision_score(y, probs["Random Forest"]) -
                          average_precision_score(y, probs["Logistic Regression"])),
        "percentile_95_interval": np.quantile(delta, [0.025, 0.975]).tolist(),
        "scope": "Fixed fitted models; stratified iid test-row resampling; excludes training and split uncertainty.",
    }


def make_figures(out, y, probabilities, result):
    """输出独立可检查的图，不凭手工填写数值绘图。"""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.2, 3.1), layout="constrained")
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y, probability)
        ap = result["models"][name]["test"]["average_precision"]
        ax.plot(recall, precision, label=f"{name} (AP {ap:.3f})", lw=1.5)
    ax.axhline(float(np.mean(y)), color="#777777", ls="--", lw=1, label="Positive prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(alpha=0.15)
    fig.savefig(out / "precision_recall.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    for ax, name in zip(axes, probabilities):
        cm = np.asarray(result["models"][name]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > cm.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"], yticklabels=["no", "yes"], xlabel="Predicted", ylabel="Actual", title=name)
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    train_path, test_path = args.data_dir / "train.csv", args.data_dir / "test.csv"
    train_raw = pd.read_csv(train_path)
    train, y_train, _ = clean_labels(train_raw)
    x_train = train.drop(columns=TARGET)
    numeric = x_train.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x_train.columns if c not in numeric]
    eda = inspect_training(train_raw, train, numeric, categorical)
    write_json(out / "eda_train.json", eda)
    train[numeric].describe().to_csv(out / "numeric_summary.csv")
    train[numeric].corr().to_csv(out / "numeric_correlations.csv")

    # 在读取测试集之前固定搜索范围、CV、评价指标和分类阈值。
    grids = {
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "Random Forest": {"model__max_depth": [12, None], "model__min_samples_leaf": [1, 5], "model__class_weight": [None, "balanced"]},
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    folds = list(cv.split(x_train, y_train))
    result = {
        "seed": SEED, "positive_label": "yes", "threshold": THRESHOLD,
        "cv": "3-fold shuffled StratifiedKFold; identical folds for both models",
        "selection_metric": "average_precision", "search_grids": grids,
        "training": eda, "models": {}, "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy", "pandas", "scipy", "matplotlib"]},
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
    }
    fitted = {}
    for name, grid in grids.items():
        print(f"Training {name}", flush=True)
        start = time.perf_counter()
        search = GridSearchCV(
            make_pipeline(numeric, categorical, name), grid,
            scoring={"ap": "average_precision", "roc_auc": "roc_auc", "f1": "f1"},
            refit="ap", cv=folds, n_jobs=args.jobs, error_score="raise", return_train_score=True,
        )
        # 将不收敛视为失败，防止把未收敛的结果写入报告。
        warnings.filterwarnings("error", category=ConvergenceWarning)
        search.fit(x_train, y_train)
        fitted[name] = search.best_estimator_
        results = pd.DataFrame(search.cv_results_)
        slug = name.lower().replace(" ", "_")
        results.to_csv(out / f"cv_{slug}.csv", index=False)
        best = search.best_index_
        record = {
            "best_params": search.best_params_, "candidates": len(results),
            "cv_ap_mean": float(search.best_score_),
            "cv_ap_std": float(results.loc[best, "std_test_ap"]),
            "cv_ap_folds": [float(results.loc[best, f"split{i}_test_ap"]) for i in range(3)],
            "search_and_refit_seconds": time.perf_counter() - start,
            "encoded_features": len(search.best_estimator_["preprocess"].get_feature_names_out()),
        }
        if name == "Logistic Regression":
            record["refit_iterations"] = search.best_estimator_["model"].n_iter_.tolist()
        result["models"][name] = record
        print(json.dumps({"model": name, **record}), flush=True)
    result["selected_by_cv"] = max(result["models"], key=lambda n: result["models"][n]["cv_ap_mean"])
    # 写出模型选择凭据后才读取测试文件。后续不再改变任何模型或阈值。
    write_json(out / "selection_before_test.json", result)
    test_raw = pd.read_csv(test_path)
    if list(test_raw.columns) != list(train_raw.columns):
        raise ValueError("Train/test schemas differ")
    test, y_test, excluded = clean_labels(test_raw)
    x_test = test.drop(columns=TARGET)
    train_hash = set(pd.util.hash_pandas_object(x_train, index=False).tolist())
    overlap = pd.util.hash_pandas_object(x_test, index=False).isin(train_hash)
    if overlap.any():
        raise ValueError(f"Detected {overlap.sum()} feature-identical train/test rows; evaluation stopped")
    result["test"] = {
        "raw_rows": len(test_raw), "labelled_rows": len(test), "missing_label_rows": excluded,
        "labels": {k: int(v) for k, v in test[TARGET].value_counts().items()},
        "positive_prevalence": float(y_test.mean()), "train_feature_overlap": int(overlap.sum()),
        "duplicate_feature_rows": int(x_test.duplicated().sum()),
        "feature_missing_cells": int(x_test.isna().sum().sum()),
        "unseen_categories": {c: sorted(set(x_test[c].dropna()) - set(x_train[c].dropna())) for c in categorical},
    }
    result["test_sha256"] = hashlib.sha256(test_path.read_bytes()).hexdigest()
    probabilities = {}
    predictions = pd.DataFrame({"csv_row_number": test.index + 2, "label": y_test.to_numpy()})
    for name, pipeline in fitted.items():
        assert pipeline.classes_.tolist() == [0, 1]
        p = pipeline.predict_proba(x_test)[:, 1]
        probabilities[name] = p
        result["models"][name]["test"] = metrics(y_test, p)
        result["models"][name]["train_ap"] = float(average_precision_score(y_train, pipeline.predict_proba(x_train)[:, 1]))
        slug = name.lower().replace(" ", "_")
        predictions[f"{slug}_probability"] = p
        predictions[f"{slug}_prediction"] = (p >= THRESHOLD).astype(int)
    # Majority baseline 只用于说明 Accuracy 的局限，不作为第三个调参模型。
    result["majority_baseline"] = metrics(y_test, np.zeros(len(y_test)))
    result["paired_bootstrap"] = paired_bootstrap(y_test.to_numpy(), probabilities)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    write_json(out / "metrics.json", result)
    make_figures(out, y_test, probabilities, result)
    print(json.dumps({"test": result["test"], "models": result["models"], "bootstrap": result["paired_bootstrap"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
