"""从保存的逐行预测独立复核报告数字，不重新调用 sklearn 的 Metrics 函数。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "results/metrics.json").read_text())
    before = json.loads((root / "results/selection_before_test.json").read_text())
    pred = pd.read_csv(root / "results/test_predictions.csv")
    y = pred.label.to_numpy()
    assert len(pred) == results["test"]["labelled_rows"]
    assert int(y.sum()) == results["test"]["labels"]["yes"]
    assert pred.csv_row_number.is_unique
    assert results["selected_by_cv"] == before["selected_by_cv"]
    checked = {}
    for name, record in results["models"].items():
        assert record["best_params"] == before["models"][name]["best_params"]
        slug = name.lower().replace(" ", "_")
        p = pred[f"{slug}_probability"].to_numpy()
        z = pred[f"{slug}_prediction"].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(z, (p >= results["threshold"]).astype(int))
        tn, fp = int(((y == 0) & (z == 0)).sum()), int(((y == 0) & (z == 1)).sum())
        fn, tp = int(((y == 1) & (z == 0)).sum()), int(((y == 1) & (z == 1)).sum())
        assert record["test"]["confusion_matrix"] == [[tn, fp], [fn, tp]]
        # AP 按不同分数的 Recall 增量加权 Precision；相同分数作为一组处理。
        order = np.argsort(-p, kind="stable")
        sorted_y, sorted_p = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y)-1]
        cumulative_tp = np.cumsum(sorted_y)[ends]
        recalls = cumulative_tp / y.sum()
        precisions = cumulative_tp / (ends + 1)
        ap = np.sum(np.diff(np.r_[0, recalls]) * precisions)
        # ROC-AUC 使用带 ties 平均秩的 Mann-Whitney 公式独立计算。
        n_pos, n_neg = int(y.sum()), int((1-y).sum())
        ranks = pd.Series(p).rank(method="average").to_numpy()
        auc = (ranks[y == 1].sum() - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
        values = {
            "average_precision": ap, "roc_auc": auc,
            "accuracy": (tn+tp)/len(y), "balanced_accuracy": (tp/(tp+fn)+tn/(tn+fp))/2,
            "precision": tp/(tp+fp), "recall": tp/(tp+fn), "f1": 2*tp/(2*tp+fp+fn),
        }
        for metric, value in values.items():
            assert np.isclose(value, record["test"][metric], rtol=0, atol=1e-12), (name, metric)
        cv = pd.read_csv(root / f"results/cv_{slug}.csv")
        assert np.isclose(cv.mean_test_ap.max(), record["cv_ap_mean"])
        checked[name] = {"all_metrics_match": True, "confusion_matrix": [[tn, fp], [fn, tp]]}
    audit = {"rows_checked": len(pred), "selection_unchanged_after_test": True, "models": checked}
    (root / "results/verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
