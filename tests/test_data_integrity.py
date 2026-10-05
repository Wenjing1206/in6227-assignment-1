"""检查 Label 处理及 Preprocessing 边界，防止评估中出现静默错误。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_experiment import clean_labels, make_pipeline


class DataIntegrityTests(unittest.TestCase):
    def test_missing_labels_are_excluded_and_positive_class_is_explicit(self):
        raw = pd.DataFrame({"value": [1, 2, 3, 4], "label": ["yes", "no", None, " "]})
        clean, y, excluded = clean_labels(raw)
        self.assertEqual(excluded, 2)
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(clean.index.tolist(), [0, 1])
        self.assertEqual(len(raw), 4)

    def test_invalid_label_stops_execution(self):
        with self.assertRaises(ValueError):
            clean_labels(pd.DataFrame({"label": ["yes", "unexpected"]}))

    def test_preprocessor_uses_training_statistics_and_accepts_unseen_category(self):
        # 测试集中的 9999 和新类别不得改变训练时的插补统计量或类别字典。
        train = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0], "category": ["A", "B", "A", "B"]})
        test = pd.DataFrame({"value": [9999.0, np.nan], "category": ["NEW", "A"]})
        for name in ["Logistic Regression", "Random Forest"]:
            with self.subTest(name=name):
                pipe = make_pipeline(["value"], ["category"], name)
                prep = pipe["preprocess"].fit(train)
                transformed = prep.transform(test)
                self.assertTrue(np.isfinite(transformed).all())
                self.assertEqual(prep.named_transformers_["num"]["impute"].statistics_[0], 2.0)
                self.assertEqual(prep.named_transformers_["cat"]["encode"].categories_[0].tolist(), ["A", "B"])
                self.assertEqual(transformed.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
