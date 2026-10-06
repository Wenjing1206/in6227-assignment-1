"""中文备注：用小型独立数据测试复用与数据隔离，不重新训练作业模型。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('skill_runner', ROOT / 'classification-report-agent/scripts/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SkillWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.config = {
            'target': 'outcome', 'seed': 42, 'folds': 3,
            'split_strategy': 'iid', 'primary_metric': 'f1_macro',
            'models': [
                {'kind': 'logistic_regression', 'reason': 'linear baseline', 'grid': {'C': [1.0]}},
                {'kind': 'random_forest', 'reason': 'nonlinear comparison', 'n_estimators': 10, 'grid': {'max_depth': [4]}},
            ],
        }
        x, y = load_iris(return_X_y=True, as_frame=True)
        x['outcome'] = y.map({0: 'alpha', 1: 'beta', 2: 'gamma'})
        x['sensor'] = np.where(np.arange(len(x)) % 2, 'indoor', 'outdoor')
        x.loc[::11, 'sensor'] = np.nan
        x.loc[::17, x.columns[0]] = np.nan
        self.data = x

    def test_multiclass_and_missing_target(self):
        # 中文备注：去掉一个训练标签后，不能把缺失值编码成第四个 class。
        data = self.data.copy()
        data.loc[0, 'outcome'] = np.nan
        result = runner.execute(data, None, {}, self.config, self.out)
        self.assertEqual(len(result['classes']), 3)
        self.assertEqual(result['missing_train_target'], 1)
        self.assertEqual(result['test_labeled_rows'], 30)
        self.assertEqual(result['train_rows'] + result['test_rows'], 149)
        for record in result['models'].values():
            self.assertEqual(np.array(record['test']['confusion_matrix']).sum(), 30)

    def test_unlabeled_unseen_and_training_only_imputation(self):
        train = self.data.iloc[:-1].copy()
        test = self.data.iloc[-1:].drop(columns='outcome').copy()
        test['sensor'] = 'new_sensor'
        test.iloc[0, 0] = 987.0
        result = runner.execute(train, test, {}, self.config, self.out)
        self.assertEqual(result['test_labeled_rows'], 0)
        self.assertIsNone(result['baseline'])
        self.assertTrue(all(item['test'] is None for item in result['models'].values()))
        predictions = pd.read_csv(self.out / 'predictions.csv')
        self.assertEqual(len(predictions), 1)
        model = joblib.load(self.out / 'selected_model.joblib')
        preprocessor = model['pipeline'].named_steps['preprocess']
        imputer = preprocessor.named_transformers_['numeric'].named_steps['impute']
        # 中文备注：test 极值不能参与最终 numerical imputation 的 median 学习。
        np.testing.assert_allclose(imputer.statistics_, train.select_dtypes(include='number').median().to_numpy())
        encoded = model['pipeline'].predict(test[model['features']])
        self.assertEqual(model['label_encoder'].inverse_transform(encoded)[0], predictions[result['winner']].iloc[0])

    def test_grouped_split_refused_before_training(self):
        self.config['split_strategy'] = 'grouped'
        with self.assertRaisesRegex(ValueError, 'group/time-aware'):
            runner.execute(self.data, None, {}, self.config, self.out)


if __name__ == '__main__':
    unittest.main()
