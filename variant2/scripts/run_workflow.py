"""通用分类实验：profile、train-only CV 选型、冻结 test 评估。"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, ParameterGrid, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from inspect_dataset import detect_target, inspect, load_tables


def save_json(path, data):
    def convert(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=convert,
                               allow_nan=False), encoding='utf-8')


def make_preprocessor(numeric, categorical, scaled):
    numeric_steps = [('impute', SimpleImputer(strategy='median'))]
    if scaled:
        numeric_steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical),
    ], remainder='drop')


def evaluate(y, prediction, probability, negative, positive):
    # yes/no 以外的 binary labels 也按传入的 positive label 排序。
    labels = [negative, positive]
    report = classification_report(y, prediction, labels=labels, output_dict=True,
                                   zero_division=0)
    result = {
        'accuracy': float(accuracy_score(y, prediction)),
        'macro_f1': float(f1_score(y, prediction, labels=labels, average='macro')),
        'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
        'confusion_matrix': confusion_matrix(y, prediction, labels=labels).tolist(),
        'per_class': {str(cls): {key: float(report[cls][key]) for key in
                                 ('precision', 'recall', 'f1-score', 'support')}
                      for cls in labels},
        'zero_predicted_positive': bool((prediction == positive).sum() == 0),
    }
    if probability is not None:
        binary_y = (y == positive).astype(int)
        result['roc_auc'] = float(roc_auc_score(binary_y, probability))
        result['average_precision'] = float(average_precision_score(binary_y, probability))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target')
    parser.add_argument('--positive')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError('此目录已有 metrics.json；请使用新的 output directory')
    tables, hashes = load_tables(args.dataset)
    target = detect_target(next(iter(tables.values())), args.target)
    profile = inspect(tables, target)
    profile['input_hashes'] = hashes
    profile['dataset_sha256'] = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    save_json(out/'profile.json', profile)
    if profile.get('cross_split_duplicate_feature_hashes', 0):
        raise ValueError('train/test 有相同 feature rows；需先判断实体重复风险')
    if any('ID/time/post-outcome' in warning for warning in profile['warnings']):
        raise ValueError('存在 ID/time/post-outcome 候选；需确认预测时点和 split')

    if 'train' in tables:
        raw_train, raw_test = tables['train'], tables['test']
        split_description = 'provided train/test; train-only StratifiedKFold'
        original_ids = {'train': raw_train.index, 'test': raw_test.index}
    else:
        full = tables['all']
        if full[target].isna().any():
            raise ValueError('单表含缺失 target；请先决定无标签行如何管理')
        indices = np.arange(len(full))
        tr_idx, te_idx = train_test_split(indices, test_size=.2, stratify=full[target],
                                          random_state=args.seed)
        raw_train, raw_test = full.iloc[tr_idx].copy(), full.iloc[te_idx].copy()
        split_description = '80/20 stratified split; train-only StratifiedKFold'
        original_ids = {'train': raw_train.index, 'test': raw_test.index}
    train = raw_train[raw_train[target].notna()].copy()
    test = raw_test[raw_test[target].notna()].copy()
    labels = sorted(str(v) for v in train[target].unique())
    if len(labels) != 2:
        raise ValueError('本执行器当前只支持 binary classification；多分类需调整 metrics')
    positive = args.positive or ('yes' if 'yes' in labels else None)
    if positive is None or positive not in labels:
        raise ValueError(f'请用 --positive 指定 positive class；可选：{labels}')
    negative = next(label for label in labels if label != positive)
    if set(test[target].astype(str)) != set(labels):
        raise ValueError('test 缺少训练中的类别，无法进行完整 binary evaluation')
    features = [col for col in train if col != target]
    numeric = [col for col in features if profile['feature_types'][col] == 'numeric']
    categorical = [col for col in features if col not in numeric]
    if profile['partitions']['train' if 'train' in tables else 'all']['constant_features']:
        raise ValueError('发现 constant feature；请检查是否应排除后再运行')
    X, y = train[features], train[target].astype(str)
    Xt, yt = test[features], test[target].astype(str)
    n_splits = min(3, int(y.value_counts().min()))
    if n_splits < 2:
        raise ValueError('最少类别样本不足以做 CV')
    cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=args.seed).split(X, y))
    fold_ids = np.full(len(train), -1)
    for fold, (_, valid) in enumerate(cv): fold_ids[valid] = fold
    assignments = []
    for part, frame in (('train', raw_train), ('test', raw_test)):
        for row in frame.index:
            included = bool(pd.notna(frame.at[row, target]))
            assignments.append({'row_id': f'{part}:{row}', 'partition': part,
                                'source_row_index': int(row), 'included': included,
                                'exclusion_reason': '' if included else 'missing_target',
                                'cv_validation_fold': int(fold_ids[train.index.get_loc(row)])
                                if part == 'train' and included else ''})
    pd.DataFrame(assignments).to_csv(out/'split_assignments.csv', index=False)

    # 三种 inductive bias：线性、非线性树集成、概率生成式；Dummy 单列 baseline。
    candidate_specs = {
        'LogisticRegression': (LogisticRegression(max_iter=2000, solver='lbfgs',
                                                 random_state=args.seed), True,
                               {'model__C': [.1, 1.], 'model__class_weight': [None, 'balanced']}),
        'RandomForest': (RandomForestClassifier(n_estimators=120, min_samples_leaf=3,
                                               n_jobs=2, random_state=args.seed), False,
                         {'model__max_depth': [12, None], 'model__class_weight': ['balanced']}),
        'GaussianNB': (GaussianNB(), False, {'model__var_smoothing': [1e-8, 1e-9]}),
    }
    scores, fitted, details, warning_log = {}, {}, [], []
    dummy = DummyClassifier(strategy='most_frequent')
    dummy_X = np.zeros((len(y), 1))
    dummy_scores = cross_val_score(dummy, dummy_X, y, cv=cv, scoring='f1_macro')
    dummy.fit(dummy_X, y)
    fitted['DummyClassifier'] = dummy
    scores['DummyClassifier'] = {'cv_macro_f1_mean': float(dummy_scores.mean()),
                                 'cv_macro_f1_std': float(dummy_scores.std()),
                                 'selected_parameters': {'strategy': 'most_frequent'},
                                 'candidate_count': 1}
    for name, (estimator, scale, grid) in candidate_specs.items():
        print(f'Training {name}: {len(ParameterGrid(grid))} configurations', flush=True)
        pipeline = Pipeline([('preprocess', make_preprocessor(numeric, categorical, scale)),
                             ('model', estimator)])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring='f1_macro', refit=True,
                              n_jobs=1, error_score='raise')
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
            warnings.simplefilter('always')
            search.fit(X, y)
        warning_log.extend({'model': name, 'category': item.category.__name__,
                            'message': str(item.message)} for item in caught)
        best = search.best_index_
        fitted[name] = search.best_estimator_
        scores[name] = {'cv_macro_f1_mean': float(search.best_score_),
                        'cv_macro_f1_std': float(search.cv_results_['std_test_score'][best]),
                        'selected_parameters': search.best_params_,
                        'candidate_count': len(search.cv_results_['params']),
                        'fit_seconds': time.perf_counter()-start,
                        'estimator_parameters': search.best_estimator_.named_steps['model'].get_params()}
        for index, params in enumerate(search.cv_results_['params']):
            details.append({'model': name, 'parameters': params,
                            'mean_macro_f1': float(search.cv_results_['mean_test_score'][index]),
                            'std_macro_f1': float(search.cv_results_['std_test_score'][index]),
                            'fold_scores': [float(search.cv_results_[f'split{i}_test_score'][index])
                                            for i in range(n_splits)]})
    save_json(out/'cv_results.json', details)
    # 先保存 train-only 的选择，再对冻结的 test 运行任何模型。
    winner = max(candidate_specs, key=lambda name: scores[name]['cv_macro_f1_mean'])
    save_json(out/'selection_before_test.json', {'selected_model': winner,
        'primary_metric': 'macro-F1', 'selection_basis': 'train-only CV',
        'candidate_cv': {name: scores[name]['cv_macro_f1_mean'] for name in candidate_specs},
        'created_utc': datetime.now(timezone.utc).isoformat()})
    predictions = pd.DataFrame({'row_id': [f'test:{idx}' for idx in test.index],
                                'true_label': yt.to_numpy()})
    for name, model in fitted.items():
        input_data = np.zeros((len(yt), 1)) if name == 'DummyClassifier' else Xt
        pred = model.predict(input_data)
        prob = (model.predict_proba(input_data)[:, list(model.classes_).index(positive)]
                if hasattr(model, 'predict_proba') else None)
        scores[name]['test'] = evaluate(yt, pred, prob, negative, positive)
        predictions[name+'_label'] = pred
        if prob is not None: predictions[name+'_p_positive'] = prob
    predictions.to_csv(out/'predictions.csv', index=False)
    save_json(out/'metrics.json', {'primary_metric': 'macro-F1', 'class_order': [negative, positive],
                                  'positive_class': positive, 'selected_by_cv': winner,
                                  'train_evaluable_rows': len(train), 'test_evaluable_rows': len(test),
                                  'models': scores})
    save_json(out/'warnings.json', warning_log)
    versions = {name: importlib.metadata.version(name) for name in
                ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl',
                 'reportlab', 'pypdf')}
    save_json(out/'run_manifest.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset_path': str(Path(args.dataset).resolve()),
        'dataset_sha256': profile['dataset_sha256'], 'input_hashes': hashes,
        'target': target, 'positive_label': positive, 'features': features,
        'numeric_features': numeric, 'categorical_features': categorical,
        'split': split_description, 'cv_folds': n_splits, 'seed': args.seed,
        'preprocessing': {'numeric': 'median imputation', 'categorical': 'constant __MISSING__ + OneHotEncoder',
                          'scaling': 'StandardScaler for LogisticRegression only',
                          'resampling': 'none'},
        'python': platform.python_version(), 'versions': versions, 'argv': sys.argv,
        'limitations': ['No data dictionary or entity/time metadata; feature availability is unverified.',
                        'No external validation or causal interpretation.']})
    (out/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',
                                         encoding='utf-8')
    print('Selected before test:', winner, 'warnings:', len(warning_log), flush=True)


if __name__ == '__main__':
    main()
