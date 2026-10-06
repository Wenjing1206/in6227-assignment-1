"""复现本次分类实验：保留给定 split，仅在 train 内选择参数。"""
import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import plistlib
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=convert, allow_nan=False), encoding='utf-8')


def profile_frame(frame):
    fields = {}
    for col in frame:
        s = frame[col]
        item = {'dtype': str(s.dtype), 'missing': int(s.isna().sum()),
                'unique_nonmissing': int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
            item.update(minimum=float(s.min()), maximum=float(s.max()),
                median=float(s.median()), q1=float(q1), q3=float(q3),
                iqr_flag_count=int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()),
                nonfinite_nonmissing=int((~np.isfinite(s.dropna())).sum()))
        else:
            item['counts'] = {str(k): int(v) for k, v in s.value_counts().items()}
        fields[col] = item
    return {'rows': len(frame), 'columns': len(frame.columns),
            'duplicate_excess': int(frame.duplicated().sum()), 'fields': fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', default='label')
    parser.add_argument('--positive', default='yes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError('请使用新的 output directory，避免覆盖已有实验结果。')
    archive = Path(args.dataset)
    with zipfile.ZipFile(archive) as z:
        raw = {k: z.read('dataset/'+k+'.csv') for k in ['train', 'test']}
    # 只把空字符串当成缺失，保留 Unknown 和 Other 作为原有类别。
    frames = {k: pd.read_csv(io.BytesIO(v), keep_default_na=False, na_values=[''])
              for k, v in raw.items()}
    train, test = frames['train'], frames['test']
    assert list(train.columns) == list(test.columns), 'train/test schema 不一致'
    assert args.target in train
    prof = {k: profile_frame(v) for k, v in frames.items()}
    features = [c for c in train if c != args.target]
    prof['cross_split_full_row_hash_overlap'] = len(set(pd.util.hash_pandas_object(train, index=False)) & set(pd.util.hash_pandas_object(test, index=False)))
    prof['cross_split_feature_hash_overlap'] = len(set(pd.util.hash_pandas_object(train[features], index=False)) & set(pd.util.hash_pandas_object(test[features], index=False)))
    if prof['cross_split_feature_hash_overlap']:
        raise ValueError('发现跨 split 相同 features，需先评估重复实体风险。')
    write_json(out/'profile.json', prof)
    # 缺失 target 无法监督训练或计算评估分数，排除并保留 row ID 审计记录。
    valid_train = train[args.target].notna(); valid_test = test[args.target].notna()
    tr = train.loc[valid_train].copy(); te = test.loc[valid_test].copy()
    classes = sorted(tr[args.target].unique().tolist())
    assert len(classes) == 2 and args.positive in classes
    assert set(te[args.target]) <= set(classes)
    negative = next(c for c in classes if c != args.positive)
    X, Xt = tr[features], te[features]
    y = (tr[args.target] == args.positive).astype(int)
    yt = (te[args.target] == args.positive).astype(int)
    num = X.select_dtypes(include='number').columns.tolist()
    cat = [c for c in features if c not in num]
    cv = list(StratifiedKFold(n_splits=3, shuffle=True,
                            random_state=args.seed).split(X, y))
    # 保存每行 split、排除理由和内部 CV validation fold，方便核验。
    fold_ids = np.full(len(tr), -1)
    for i, (_, validation) in enumerate(cv):
        fold_ids[validation] = i
    split_rows = []
    for name, frame in frames.items():
        valid = frame[args.target].notna()
        for idx in frame.index:
            split_rows.append({'row_id': f'{name}:{idx}', 'source_file': name+'.csv',
                'source_csv_line': int(idx)+2, 'partition': name,
                'included': bool(valid.loc[idx]),
                'exclusion_reason': '' if valid.loc[idx] else 'missing_target',
                'cv_validation_fold': int(fold_ids[tr.index.get_loc(idx)])
                    if name == 'train' and valid.loc[idx] else ''})
    pd.DataFrame(split_rows).to_csv(out/'split_assignments.csv', index=False)

    def preprocess(scale):
        numeric = [('imputer', SimpleImputer(strategy='median'))]
        if scale:
            numeric.append(('scaler', StandardScaler()))
        return ColumnTransformer([
            ('numeric', Pipeline(numeric), num),
            ('categorical', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
        ], remainder='drop')

    models = {
        'LogisticRegression': (
            Pipeline([('preprocess', preprocess(True)),
                      ('model', LogisticRegression(solver='lbfgs', max_iter=2000,
                                                   random_state=args.seed))]),
            {'model__C': [.1, 1.], 'model__class_weight': [None, 'balanced']}),
        'RandomForest': (
            Pipeline([('preprocess', preprocess(False)),
                      ('model', RandomForestClassifier(n_estimators=160,
                          min_samples_leaf=3, n_jobs=2, random_state=args.seed))]),
            {'model__max_depth': [12, None], 'model__class_weight': [None, 'balanced']})}
    entries, fitted, warning_log, cv_detail = {}, {}, [], []
    dummy = DummyClassifier(strategy='most_frequent')
    start = time.perf_counter()
    d_scores = cross_val_score(dummy, np.zeros((len(y), 1)), y, cv=cv, scoring='f1_macro')
    dummy.fit(np.zeros((len(y), 1)), y)
    fitted['DummyClassifier'] = dummy
    entries['DummyClassifier'] = {'cv_macro_f1_mean': float(d_scores.mean()),
        'cv_macro_f1_std': float(d_scores.std()), 'cv_fold_scores': d_scores.tolist(),
        'selected_parameters': dummy.get_params(), 'fit_seconds': time.perf_counter()-start}
    for name, (pipe, grid) in models.items():
        print('Training', name, '4 candidates x 3 folds', flush=True)
        start = time.perf_counter()
        search = GridSearchCV(pipe, grid, scoring='f1_macro', cv=cv,
                              n_jobs=1, refit=True, error_score='raise', return_train_score=False)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
            warnings.simplefilter('always')
            search.fit(X, y)
        warning_log.extend({'model': name, 'category': w.category.__name__,
                            'message': str(w.message)} for w in captured)
        result = search.cv_results_; best = search.best_index_
        entries[name] = {'cv_macro_f1_mean': float(search.best_score_),
            'cv_macro_f1_std': float(result['std_test_score'][best]),
            'cv_fold_scores': [float(result[f'split{i}_test_score'][best]) for i in range(3)],
            'selected_parameters': search.best_params_,
            'estimator_parameters': search.best_estimator_.named_steps['model'].get_params(),
            'search_grid': grid, 'fit_seconds': time.perf_counter()-start}
        for i, params in enumerate(result['params']):
            cv_detail.append({'model': name, 'parameters': params,
                'mean_macro_f1': float(result['mean_test_score'][i]),
                'std_macro_f1': float(result['std_test_score'][i]),
                'fold_scores': [float(result[f'split{j}_test_score'][i]) for j in range(3)]})
        fitted[name] = search.best_estimator_
        entries[name]['transformed_feature_count'] = len(fitted[name].named_steps['preprocess'].get_feature_names_out())
        if name == 'LogisticRegression':
            entries[name]['final_n_iter'] = fitted[name].named_steps['model'].n_iter_.tolist()
        print(name, entries[name], flush=True)

    # 在读取任何 test 性能之前，按 CV 冻结最终推荐模型。
    winner = max(models, key=lambda name: entries[name]['cv_macro_f1_mean'])
    write_json(out/'selection_before_test.json', {'selection_basis': 'train-only CV macro-F1',
        'selected_model': winner, 'cv_results': entries, 'time_utc': datetime.now(timezone.utc).isoformat()})
    preds = pd.DataFrame({'row_id': ['test:'+str(i) for i in te.index],
                         'source_csv_line': te.index+2, 'true_label': te[args.target].to_numpy()})
    for name, model in fitted.items():
        data = np.zeros((len(yt), 1)) if name == 'DummyClassifier' else Xt
        pred = model.predict(data)
        probability = model.predict_proba(data)[:, list(model.classes_).index(1)]
        report = classification_report(yt, pred, labels=[0, 1], target_names=[negative, args.positive],
                                       output_dict=True, zero_division=0)
        entries[name]['test'] = {'accuracy': accuracy_score(yt, pred),
            'macro_f1': f1_score(yt, pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(yt, pred),
            'roc_auc': roc_auc_score(yt, probability),
            'average_precision': average_precision_score(yt, probability),
            'confusion_matrix': confusion_matrix(yt, pred, labels=[0, 1]).tolist(),
            'classification_report': report,
            'no_predicted_positive': bool(np.sum(pred) == 0)}
        assert sum(sum(row) for row in entries[name]['test']['confusion_matrix']) == len(yt)
        preds[name+'_label'] = np.where(pred == 1, args.positive, negative)
        preds[name+'_p_yes'] = probability
    preds.to_csv(out/'predictions.csv', index=False)
    write_json(out/'metrics.json', {'primary_metric': 'macro-F1', 'class_order': [negative, args.positive],
        'positive_class': args.positive, 'selected_by_cv': winner,
        'train_evaluable_rows': len(y), 'test_evaluable_rows': len(yt), 'models': entries})
    write_json(out/'cv_results.json', cv_detail)
    write_json(out/'warnings.json', warning_log)
    versions = {package: importlib.metadata.version(package) for package in
        ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl',
         'cloudpickle', 'narwhals', 'reportlab', 'pypdf', 'pypdfium2']}
    (out/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k, v in versions.items())+'\n')
    interface = {'name': 'Codex desktop Agent Harness', 'version': 'unknown'}
    app_plist = Path('/Applications/Codex.app/Contents/Info.plist')
    if app_plist.exists():
        info = plistlib.loads(app_plist.read_bytes())
        interface.update(version=info.get('CFBundleShortVersionString', 'unknown'),
                         build=info.get('CFBundleVersion', 'unknown'))
    write_json(out/'run_manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_path': str(archive.resolve()), 'dataset_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'input_hashes': {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        'read_options': {'encoding': 'utf-8', 'keep_default_na': False, 'na_values': ['']},
        'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions,
        'seed': args.seed, 'target': args.target, 'positive_label': args.positive,
        'numeric_features': num, 'categorical_features': cat,
        'split': 'provided train/test; 3-fold StratifiedKFold inside train',
        'argv': sys.argv, 'llm': {'family': 'GPT-6 (session identity)',
            'exact_model_version': 'unknown; user verification required'}, 'interface': interface,
        'limitations': ['No data dictionary or entity/time metadata supplied.',
            'Predictive availability of all features is an assumption, not verified.',
            'Aggregate test profile was inspected before fitting; no test performance used for selection.',
            'No external validation or causal interpretation.']})
    print('FINISHED; CV-selected model:', winner, 'warnings:', len(warning_log), flush=True)


if __name__ == '__main__':
    main()
