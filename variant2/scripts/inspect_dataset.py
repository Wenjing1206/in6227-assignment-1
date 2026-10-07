"""读取表格并生成可审计的 dataset inspection；不修改原始数据。"""
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_tables(path):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix == '.zip':
        with zipfile.ZipFile(source) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith('.csv')]
            chosen = {}
            for part in ('train', 'test'):
                matches = [name for name in members if Path(name).stem.lower() == part]
                if len(matches) != 1:
                    raise ValueError(f'ZIP 中需要唯一的 {part}.csv；实际找到 {len(matches)} 个')
                chosen[part] = archive.read(matches[0])
        tables = {part: pd.read_csv(io.BytesIO(data), keep_default_na=False, na_values=[''])
                  for part, data in chosen.items()}
        hashes = {part: hashlib.sha256(data).hexdigest() for part, data in chosen.items()}
    elif suffix in ('.csv', '.tsv'):
        tables = {'all': pd.read_csv(source, sep='\t' if suffix == '.tsv' else ',',
                                     keep_default_na=False, na_values=[''])}
        hashes = {'all': hashlib.sha256(source.read_bytes()).hexdigest()}
    elif suffix in ('.xlsx', '.xls'):
        sheets = pd.read_excel(source, sheet_name=None)
        if len(sheets) != 1:
            raise ValueError(f'Excel 有 {len(sheets)} 个 sheets；请先指定唯一数据表')
        tables = {'all': next(iter(sheets.values()))}
        hashes = {'all': hashlib.sha256(source.read_bytes()).hexdigest()}
    elif suffix == '.parquet':
        tables = {'all': pd.read_parquet(source)}
        hashes = {'all': hashlib.sha256(source.read_bytes()).hexdigest()}
    else:
        raise ValueError(f'暂不支持 {suffix}；可使用 CSV、TSV、Excel、Parquet 或 train/test CSV ZIP')
    schemas = [list(table.columns) for table in tables.values()]
    if len({tuple(schema) for schema in schemas}) != 1:
        raise ValueError('train/test 字段不一致')
    if any(table.columns.duplicated().any() or table.empty for table in tables.values()):
        raise ValueError('数据包含重复字段名或空表')
    return tables, hashes


def detect_target(frame, requested=None):
    if requested:
        if requested not in frame:
            raise ValueError(f'target 字段不存在：{requested}')
        return requested
    candidates = [col for col in frame if re.fullmatch(r'(?i)(target|label|class|y)', str(col))]
    if len(candidates) != 1:
        raise ValueError(f'target 不唯一；候选字段：{candidates}。请用 --target 指定')
    return candidates[0]


def inspect(tables, target):
    result = {'target': target, 'partitions': {}, 'feature_types': {}, 'warnings': []}
    train = tables.get('train', tables.get('all'))
    classes = train[target].dropna().unique().tolist()
    if len(classes) < 2 or len(classes) > 20:
        raise ValueError(f'target 有 {len(classes)} 个类别；需确认分类语义与可评估性')
    if pd.api.types.is_numeric_dtype(train[target]):
        result['warnings'].append('数值型 target 的类别语义需要人工确认')
    for col in train.drop(columns=target):
        series = train[col]
        result['feature_types'][col] = ('numeric' if pd.api.types.is_numeric_dtype(series)
                                        else 'categorical')
        if re.search(r'(?i)(^id$|_id$|timestamp|date|post_|outcome|result)', str(col)):
            result['warnings'].append(f'{col}: 需要确认 ID/time/post-outcome 语义')
    for part, frame in tables.items():
        missing = {str(col): int(frame[col].isna().sum()) for col in frame if frame[col].isna().any()}
        counts = {str(k): int(v) for k, v in frame[target].value_counts(dropna=False).items()}
        valid = frame[target].dropna()
        shares = valid.value_counts(normalize=True)
        result['partitions'][part] = {
            'rows': len(frame), 'columns': len(frame.columns),
            'missing_by_column': missing, 'missing_feature_cells': int(frame.drop(columns=target).isna().sum().sum()),
            'missing_target_rows': int(frame[target].isna().sum()),
            'duplicate_full_rows': int(frame.duplicated().sum()),
            'duplicate_feature_rows': int(frame.drop(columns=target).duplicated().sum()),
            'constant_features': [str(c) for c in frame.drop(columns=target) if frame[c].nunique(dropna=False) <= 1],
            'target_counts': counts,
            'minority_share': float(shares.min()) if len(shares) else None,
            'imbalance_ratio': float(shares.max()/shares.min()) if len(shares) and shares.min() else None,
        }
    if 'train' in tables:
        a = pd.util.hash_pandas_object(tables['train'].drop(columns=target), index=False)
        b = pd.util.hash_pandas_object(tables['test'].drop(columns=target), index=False)
        result['cross_split_duplicate_feature_hashes'] = len(set(a) & set(b))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--target')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    tables, hashes = load_tables(args.dataset)
    target = detect_target(next(iter(tables.values())), args.target)
    profile = inspect(tables, target)
    profile['input_hashes'] = hashes
    profile['dataset_sha256'] = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'target': target, 'feature_types': profile['feature_types'],
                      'partitions': profile['partitions'], 'warnings': profile['warnings']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
