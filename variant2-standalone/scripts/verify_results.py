"""用保存的 predictions 独立复算指标，防止报告引用错误。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    out = Path(parser.parse_args().run_dir)
    metrics = json.loads((out/'metrics.json').read_text())
    selected = json.loads((out/'selection_before_test.json').read_text())
    profile = json.loads((out/'profile.json').read_text())
    predictions = pd.read_csv(out/'predictions.csv')
    splits = pd.read_csv(out/'split_assignments.csv')
    labels = metrics['class_order']
    positive = metrics['positive_class']
    checks = []
    def check(label, condition):
        checks.append({'check': label, 'passed': bool(condition)})
        if not condition: raise AssertionError(label)
    check('row IDs unique', predictions.row_id.is_unique and splits.row_id.is_unique)
    check('test row IDs equal split assignments', set(predictions.row_id) == set(
        splits.loc[(splits.partition == 'test') & splits.included, 'row_id']))
    check('target exclusion count', int((~splits.included).sum()) == sum(
        part['missing_target_rows'] for part in profile['partitions'].values()))
    check('selection frozen before test', selected['selected_model'] == metrics['selected_by_cv'])
    check('winner is highest formal CV score', selected['selected_model'] == max(
        selected['candidate_cv'], key=selected['candidate_cv'].get))
    check('CV fold assignments complete', splits.loc[(splits.partition=='train') & splits.included,
        'cv_validation_fold'].notna().all())
    y = predictions.true_label.astype(str)
    for name, item in metrics['models'].items():
        pred = predictions[name+'_label'].astype(str)
        expected = item['test']
        cm = confusion_matrix(y, pred, labels=labels)
        check(name+' confusion matrix', cm.tolist() == expected['confusion_matrix'])
        check(name+' confusion total', int(cm.sum()) == len(predictions))
        calculated = {'accuracy': accuracy_score(y,pred),
                      'macro_f1': f1_score(y,pred,labels=labels,average='macro'),
                      'balanced_accuracy': balanced_accuracy_score(y,pred)}
        score_col = name+'_p_positive'
        if score_col in predictions:
            score = predictions[score_col]
            binary_y = (y == positive).astype(int)
            calculated['roc_auc'] = roc_auc_score(binary_y,score)
            calculated['average_precision'] = average_precision_score(binary_y,score)
        for key, value in calculated.items():
            check(name+' '+key, np.isclose(value, expected[key], atol=1e-12, rtol=0))
        for index, cls in enumerate(labels):
            support = int((y == cls).sum())
            recall = cm[index,index]/support if support else 0
            check(name+' '+cls+' support', support == expected['per_class'][cls]['support'])
            check(name+' '+cls+' recall', np.isclose(recall,expected['per_class'][cls]['recall'],atol=1e-12))
    result = {'passed': len(checks), 'checks': checks}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('All',len(checks),'verification checks passed')


if __name__ == '__main__': main()
