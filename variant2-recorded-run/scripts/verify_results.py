"""从保存的逐行预测独立复算指标，不重新训练或调整模型。"""
import argparse
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, average_precision_score, classification_report)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-dir', required=True)
    out = Path(ap.parse_args().run_dir)
    metrics = json.loads((out/'metrics.json').read_text())
    manifest = json.loads((out/'run_manifest.json').read_text())
    predictions = pd.read_csv(out/'predictions.csv')
    splits = pd.read_csv(out/'split_assignments.csv')
    checks = []
    def check(name, condition):
        checks.append({'check': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)
    with zipfile.ZipFile(manifest['dataset_path']) as z:
        raw_test = pd.read_csv(io.BytesIO(z.read('dataset/test.csv')),
                              keep_default_na=False, na_values=[''])
    test = raw_test.loc[raw_test.label.notna()]
    check('prediction row IDs match original evaluable test rows',
          predictions.row_id.tolist() == ['test:'+str(i) for i in test.index])
    check('saved true labels match original CSV', predictions.true_label.tolist() == test.label.tolist())
    check('all split row IDs are unique', not splits.row_id.duplicated().any())
    check('four missing targets excluded', int((~splits.included).sum()) == 4)
    check('train CV assignment complete', set(splits.loc[(splits.partition=='train') & splits.included,
                                                       'cv_validation_fold']) == {0., 1., 2.})
    y = (predictions.true_label == 'yes').astype(int)
    for name, entry in metrics['models'].items():
        p = (predictions[name+'_label']=='yes').astype(int)
        score = predictions[name+'_p_yes']
        expected = entry['test']
        computed = {'accuracy': accuracy_score(y,p), 'macro_f1': f1_score(y,p,average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y,p), 'roc_auc': roc_auc_score(y,score),
            'average_precision': average_precision_score(y,score)}
        for key, val in computed.items():
            check(name+' '+key+' recomputation', np.isclose(val,expected[key], atol=1e-12,rtol=0))
        cm = confusion_matrix(y,p,labels=[0,1])
        check(name+' confusion matrix', cm.tolist()==expected['confusion_matrix'])
        check(name+' confusion matrix total', int(cm.sum())==len(test))
        report = classification_report(y,p,labels=[0,1],target_names=['no','yes'],
                                       output_dict=True,zero_division=0)
        for cls in ['no','yes']:
            for key in ['precision','recall','f1-score','support']:
                check(name+' '+cls+' '+key, np.isclose(report[cls][key],
                    expected['classification_report'][cls][key],atol=1e-12,rtol=0))
    selected = json.loads((out/'selection_before_test.json').read_text())
    check('final recommendation matches pre-test selection',
          selected['selected_model']==metrics['selected_by_cv'])
    profile = json.loads((out/'profile.json').read_text())
    check('no exact features shared between train/test', profile['cross_split_feature_hash_overlap']==0)
    (out/'verification.json').write_text(json.dumps({'passed':len(checks), 'checks':checks},indent=2))
    print('All',len(checks),'verification checks passed.')


if __name__=='__main__':
    main()
