"""中文备注：从逐行 predictions 独立计算 confusion matrix 和指标，核对报告数据源。"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def verify(run):
    r=json.loads((run/'metrics.json').read_text()); d=pd.read_csv(run/'predictions.csv',dtype={'actual':'string'});d=d[d.actual.notna()]
    labels=r['classes']; results={}
    for model,m in r['models'].items():
        if m['test'] is None: continue
        cm=np.array([[int(((d.actual==a)&(d[model].astype(str)==b)).sum()) for b in labels] for a in labels])
        diagonal=np.diag(cm);actual=cm.sum(axis=1);predicted=cm.sum(axis=0)
        # F1 使用 2TP / (实际正例 + 预测正例)，不调用训练流程中的 sklearn metrics。
        f1=np.divide(2*diagonal,actual+predicted,out=np.zeros(len(labels),dtype=float),where=(actual+predicted)>0)
        recalls=np.divide(diagonal,actual,out=np.zeros(len(labels),dtype=float),where=actual>0)
        values={'accuracy':float(diagonal.sum()/cm.sum()),'f1_macro':float(f1.mean()),'balanced_accuracy':float(recalls[actual>0].mean())}
        assert cm.tolist()==m['test']['confusion_matrix']
        for k,v in values.items(): assert abs(v-m['test'][k])<1e-12,(model,k,v,m['test'][k])
        results[model]={'confusion_matrix':cm.tolist(),'independent_metrics':values,'status':'passed'}
    (run/'independent_verification.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();verify(a.run)
