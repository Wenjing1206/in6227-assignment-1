"""可复用 classification 执行器：AI 决策保存在 JSON，数值结果来自真实训练。"""
import argparse, hashlib, json, platform, tempfile, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x,'item') else str(x)))


def read_table(path):
    # 格式由扩展名确定，CSV/TSV 的空白字符串统一按缺失值处理。
    ext=path.suffix.lower()
    if ext in ['.csv','.tsv']: d=pd.read_csv(path,sep='\t' if ext=='.tsv' else ',')
    elif ext=='.xlsx': d=pd.read_excel(path)
    elif ext=='.parquet': d=pd.read_parquet(path)
    else: raise ValueError('Unsupported format: '+ext)
    for col in d.select_dtypes(include=['object','string']).columns:
        d[col]=d[col].map(lambda x: x.strip() if isinstance(x,str) else x).replace('',np.nan)
    return d.replace([np.inf,-np.inf],np.nan)


def load_data(source):
    source=Path(source).expanduser().resolve(); temporary=None
    if source.suffix.lower()=='.zip':
        temporary=tempfile.TemporaryDirectory(); root=Path(temporary.name)
        with zipfile.ZipFile(source) as z:
            for item in z.infolist():
                dest=(root/item.filename).resolve()
                if not dest.is_relative_to(root): raise ValueError('Unsafe ZIP path')
            z.extractall(root)
    else: root=source
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in ['.csv','.tsv','.xlsx','.parquet'] and '__MACOSX' not in p.parts) if root.is_dir() else [root]
    trains=[p for p in files if p.stem.lower()=='train']; tests=[p for p in files if p.stem.lower()=='test']
    if len(trains)==1 and len(tests)<=1: train=trains[0]; test=tests[0] if tests else None
    elif len(files)==1: train=files[0]; test=None
    else: raise ValueError('Ambiguous files: explicitly provide one dataset or train/test pair')
    a=read_table(train); b=read_table(test) if test else None
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [train,test] if p}
    if temporary: temporary.cleanup()
    return a,b,sources


def profile(d):
    nums=d.select_dtypes(include='number'); flags={}
    for c in nums:
        q1,q3=nums[c].quantile([.25,.75]); span=q3-q1
        flags[c]=int(((nums[c]<q1-1.5*span)|(nums[c]>q3+1.5*span)).sum())
    return {'rows':len(d),'columns':len(d.columns),'dtypes':d.dtypes.astype(str).to_dict(),
        'missing':d.isna().sum().to_dict(),'unique':d.nunique().to_dict(),'duplicate_rows':int(d.duplicated().sum()),
        'iqr_flags':flags,'numeric_summary':nums.describe().to_dict(),
        'low_cardinality_counts':{c:d[c].value_counts(dropna=False).to_dict() for c in d if d[c].nunique()<=30}}


def score(y,p,prob,classes):
    out={'n':len(y),'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),
         'f1_macro':f1_score(y,p,labels=np.arange(len(classes)),average='macro',zero_division=0),
         'confusion_matrix':confusion_matrix(y,p,labels=np.arange(len(classes))).tolist(),
         'classification_report':classification_report(y,p,labels=np.arange(len(classes)),target_names=classes,output_dict=True,zero_division=0)}
    if len(classes)==2 and len(np.unique(y))==2:
        out['roc_auc']=roc_auc_score(y,prob[:,1]);out['average_precision']=average_precision_score(y,prob[:,1])
    return out


def execute(a,b,source,c,out):
    target=c['target']; seed=c.get('seed',42)
    if c['split_strategy']!='iid': raise ValueError('Implement group/time-aware evaluation before proceeding')
    if target not in a: raise ValueError('Target absent from train')
    raw_train=len(a); missing_train=int(a[target].isna().sum());a=a.loc[a[target].notna()].copy()
    if a[target].nunique()<2: raise ValueError('Need at least two classes')
    if a[target].value_counts().min()<2: raise ValueError('Insufficient observations in a class for reliable split')
    original_a=a.copy();original_b=None if b is None else b.copy()
    if b is None:
        a,b=train_test_split(a,test_size=.2,stratify=a[target],random_state=seed); split='internal stratified 80/20'
    else: split='provided train/test'
    drop=c.get('drop_columns',[])
    cols=[x for x in a if x!=target and x not in drop]
    if not cols: raise ValueError('No features remain')
    if set(cols)-set(b): raise ValueError('Missing test feature columns')
    X=a[cols]; Xt=b[cols]
    # 相同 feature 向量跨集合可能泄漏；遇到时停止，让 AI 审查其来源。
    overlaps=int(pd.util.hash_pandas_object(X,index=False).isin(pd.util.hash_pandas_object(Xt,index=False)).sum())
    if overlaps: raise ValueError(f'{overlaps} train rows have feature duplicates in test; investigate before evaluation')
    encoder=LabelEncoder().fit(a[target].astype(str)); classes=encoder.classes_.tolist(); y=encoder.transform(a[target].astype(str))
    mask=b[target].notna() if target in b else pd.Series(False,index=b.index)
    yt=encoder.transform(b.loc[mask,target].astype(str)) if mask.any() else None
    folds=min(int(c.get('folds',3)),int(pd.Series(y).value_counts().min()))
    if folds<2: raise ValueError('Insufficient class count for CV')
    num=X.select_dtypes(include='number').columns.tolist();cat=[x for x in cols if x not in num]
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    records={}; fitted={}; all_warnings=[]
    specs=c['models']
    if len(specs)<2 or len({s['kind'] for s in specs})!=len(specs): raise ValueError('Choose at least two different classifiers')
    metric=c['primary_metric']
    if metric not in ['f1_macro','balanced_accuracy','accuracy']: raise ValueError('Unsupported selection metric')
    for spec in specs:
        kind=spec['kind']; steps=[('impute',SimpleImputer(strategy=c.get('numeric_imputation','median'),keep_empty_features=True))]
        if kind=='logistic_regression': steps.append(('scale',StandardScaler()))
        prep=ColumnTransformer([('numeric',Pipeline(steps),num),('category',Pipeline([
            ('impute',SimpleImputer(strategy=c.get('categorical_imputation','most_frequent'),fill_value='__MISSING__',keep_empty_features=True)),
            ('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=c.get('min_frequency',5),max_categories=c.get('max_categories',64)))]),cat)],sparse_threshold=1.0)
        weight=spec.get('class_weight')
        if kind=='logistic_regression': model=LogisticRegression(max_iter=2000,class_weight=weight,random_state=seed)
        elif kind in ['random_forest','extra_trees']:
            cls=RandomForestClassifier if kind=='random_forest' else ExtraTreesClassifier
            model=cls(n_estimators=spec.get('n_estimators',160),class_weight=weight,random_state=seed,n_jobs=1)
        else: raise ValueError('Unsupported classifier: '+kind)
        pipe=Pipeline([('preprocess',prep),('model',model)])
        search=GridSearchCV(pipe,{'model__'+k:v for k,v in spec.get('grid',{}).items()},scoring=metric,cv=cv,n_jobs=2,refit=True,error_score='raise')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');search.fit(X,y)
        all_warnings.extend(str(w.message) for w in caught)
        cv_rows=[]
        for i,params in enumerate(search.cv_results_['params']):
            cv_rows.append({'params':params,'mean':float(search.cv_results_['mean_test_score'][i]),'std':float(search.cv_results_['std_test_score'][i]),'fold_scores':[float(search.cv_results_[f'split{k}_test_score'][i]) for k in range(folds)]})
        records[kind]={'cv_best':search.best_score_,'best_params':search.best_params_,'cv_candidates':cv_rows,'reason':spec['reason']}
        fitted[kind]=search.best_estimator_
        print(kind,search.best_score_,search.best_params_,flush=True)
    winner=max(records,key=lambda k:records[k]['cv_best'])
    # 所有模型的配置和 winner 固定后，才计算 test 结果。
    predictions=pd.DataFrame({'source_row':b.index+2,'actual':b[target].astype('string') if target in b else pd.Series(pd.NA,index=b.index,dtype='string')})
    for kind,model in fitted.items():
        p=model.predict(Xt); prob=model.predict_proba(Xt)
        predictions[kind]=encoder.inverse_transform(p)
        for i,label in enumerate(classes): predictions[kind+'__p_'+label]=prob[:,i]
        records[kind]['test']=score(yt,p[mask.to_numpy()],prob[mask.to_numpy()],classes) if yt is not None else None
    dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(y),1)),y)
    dp=dummy.predict(np.zeros((len(b),1)));dprob=dummy.predict_proba(np.zeros((len(b),1)))
    baseline=score(yt,dp[mask.to_numpy()],dprob[mask.to_numpy()],classes) if yt is not None else None
    predictions.to_csv(out/'predictions.csv',index=False)
    joblib.dump({'pipeline':fitted[winner],'label_encoder':encoder,'features':cols},out/'selected_model.joblib')
    result={'split':split,'raw_train_rows':raw_train,'missing_train_target':missing_train,'train_rows':len(a),'test_rows':len(b),
        'test_labeled_rows':int(mask.sum()),'missing_test_target':int((~mask).sum()),'classes':classes,
        'train_class_counts':a[target].astype(str).value_counts().to_dict(),'test_class_counts':b.loc[mask,target].astype(str).value_counts().to_dict() if target in b else {},
        'numeric_features':num,'categorical_features':cat,'train_feature_missing':int(X.isna().sum().sum()),'test_feature_missing':int(Xt.isna().sum().sum()),
        'outlier_flags':profile(X)['iqr_flags'],'folds':folds,'seed':seed,'primary_metric':metric,'winner':winner,'models':records,'baseline':baseline,
        'overlap_rows':overlaps,'warnings':all_warnings,'source_sha256':source,'versions':{'python':platform.python_version(),'pandas':pd.__version__,'scikit_learn':sklearn.__version__},
        'decisions_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()}
    save(out/'metrics.json',result); save(out/'decisions.json',c)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['profile','run']);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--config');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);a,b,sources=load_data(args.data)
    save(out/'profile.json',{'train':profile(a),'test':profile(b) if b is not None else None,'source_sha256':sources})
    if args.action=='run':
        if not args.config: p.error('--config is required for run')
        execute(a,b,sources,json.loads(Path(args.config).read_text()),out)
if __name__=='__main__': main()
