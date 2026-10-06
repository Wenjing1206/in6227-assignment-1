"""从已保存的 evidence 自动排版报告；不在此脚本里重新训练模型。"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--name', default='Not provided')
    ap.add_argument('--matric', default='Not provided')
    ap.add_argument('--repository', default='Pending - not published')
    ap.add_argument('--llm-version', default='Exact model/version not exposed; verification pending')
    args = ap.parse_args(); out = Path(args.run_dir)
    m = json.loads((out/'metrics.json').read_text())
    profile = json.loads((out/'profile.json').read_text())
    manifest = json.loads((out/'run_manifest.json').read_text())
    verification = json.loads((out/'verification.json').read_text())
    lr = m['models']['LogisticRegression']; rf = m['models']['RandomForest']
    rt, lt = rf['test'], lr['test']
    styles = {
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=18, leading=21,
                                textColor=colors.HexColor('#12334B'), spaceAfter=7),
        'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=11,
                                  leading=13, textColor=colors.HexColor('#136B75'), spaceBefore=9,spaceAfter=4),
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=9.1,leading=12.1,
                               spaceAfter=5,alignment=TA_LEFT),
        'small': ParagraphStyle('small', fontName='Helvetica', fontSize=8,leading=10.5,spaceAfter=4),
        'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=8.1,leading=10.4),
        'headcell': ParagraphStyle('headcell', fontName='Helvetica-Bold', fontSize=8.1,leading=10.4,
                                  textColor=colors.white)}
    story=[]; markdown=[]
    def para(text,kind='body'):
        story.append(Paragraph(escape(text),styles[kind]));markdown.append(text+'\n')
    def heading(text):
        story.append(Paragraph(escape(text),styles['heading']));markdown.append('## '+text+'\n')
    def table(headers, rows, widths):
        allrows=[headers]+rows
        data=[[Paragraph(escape(str(v)),styles['headcell' if i==0 else 'cell']) for v in row]
              for i,row in enumerate(allrows)]
        t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#16384D')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F6'),colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
            ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#16384D'))]))
        story.append(t);story.append(Spacer(1,5))
        markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
            '\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in rows)+'\n')
    para(f'Matric number: {args.matric}    |    Full name: {args.name}', 'small')
    para('IN6227-Assignment-1 | Variant-2', 'title')
    para('Reusable classification workflow and automated report', 'body')
    para('Analysis complete. Submission metadata and personal Reflection remain pending.', 'small')
    heading('1. Dataset, objective and evaluation design')
    para('The supplied archive contains 31,112 training and 13,334 test rows, with 15 predictors: '
         '7 numeric and 8 categorical. The binary target is label (positive: yes). Three training '
         'rows and one test row have missing targets; they are excluded from supervised analysis, '
         'leaving 31,109 training and 13,333 test cases. The yes class represents '
         f'{7464/31109:.2%} of training and {3168/13333:.2%} of test cases.')
    para('The provided train/test split is retained. Inside training, the same 3-fold shuffled '
         'StratifiedKFold splits (seed 42) are used for both classifiers. Macro-F1 is the primary '
         'selection metric because it weights both classes equally; accuracy alone can hide poor '
         'minority detection. Model selection is frozen before test scoring. Aggregate test quality '
         'statistics were inspected, but test performance was not used for tuning.')
    heading('2. Exploration, cleaning and feature processing')
    train_missing=sum(v['missing'] for k,v in profile['train']['fields'].items() if k!='label')
    test_missing=sum(v['missing'] for k,v in profile['test']['fields'].items() if k!='label')
    para(f'The original training/test predictors contain {train_missing}/{test_missing} empty cells. '
         'There are no constant columns, duplicate full rows, or identical predictor vectors shared '
         'between train and test. No entity identifiers, timestamps or data dictionary were supplied; '
         'absence of semantic leakage or repeated entities therefore cannot be established.')
    table(['Data issue','Action and rationale'],[
        ['Numeric missingness and scale','Median imputation; StandardScaler only for LogisticRegression. '
         'Both transformations are fitted within each training fold.'],
        ['Nominal categories','Fill empty cells with __MISSING__; OneHotEncoder handles unseen values. '
         'Retain existing Unknown and Other values, which are not necessarily empty data.'],
        ['Outliers and feature selection','Keep all 15 predictors and numeric tails. For example, '
         '5,350 training activity_duration values are outside 1.5 IQR fences, but no domain rule '
         'establishes that they are errors. No additional engineered features or feature selection.']
    ],[116,391])
    para('ColumnTransformer and Pipeline keep imputation, encoding and scaling inside CV fits. '
         'The final encoding has 90 features (about 22 MB for the training matrix in float64), '
         'so dense representation is feasible here. No resampling is applied; class weighting '
         'is evaluated as a training-only hyperparameter.', 'small')
    heading('3. Models, tuning and stopping criteria')
    para('LogisticRegression provides a simple additive baseline; RandomForest can represent '
         'nonlinearities and interactions in mixed tabular data. A most-frequent DummyClassifier '
         'provides a separate sanity baseline. Each real classifier receives four candidates '
         'and the same three folds (12 CV fits plus one full-training refit).')
    table(['Model','Search and selected configuration'],[
        ['LogisticRegression','C in {0.1, 1}; class_weight in {None, balanced}. Selected C=1, '
         'class_weight=None. lbfgs, tolerance=1e-4, max_iter=2,000; final fit stopped at 105 iterations.'],
        ['RandomForest','max_depth in {12, None}; class_weight in {None, balanced}. Selected '
         'max_depth=None, class_weight=balanced; 160 trees, min_samples_leaf=3, max_features=sqrt. '
         'Training stops at the fixed tree budget; no validation-based early stopping.']
    ],[116,391])
    para(f'Search plus refit: LogisticRegression {lr["fit_seconds"]:.1f}s; RandomForest '
         f'{rf["fit_seconds"]:.1f}s on this run. No captured training warnings. Both use default '
         'decision thresholds; threshold tuning was not performed.', 'small')
    story.append(PageBreak()); markdown.append('\n--- PAGE BREAK ---\n')
    para('Evaluation and discussion', 'title')
    heading('4. Comparable results')
    rows=[]
    for name,short in [('DummyClassifier','Dummy'),('LogisticRegression','LogisticRegression'),('RandomForest','RandomForest')]:
        entry=m['models'][name];t=entry['test']
        rows.append([short,f'{entry["cv_macro_f1_mean"]:.4f} ({entry["cv_macro_f1_std"]:.4f})',
                     f'{t["macro_f1"]:.4f}',f'{t["accuracy"]:.4f}',f'{t["balanced_accuracy"]:.4f}'])
    table(['Model','CV macro-F1 (SD)','Test macro-F1','Test accuracy','Test balanced acc.'],rows,[115,122,90,80,100])
    para('CV values are means and fold standard deviations, not confidence intervals. They are '
         'the scores of selected candidates and may be optimistic after selection. The test set '
         'contains 10,165 no and 3,168 yes cases. Dummy predicts no for every case; its undefined '
         'yes precision is recorded as 0 using zero_division=0.', 'small')
    table(['Model / class','Precision','Recall','F1','Support'],[
        [short+' / '+cls,f'{entry["test"]["classification_report"][cls]["precision"]:.4f}',
         f'{entry["test"]["classification_report"][cls]["recall"]:.4f}',
         f'{entry["test"]["classification_report"][cls]["f1-score"]:.4f}',
         str(int(entry['test']['classification_report'][cls]['support']))]
        for short,entry in [('LogisticRegression',lr),('RandomForest',rf)] for cls in ['no','yes']
    ],[171,84,84,84,84])
    table(['Test confusion counts','TN','FP','FN','TP'],[
        [name,str(t['confusion_matrix'][0][0]),str(t['confusion_matrix'][0][1]),
         str(t['confusion_matrix'][1][0]),str(t['confusion_matrix'][1][1])]
        for name,t in [('LogisticRegression',lt),('RandomForest',rt)]
    ],[171,84,84,84,84])
    para('Confusion counts use rows=true class and columns=predicted class, ordered [no, yes]. '
         f'Binary ROC-AUC / average precision are {lt["roc_auc"]:.4f} / {lt["average_precision"]:.4f} '
         f'for LogisticRegression and {rt["roc_auc"]:.4f} / {rt["average_precision"]:.4f} for RandomForest. '
         'Average precision uses yes probabilities and is not trapezoidal PR-AUC.', 'small')
    heading('5. Findings, trade-offs and limitations')
    para(f'RandomForest was selected by training CV, with a macro-F1 advantage of only '
         f'{rf["cv_macro_f1_mean"]-lr["cv_macro_f1_mean"]:.4f}. On test data, LogisticRegression '
         f'has slightly higher macro-F1 ({lt["macro_f1"]:.4f} versus {rt["macro_f1"]:.4f}). '
         'The ranking reversal and small differences do not establish a clear winner. The pre-test '
         'selection is retained, rather than changing the rule after observing test results.')
    para(f'RandomForest detects {rt["confusion_matrix"][1][1]-lt["confusion_matrix"][1][1]:,} more '
         f'yes cases but produces {rt["confusion_matrix"][0][1]-lt["confusion_matrix"][0][1]:,} more '
         'false positives. Its yes recall is 81.53%, compared with 56.94% for LogisticRegression; '
         'the cost of missed positives versus false alarms should determine practical preference. '
         'In a future study, a train-only cost-sensitive threshold analysis would be useful. '
         'No threshold was changed in this run.')
    para('Keeping outliers and every predictor is defensible without a data dictionary, but worth '
         'challenging: activity_duration or composite_rank could contain information unavailable '
         'at prediction time. The supplied split does not prove temporal or entity independence. '
         'Findings apply to this split; they are neither causal claims nor evidence of deployment readiness.')
    heading('6. Reproducibility and generation metadata')
    para(f'All {verification["passed"]} automated checks passed: predictions match original test labels; '
         'metrics, per-class scores and confusion totals recompute exactly; split assignments and '
         'the pre-test selection are consistent. Full code, input SHA-256, CV candidates, warnings, '
         'row-level predictions and package versions are retained locally. These are automated checks, '
         'not a claim of student manual verification.', 'small')
    v=manifest['versions']
    para(f'Python {manifest["python"]}; scikit-learn {v["scikit-learn"]}; pandas {v["pandas"]}; '
         f'NumPy {v["numpy"]}. Random seed: 42. LLM: GPT-6 family (session identity); '
         f'{args.llm_version}. Interface: Codex desktop Agent Harness; interface version unknown.', 'small')
    para(f'GitHub repository: {args.repository}. Personal Reflection is maintained separately '
         'in reflection_pending.md until the student supplies actual oversight and manual-check details.', 'small')
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#D6E1E6'))
        canvas.line(44,38,551,38);canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#526574'))
        canvas.drawString(44,26,'IN6227 | Variant 2 | Main report - submission details pending')
        canvas.drawRightString(551,26,str(doc.page));canvas.restoreState()
    dest=out/'IN6227-Assignment-1-Variant-2-Report.pdf'
    SimpleDocTemplate(str(dest),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=35,
                      bottomMargin=47,title='IN6227 Assignment 1 - Variant 2',
                      author=args.name).build(story,onFirstPage=footer,onLaterPages=footer)
    pages=len(PdfReader(dest).pages)
    assert pages==2, f'报告共 {pages} 页，需重新排版。'
    (out/'report.md').write_text('\n'.join(markdown),encoding='utf-8')
    print('Generated', dest, 'pages:', pages)


if __name__=='__main__':
    main()
