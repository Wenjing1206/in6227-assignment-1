"""只从本次保存的 evidence 生成两页以内的 PDF。"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--name', default='Pending student input')
    parser.add_argument('--matric', default='Pending student input')
    parser.add_argument('--repository', default='Pending repository URL')
    args = parser.parse_args()
    out = Path(args.run_dir)
    profile = json.loads((out/'profile.json').read_text())
    metrics = json.loads((out/'metrics.json').read_text())
    manifest = json.loads((out/'run_manifest.json').read_text())
    verify = json.loads((out/'verification.json').read_text())
    if not all(item['passed'] for item in verify['checks']):
        raise ValueError('Verification 未全部通过')
    selected = metrics['selected_by_cv']
    candidates = [name for name in metrics['models'] if name != 'DummyClassifier']
    train = profile['partitions'].get('train', profile['partitions'].get('all'))
    test = profile['partitions'].get('test')
    if test is None:
        test = {'rows': metrics['test_evaluable_rows'], 'missing_target_rows': 0,
                'missing_feature_cells': 0, 'duplicate_full_rows': 0}
    labels = metrics['class_order']
    positive = metrics['positive_class']
    styles = {
        'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=16.5,
                                leading=19, textColor=colors.HexColor('#17384D'), spaceAfter=7),
        'heading': ParagraphStyle('heading', fontName='Helvetica-Bold', fontSize=10.4,
                                  leading=12, textColor=colors.HexColor('#116C76'),
                                  spaceBefore=8, spaceAfter=3),
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=8.65,
                               leading=11.5, spaceAfter=4),
        'small': ParagraphStyle('small', fontName='Helvetica', fontSize=7.8,
                                leading=10.1, spaceAfter=3),
        'cell': ParagraphStyle('cell', fontName='Helvetica', fontSize=7.6,
                               leading=9.8),
        'head': ParagraphStyle('head', fontName='Helvetica-Bold', fontSize=7.6,
                               leading=9.8, textColor=colors.white),
    }
    story, markdown = [], []
    def line(value, style='body'):
        story.append(Paragraph(escape(str(value)), styles[style]))
        markdown.append(str(value)+'\n')
    def heading(value):
        story.append(Paragraph(escape(value), styles['heading']))
        markdown.append('## '+value+'\n')
    def table(headers, rows, widths):
        data = [[Paragraph(escape(str(cell)), styles['head' if row_index == 0 else 'cell'])
                 for cell in row] for row_index, row in enumerate([headers]+rows)]
        obj = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
        obj.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17384D')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EEF4F5'),colors.white]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),5), ('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4),
        ]))
        story.extend([obj,Spacer(1,4)])
        markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n')
        markdown.extend('| '+' | '.join(map(str,row))+' |\n' for row in rows)

    line(f'Matric number: {args.matric}    |    Full name: {args.name}', 'small')
    line('IN6227-Assignment-1 | Variant-2', 'title')
    line('Automated tabular classification workflow - draft pending student Reflection', 'small')
    heading('1. Dataset and evaluation design')
    counts = train['target_counts']
    yes_count = counts[positive]
    line(f'The supplied data contain {train["rows"]:,} train and {test["rows"]:,} test rows. '
         f'Target: {profile["target"]} ({labels[0]} / {positive}); {yes_count:,} of '
         f'{metrics["train_evaluable_rows"]:,} evaluable training rows are {positive} '
         f'({yes_count/metrics["train_evaluable_rows"]:.1%}). '
         f'There are {len(manifest["numeric_features"])} numeric and '
         f'{len(manifest["categorical_features"])} categorical features.')
    line(f'The provided train/test partition is retained. {train["missing_target_rows"]} train and '
         f'{test["missing_target_rows"]} test rows with missing target are excluded. '
         f'All formal models share {manifest["cv_folds"]}-fold stratified CV within training '
         'using seed 42. Macro-F1 is fixed as the primary selection metric because class '
         'imbalance makes accuracy alone misleading. Test is scored after CV selection.')
    heading('2. Inspection and preprocessing decisions')
    table(['Observation','Decision and reason'], [
        [f'{train["missing_feature_cells"]} train / {test["missing_feature_cells"]} test empty feature cells',
         'Numeric median imputation; categorical blanks become __MISSING__. Existing Unknown/Other values remain categories.'],
        [f'{train["duplicate_full_rows"]} duplicate train rows; imbalance ratio {train["imbalance_ratio"]:.2f}:1',
         'No row deletion or resampling. Compare class weighting in train-only CV; report per-class results.'],
        [f'{len(manifest["numeric_features"])} numeric / {len(manifest["categorical_features"])} categorical',
         'OneHotEncoder handles unseen categories. StandardScaler applies to numeric inputs for LogisticRegression only.']
    ],[155,352])
    line('Each learned transformation is inside ColumnTransformer and Pipeline, so CV folds fit '
         'their own imputation, encoding and scaling. All 15 features are retained; without a data '
         'dictionary, statistical tails are not assumed to be errors. No engineered features were '
         'added because their timing and domain meaning are unverified.', 'small')
    heading('3. Candidate models and training')
    descriptions = {
        'LogisticRegression':'Linear additive reference; C={0.1,1}, class_weight={None,balanced}, max_iter=2000.',
        'RandomForest':'Nonlinear interactions; 120 trees, min_samples_leaf=3, max_depth={12,None}, class_weight=balanced.',
        'GaussianNB':'Generative probabilistic contrast; var_smoothing={1e-8,1e-9}.',
    }
    table(['Model','Inductive bias and searched settings'],
          [[name,descriptions.get(name,str(metrics['models'][name]['selected_parameters']))]
           for name in candidates], [145,362])
    line('A most-frequent DummyClassifier is the majority-class baseline. GridSearchCV uses '
         'the same folds and macro-F1 for all formal models. No test-based tuning or threshold '
         'change was performed. Candidate counts and selected parameters are saved in metrics.json.', 'small')
    story.append(PageBreak()); markdown.append('\n--- PAGE BREAK ---\n')
    line('Evaluation and interpretation', 'title')
    heading('4. CV selection and held-out test results')
    rows = []
    for name, entry in metrics['models'].items():
        test_m = entry['test']
        rows.append([name, f'{entry["cv_macro_f1_mean"]:.4f} ± {entry["cv_macro_f1_std"]:.4f}',
                     f'{test_m["macro_f1"]:.4f}', f'{test_m["accuracy"]:.4f}',
                     f'{test_m["balanced_accuracy"]:.4f}'])
    table(['Model','CV macro-F1 ± SD','Test macro-F1','Accuracy','Balanced acc.'],
          rows,[130,112,95,80,90])
    selected_metric = metrics['models'][selected]['test']
    table(['Selected model / class','Precision','Recall','F1','Support'],
          [[cls]+[f'{selected_metric["per_class"][cls][key]:.4f}'
                  if key != 'support' else str(int(selected_metric['per_class'][cls][key]))
                  for key in ('precision','recall','f1-score','support')]
           for cls in labels],[170,84,84,84,85])
    matrix = selected_metric['confusion_matrix']
    line(f'CV selected {selected} before test scoring. Its confusion matrix (rows=true, '
         f'columns=predicted; order {labels}) is [[{matrix[0][0]}, {matrix[0][1]}], '
         f'[{matrix[1][0]}, {matrix[1][1]}]], total {sum(map(sum,matrix)):,}. '
         f'ROC-AUC={selected_metric["roc_auc"]:.4f}; average precision='
         f'{selected_metric["average_precision"]:.4f} for positive class {positive}.')
    line('CV SD is fold variation, not a confidence interval; the best-candidate CV mean may be '
         'optimistic after selection. Dummy predicts only the majority class. Precision for an '
         'unpredicted class is recorded as 0 using zero_division=0.', 'small')
    heading('5. Findings and critical evaluation')
    rank = sorted(candidates,key=lambda name:metrics['models'][name]['cv_macro_f1_mean'],reverse=True)
    first, second = rank[:2]
    gap = metrics['models'][first]['cv_macro_f1_mean']-metrics['models'][second]['cv_macro_f1_mean']
    line(f'The leading two CV means differ by {gap:.8f}; this near-tie does not establish '
         f'a robust superiority for {first}. The frozen test values and per-class trade-offs '
         'should guide later decisions with domain error costs. No model was reselected using test results.')
    if 'RandomForest' in metrics['models'] and 'LogisticRegression' in metrics['models']:
        rf = metrics['models']['RandomForest']['test']['confusion_matrix']
        lr = metrics['models']['LogisticRegression']['test']['confusion_matrix']
        line(f'RandomForest finds {rf[1][1]-lr[1][1]:,} more positive cases than LogisticRegression '
             f'but incurs {rf[0][1]-lr[0][1]:,} more false positives. Choosing between these '
             'operating points requires a stated cost for missed positives and false alarms.')
    line('No data dictionary, entity IDs or timestamps were supplied. Feature availability at '
         'prediction time and entity independence cannot be proven. Results describe this split '
         'and are not causal or deployment claims.', 'small')
    heading('6. Reproducibility and metadata')
    line(f'{verify["passed"]} saved-result checks passed: split IDs, exclusion counts, frozen '
         'selection, confusion totals and metrics recomputation. This is agent/programmatic '
         'review, not student manual verification. Input SHA-256, predictions, CV candidates, '
         'parameters and package versions are retained locally.', 'small')
    line(f'Python {manifest["python"]}; scikit-learn {manifest["versions"]["scikit-learn"]}; '
         f'pandas {manifest["versions"]["pandas"]}; seed {manifest["seed"]}. '
         'LLM: GPT-6 (exact version unavailable in session); interface: Codex desktop Agent Harness '
         '(version unavailable).', 'small')
    line(f'GitHub skill: {args.repository}. Personal Reflection requires the student\'s actual '
         'oversight, critical evaluation and manual check; see Reflection-draft.md.', 'small')
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#C8D9DE'))
        canvas.line(44,38,551,38)
        canvas.setFont('Helvetica',7.5)
        canvas.setFillColor(colors.HexColor('#526574'))
        canvas.drawString(44,26,'IN6227 | Variant 2 | Draft pending student details')
        canvas.drawRightString(551,26,str(doc.page))
        canvas.restoreState()
    destination = out/'IN6227-Variant-2-Report.pdf'
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=44, rightMargin=44,
                      topMargin=36, bottomMargin=48, title='IN6227 Variant 2 classification report',
                      author=args.name).build(story,onFirstPage=footer,onLaterPages=footer)
    pages = len(PdfReader(destination).pages)
    if pages > 2:
        raise AssertionError(f'报告 {pages} 页，超过两页上限')
    (out/'report.md').write_text('\n'.join(markdown),encoding='utf-8')
    print('Generated',destination,'pages',pages)


if __name__ == '__main__': main()
